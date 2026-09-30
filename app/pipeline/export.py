"""Step 6 – pack sprite sheet(s) + individual PNGs + JSON (TexturePacker 'JSON Hash')."""
import json
import time
import zipfile

import numpy as np
from PIL import Image

from .. import store
from ..config import load_settings
from . import refine


def asset_image(pid, a, scale: float) -> Image.Image:
    """Best version resized to `scale` × original screenshot size."""
    d = store.pdir(pid)
    v = a["versions"][a["active"]]
    im = Image.open(d / v["file"]).convert("RGBA")
    native = float(v.get("scale", 1.0)) or 1.0
    f = scale / native
    if abs(f - 1) > 0.01:
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    return im


def run(pid, params, ctx):
    from rectpack import newPacker, MaxRectsBssf, PackingBin, PackingMode

    s = load_settings()
    scale = float(params.get("scale") or s["export_scale"])
    maxsz = int(params.get("max_size") or s["atlas_max_size"])
    pad = int(s["atlas_padding"])
    skip_dup = bool(params.get("skip_duplicates", s["skip_duplicates"]))
    compact = bool(params.get("compact_nine_slice", s.get("compact_nine_slice", True)))
    proj = store.load(pid)
    items = [a for a in proj["assets"] if a["keep"] and a["active"] is not None
             and not (skip_dup and a["dup_of"])]
    if not items:
        raise RuntimeError("Không có asset nào để export (cần Keep + đã trích xuất).")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = store.pdir(pid) / "export" / stamp
    (out / "sprites").mkdir(parents=True, exist_ok=True)

    names = set()
    for a in items:
        n, i = a["name"], 2
        while n in names:
            n = f'{a["name"]}_{i}'
            i += 1
        names.add(n)
        a["name"] = n
    imgs = {}
    sheet_items = []
    for i, a in enumerate(items):
        ctx.progress(i, len(items) * 2, f"Chuẩn bị {a['name']}")
        im = asset_image(pid, a, scale)
        a["_borders"] = None
        if a["nine_slice"] and a["type"] != "background":
            arr = np.array(im)
            if compact:
                arr, b = refine.compact_nine_slice(arr)
                im = Image.fromarray(arr)
            else:
                b = refine.nine_slice(arr)
                b = b and {k: b[k] for k in ("l", "t", "r", "b")}
            a["_borders"] = b
        imgs[a["id"]] = im
        im.save(out / "sprites" / f"{a['name']}.png")
        if a["type"] != "background":
            sheet_items.append(a)

    too_big = [a for a in sheet_items if max(imgs[a["id"]].size) + 2 * pad > maxsz]
    sheet_items = [a for a in sheet_items if a not in too_big]
    frames_meta = {}
    atlases = []
    if sheet_items:
        rects_in = [(imgs[a["id"]].width + 2 * pad, imgs[a["id"]].height + 2 * pad, a["id"])
                    for a in sheet_items]

        def try_pack(bw, bh, nbins):
            pk = newPacker(mode=PackingMode.Offline, bin_algo=PackingBin.BFF,
                           pack_algo=MaxRectsBssf, rotation=False)
            for w, h, rid in rects_in:
                pk.add_rect(w, h, rid=rid)
            for _ in range(nbins):
                pk.add_bin(bw, bh)
            pk.pack()
            return pk

        packer = None
        size = 128
        while size <= maxsz and packer is None:   # smallest single power-of-two sheet
            for bw, bh in ((size, size // 2), (size, size)):
                pk = try_pack(bw, bh, 1)
                if len(pk.rect_list()) == len(rects_in):
                    packer = pk
                    break
            size *= 2
        if packer is None:
            packer = try_pack(maxsz, maxsz, 64)
        by_bin: dict[int, list] = {}
        for b, x, y, w, h, rid in packer.rect_list():
            by_bin.setdefault(b, []).append((x, y, w, h, rid))
        amap = {a["id"]: a for a in sheet_items}
        for bi, rects in sorted(by_bin.items()):
            W = max(x + w for x, y, w, h, _ in rects)
            H = max(y + h for x, y, w, h, _ in rects)
            W, H = _pow2(W), _pow2(H)
            sheet = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            frames = {}
            for x, y, w, h, rid in rects:
                a = amap[rid]
                im = imgs[rid]
                sheet.alpha_composite(im, (x + pad, y + pad))
                fr = {"frame": {"x": x + pad, "y": y + pad, "w": im.width, "h": im.height},
                      "rotated": False, "trimmed": False,
                      "spriteSourceSize": {"x": 0, "y": 0, "w": im.width, "h": im.height},
                      "sourceSize": {"w": im.width, "h": im.height},
                      "pivot": {"x": 0.5, "y": 0.5}}
                if a.get("_borders"):
                    fr["borders"] = a["_borders"]
                frames[f"{a['name']}.png"] = fr
            name = f"atlas_{bi}.png"
            sheet.save(out / name)
            meta = {"frames": frames,
                    "meta": {"app": "UIAssetExtractor", "version": "1.0", "image": name,
                             "format": "RGBA8888", "size": {"w": W, "h": H}, "scale": str(scale)}}
            (out / f"atlas_{bi}.json").write_text(json.dumps(meta, indent=1), "utf-8")
            atlases.append(name)
            frames_meta.update(frames)
            ctx.progress(len(items) + bi, len(items) * 2, f"Đóng gói {name}")

    manifest = {
        "project": proj["name"], "scale": scale, "atlases": atlases,
        "not_in_atlas": [a["name"] for a in too_big],
        "assets": [{"name": a["name"], "type": a["type"], "text": a["text"],
                    "source_bbox": a["bbox"], "nine_slice": a["nine_slice"],
                    "borders": a.get("_borders"), "file": f"sprites/{a['name']}.png"}
                   for a in items],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), "utf-8")
    zp = out.parent / f"{stamp}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))

    def f(p):
        p["exports"].insert(0, {"stamp": stamp, "zip": f"export/{stamp}.zip",
                                "folder": f"export/{stamp}", "atlases": atlases,
                                "count": len(items), "scale": scale})
        p["exports"] = p["exports"][:20]
        p["stage"] = "exported"
    store.update(pid, f)
    ctx.progress(1, 1, f"Export xong: {len(items)} sprite, {len(atlases)} atlas")


def _pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p
