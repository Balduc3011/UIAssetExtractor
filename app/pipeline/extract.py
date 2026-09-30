"""Step 3 – local extraction: matte, remove occluders, remove text, trim, 9-slice, dedupe."""
import time

import cv2
import numpy as np
from PIL import Image

from .. import store
from ..config import load_settings
from ..gpu import models
from .imgutil import background_estimate, inter_frac, pad_box, trim_alpha

PAD = 3


def border_distance(crop: np.ndarray, ring=2, k=4) -> np.ndarray:
    """Per-pixel Lab distance to the nearest backdrop colour (sampled on the border)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    edge = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
                           lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    k = min(k, len(edge))
    _, _, centers = cv2.kmeans(edge, k, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               2, cv2.KMEANS_PP_CENTERS)
    d = np.min(np.linalg.norm(lab[:, :, None, :] - centers[None, None], axis=3), axis=2)
    return d


def matte_grabcut(crop: np.ndarray, bgcrop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return np.full((h, w), 255, np.uint8)
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    # backdrop colour model sampled from the crop border ring
    diff = border_distance(crop)
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 50, 140),
                       np.ones((3, 3), np.uint8))
    mask[(diff < 10) & (edges == 0)] = cv2.GC_PR_BGD
    b = 2
    mask[:b, :] = cv2.GC_BGD
    mask[-b:, :] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD
    mask[:, -b:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.full((h, w), 255, np.uint8)
    a = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean_alpha(a, diff)


def _clean_alpha(a: np.ndarray, diff: np.ndarray | None = None) -> np.ndarray:
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), 8)
    if n > 2:
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < max(12, 0.02 * biggest):
                a[lab == i] = 0
    # fill small holes
    inv = (a == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = a.shape
    for i in range(1, n):
        x, y, ww, hh, ar = stats[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if touches:
            continue
        region = lab == i
        # enclosed hole: fill if small, or if it doesn't look like the backdrop
        if ar < 0.01 * h * w or (diff is not None and float(np.median(diff[region])) > 16):
            a[region] = 255
    # soft 1px edge
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def matte_ai(crop: np.ndarray, s) -> np.ndarray:
    from rembg import remove
    sess = models.rembg(s["ai_matte_model"], s["gpu_mode"])
    out = remove(Image.fromarray(crop), session=sess, only_mask=True)
    return np.array(out.convert("L"))


def remove_text(rgb: np.ndarray, alpha: np.ndarray, boxes_local: list):
    """Erase glyphs. On a plate: paint strokes with the plate's fill colour.
    On a transparent area: make the strokes transparent. Returns (rgb, alpha)."""
    if not boxes_local:
        return rgb, alpha
    h, w = rgb.shape[:2]
    out = rgb.copy()
    a_out = alpha.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    full = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes_local:
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        box = np.zeros((h, w), bool)
        box[y0:y1, x0:x1] = True
        ring = np.zeros((h, w), bool)
        ring[max(0, y0 - 3):min(h, y1 + 3), max(0, x0 - 3):min(w, x1 + 3)] = True
        ring[y0:y1, x0:x1] = False
        # text floating over a transparent area (e.g. inside a frame)?
        see_through = (alpha[box] < 60).mean() > 0.25
        px = lab[box & (alpha > 128)]
        if len(px) < 8:
            continue
        # two colour clusters inside the box: the larger one is the plate fill
        _, lbl, cen = cv2.kmeans(px, 2, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
        big = int(np.argmax(np.bincount(lbl.ravel(), minlength=2)))
        fill = cen[big]
        d = np.linalg.norm(lab - fill, axis=2)
        raw = box & (d > 20)
        stroke = raw.astype(np.uint8) * 255
        stroke = cv2.dilate(stroke, np.ones((3, 3), np.uint8), iterations=2)
        stroke &= (box * 255).astype(np.uint8)
        if see_through:
            # text floating over a transparent hole: drop text and its fill entirely
            a_out[box & (alpha > 0)] = 0
            continue
        # if "not fill colour" covers most of the box, the fill isn't flat (wood, gradient)
        textured = raw[box].mean() > 0.75
        if textured:
            stroke = (box & (d > 35)).astype(np.uint8) * 255
            stroke = cv2.dilate(stroke, np.ones((3, 3), np.uint8), iterations=2)
            # wood / gradient fill: reconstruct from surroundings instead of a flat colour
            out = cv2.inpaint(out, stroke, 5, cv2.INPAINT_TELEA)
            continue
        fill_rgb = cv2.cvtColor(fill.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        out[stroke > 0] = fill_rgb
        full |= stroke
    edge = cv2.dilate(full, np.ones((3, 3), np.uint8)) & ~cv2.erode(full, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(out, edge, 3, cv2.INPAINT_TELEA)
    return out, a_out


def fill_enclosed_occlusion(rgb, alpha, occl):
    """Holes left by overlays that are fully enclosed by the element (e.g. an icon sitting
    on a panel) are filled by inpainting instead of staying transparent."""
    hole = (alpha < 128).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole, 4)
    h, w = alpha.shape
    fillm = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, ww, hh, ar = stats[i]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        reg = lab == i
        if (occl[reg]).mean() > 0.5:
            fillm[reg] = 255
    if fillm.any():
        fillm = cv2.dilate(fillm, np.ones((3, 3), np.uint8))
        rgb = cv2.inpaint(rgb, fillm, 5, cv2.INPAINT_TELEA)
        alpha = alpha.copy()
        alpha[fillm > 0] = 255
    return rgb, alpha


def nine_slice(im: Image.Image, thr=6.0) -> dict | None:
    """Find stretchable middle band -> borders {l,t,r,b}."""
    a = np.array(im.convert("RGBA")).astype(np.float32)
    h, w = a.shape[:2]
    if w < 12 or h < 12:
        return None

    def band(arr, axis_len):
        c = axis_len // 2
        ref = arr[c]
        lo = c
        while lo > 1 and np.abs(arr[lo - 1] - ref).mean() < thr:
            lo -= 1
        hi = c
        while hi < axis_len - 2 and np.abs(arr[hi + 1] - ref).mean() < thr:
            hi += 1
        return lo, hi

    cols = np.transpose(a, (1, 0, 2))
    l, r = band(cols, w)
    t, b = band(a, h)
    res = {"l": int(l), "r": int(w - 1 - r), "t": int(t), "b": int(h - 1 - b)}
    if (r - l) < w * 0.1 and (b - t) < h * 0.1:
        return None
    return res


def extract_one(pid, proj, a, src, bg, s):
    W, H = src.shape[1], src.shape[0]
    x, y, w, h = pad_box(a["bbox"], PAD, W, H)
    crop = src[y:y + h, x:x + w].copy()
    d = store.pdir(pid)
    Image.fromarray(crop).save(d / "crops" / f"{a['id']}.png")

    method = a.get("method") or s["matte_method"]
    if a["type"] == "background":
        alpha = np.full((h, w), 255, np.uint8)
    elif method == "ai":
        alpha = matte_ai(crop, s)
    elif method == "none":
        alpha = np.full((h, w), 255, np.uint8)
        alpha[:PAD], alpha[-PAD:], alpha[:, :PAD], alpha[:, -PAD:] = 0, 0, 0, 0
    else:
        alpha = matte_grabcut(crop, bg[y:y + h, x:x + w])

    # remove pixels belonging to assets stacked on top (badges etc.)
    occl = np.zeros((h, w), bool)
    for o in proj["assets"]:
        if o["id"] == a["id"] or o["type"] in ("background", "text"):
            continue
        if inter_frac(a["bbox"], o["bbox"]) > 0.9:
            continue  # o contains a -> it is a parent, not an overlay
        on_top = o["z"] > a["z"] or (
            o.get("group") != a.get("group") and inter_frac(o["bbox"], a["bbox"]) > 0.9
            and o["bbox"][2] * o["bbox"][3] < 0.6 * a["bbox"][2] * a["bbox"][3])
        if not on_top:
            continue
        ox, oy, ow, oh = pad_box(o["bbox"], PAD, W, H)
        x0, y0 = max(ox - x, 0), max(oy - y, 0)
        x1, y1 = min(ox + ow - x, w), min(oy + oh - y, h)
        if x1 <= x0 or y1 <= y0:
            continue
        omask = _asset_alpha(pid, o)
        if omask is not None and omask.shape == (oh, ow):
            sub = omask[y0 + y - oy:y1 + y - oy, x0 + x - ox:x1 + x - ox] > 128
            occl[y0:y1, x0:x1] |= sub
        else:
            bx, by, bw, bh = o["bbox"]
            occl[max(by - y, 0):max(0, min(by + bh - y, h)), max(bx - x, 0):max(0, min(bx + bw - x, w))] = True
    Image.fromarray(alpha).save(d / "mattes" / f"{a['id']}.png")  # pre-occlusion matte
    occl = cv2.dilate(occl.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    fg = alpha > 128
    occ_ratio = float((occl & fg).sum() / max(1, fg.sum()))
    alpha[occl] = 0
    crop, alpha = fill_enclosed_occlusion(crop, alpha, occl)

    rgb = crop
    # text-only assets lying on this element are always erased (engine renders text);
    # OCR text inside the element is erased when remove_text is on
    text_assets = [o["bbox"] for o in proj["assets"]
                   if o["type"] == "text" and o["id"] != a["id"]
                   and inter_frac(o["bbox"], a["bbox"]) > 0.6]
    tboxes = (list(a["text_boxes"]) if a["remove_text"] else []) + \
        ([] if a["type"] == "text" else text_assets)
    if tboxes:
        local = [[bx - x, by - y, bw, bh] for bx, by, bw, bh in tboxes]
        rgb, alpha = remove_text(crop, alpha, local)

    im = Image.fromarray(np.dstack([rgb, alpha]))
    if a["type"] != "background":
        im, _ = trim_alpha(im)
    fn = f"{a['id']}_x{int(time.time() * 1000) % 10 ** 8}.png"
    im.save(d / "assets" / fn)
    return {"file": f"assets/{fn}", "kind": "extract", "w": im.width, "h": im.height,
            "scale": 1.0, "created": time.time()}, occ_ratio, (nine_slice(im) if a["nine_slice"] else None)


def _asset_alpha(pid, o):
    """Raw matte (padded-bbox sized) of another asset, if already extracted."""
    f = store.pdir(pid) / "mattes" / f"{o['id']}.png"
    if f.exists():
        return np.array(Image.open(f))
    return None


def run(pid: str, params: dict, ctx):
    s = load_settings()
    proj = store.load(pid)
    d = store.pdir(pid)
    (d / "mattes").mkdir(exist_ok=True)
    src = np.array(Image.open(d / "source.png").convert("RGB"))
    bg = background_estimate(src)
    ids = params.get("asset_ids")
    targets = [a for a in proj["assets"] if (not ids or a["id"] in ids)]
    if not ids:
        targets = [a for a in targets if a["type"] != "background"]
    # top-most first so their mattes exist when lower assets subtract them
    targets.sort(key=lambda a: -a["z"])
    for i, a in enumerate(targets):
        ctx.progress(i, len(targets), f"Trích xuất {a['name']} ({i + 1}/{len(targets)})")
        try:
            ver, occ, borders = extract_one(pid, proj, a, src, bg, s)

            def f(p, a=a, ver=ver, occ=occ, borders=borders):
                t = store.get_asset(p, a["id"])
                if not t:
                    return
                t["versions"].append(ver)
                t["active"] = len(t["versions"]) - 1
                t["status"] = "extracted"
                t["occluded"] = round(occ, 3)
                t["borders"] = borders
                t["error"] = None
            store.update(pid, f)
        except Exception as e:
            ctx.log(f"{a['name']}: {e}")

            def f(p, a=a, e=e):
                t = store.get_asset(p, a["id"])
                if t:
                    t["status"], t["error"] = "error", str(e)
            store.update(pid, f)
    ctx.progress(len(targets), len(targets), "Kiểm tra trùng lặp…")
    dedupe(pid)
    store.update(pid, lambda p: p.__setitem__("stage", "extracted"))
    ctx.progress(1, 1, f"Đã trích xuất {len(targets)} asset")


def dedupe(pid: str):
    import imagehash
    proj = store.load(pid)
    d = store.pdir(pid)
    seen = []
    dups = {}
    for a in proj["assets"]:
        if a["type"] == "background" or a["active"] is None:
            continue
        v = a["versions"][a["active"]]
        try:
            im = Image.open(d / v["file"]).convert("RGBA")
        except Exception:
            continue
        flat = Image.new("RGBA", im.size, (128, 128, 128, 255))
        flat.alpha_composite(im)
        hsh = imagehash.phash(flat.convert("RGB"), hash_size=8)
        ar = im.width / max(1, im.height)
        match = None
        for sid, sh, sar, ssz in seen:
            if hsh - sh <= 6 and abs(ar - sar) < 0.12 and 0.8 < (im.width / max(1, ssz)) < 1.25:
                match = sid
                break
        if match:
            dups[a["id"]] = match
        else:
            seen.append((a["id"], hsh, ar, im.width))

    def f(p):
        for a in p["assets"]:
            a["dup_of"] = dups.get(a["id"])
    store.update(pid, f)
