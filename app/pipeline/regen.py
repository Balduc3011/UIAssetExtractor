"""Step 5 – OpenAI image model: redraw an asset cleanly (transparent, no text, complete)."""
import time

import cv2
import numpy as np
from PIL import Image

from .. import store
from ..config import load_settings
from .ai import image_edit, pick_size
from .imgutil import has_real_alpha, pad_box, trim_alpha
from .qa import qa_assets

PROMPT = """Game UI sprite extraction. The reference image is a crop from a mobile game UI.
Redraw ONLY this element: "{name}" ({type}). {desc}
Art style: {style}
Requirements:
- match the reference exactly: same silhouette, proportions, colors, outlines, highlights and shading
- a single isolated element, centered, fully visible, nothing cut off{occl}
- transparent background, no backdrop, no cast shadow onto the background
{text}- no other UI elements, no badges or stickers unless they are part of this element
- crisp clean edges, high quality game asset{hint}"""

BG_PROMPT = """This is a mobile game UI screenshot. Repaint the masked areas so that ALL UI
elements (buttons, icons, panels, text, ads) are removed and only the clean backdrop remains:
{desc}. Continue the texture/pattern seamlessly. Style: {style}. No text, no UI."""


def _reference(src: Image.Image, a) -> tuple[Image.Image, str]:
    W, H = src.size
    x, y, w, h = pad_box(a["bbox"], max(4, int(0.08 * max(a["bbox"][2:]))), W, H)
    crop = src.crop((x, y, x + w, y + h)).convert("RGB")
    size = pick_size(w, h)
    tw, th = [int(v) for v in size.split("x")]
    s = min(tw / w, th / h) * 0.8
    crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (128, 128, 128))
    canvas.paste(crop, ((tw - crop.width) // 2, (th - crop.height) // 2))
    return canvas, size


def _knockout(im: Image.Image) -> Image.Image:
    """Model ignored transparency -> flood-fill backdrop from the corners."""
    arr = np.array(im.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for pt in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(arr.copy(), mask, pt, 0, (12, 12, 12), (12, 12, 12),
                      cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 8)
    alpha = np.where(mask[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return Image.fromarray(np.dstack([arr, alpha]))


def regen_one(pid, a, proj, src, hint=""):
    s = load_settings()
    style = proj.get("screen", {}).get("style", "") or "polished casual mobile game UI"
    d = store.pdir(pid)
    if a["type"] == "background":
        W, H = src.size
        size = pick_size(W, H)
        tw, th = [int(v) for v in size.split("x")]
        ref = src.convert("RGB").resize((tw, th), Image.LANCZOS)
        mask = Image.new("RGBA", (tw, th), (0, 0, 0, 255))
        m = np.array(mask)
        for o in proj["assets"]:
            if o["type"] == "background":
                continue
            x, y, w, h = pad_box(o["bbox"], 4, W, H)
            m[int(y * th / H):int((y + h) * th / H), int(x * tw / W):int((x + w) * tw / W), 3] = 0
        out = image_edit(pid, [ref], BG_PROMPT.format(desc=a["description"] or "backdrop",
                                                      style=style),
                         size, mask=Image.fromarray(m), transparent=False)
        out = out.resize((W * 2, H * 2), Image.LANCZOS)
        scale = 2.0
    else:
        ref, size = _reference(src, a)
        text = "- absolutely no text, letters or numbers anywhere on the element\n"
        occl = (" — reconstruct parts hidden by overlapping elements in the reference"
                if a.get("occluded", 0) > 0.01 else "")
        prompt = PROMPT.format(name=a["name"].replace("_", " "), type=a["type"],
                               desc=a["description"], style=style, occl=occl, text=text,
                               hint=f"\nFix from previous attempt: {hint}" if hint else "")
        out = image_edit(pid, [ref], prompt, size)
        if not has_real_alpha(out):
            out = _knockout(out)
        out, _ = trim_alpha(out)
        scale = out.width / max(1, a["bbox"][2])
    fn = f"{a['id']}_g{int(time.time() * 1000) % 10 ** 8}.png"
    out.save(d / "regen" / fn)
    return {"file": f"regen/{fn}", "kind": "regen", "w": out.width, "h": out.height,
            "scale": round(scale, 3), "created": time.time()}


def run(pid, params, ctx):
    s = load_settings()
    ids = params.get("asset_ids") or []
    if not ids:
        raise RuntimeError("Chọn ít nhất 1 asset để tạo lại.")
    src = Image.open(store.pdir(pid) / "source.png")
    retries = int(s["max_regen_retries"])
    th = float(s["qa_threshold"])
    for i, aid in enumerate(ids):
        proj = store.load(pid)
        a = store.get_asset(proj, aid)
        if not a:
            continue
        ctx.progress(i, len(ids), f"GPT vẽ lại {a['name']} ({i + 1}/{len(ids)})")
        hint = params.get("hint") or (a.get("qa") or {}).get("hint", "")
        attempt = 0
        while True:
            try:
                ver = regen_one(pid, a, proj, src, hint)
            except Exception as e:
                ctx.log(f"{a['name']}: {e}")
                if "giới hạn" in str(e):
                    raise
                break

            def f(p, ver=ver):
                t = store.get_asset(p, aid)
                t["versions"].append(ver)
                t["active"] = len(t["versions"]) - 1
                t["status"] = "regenerated"
            store.update(pid, f)
            if (not s["auto_qa_after_regen"] or a["type"] == "background"
                    or s.get("ai_provider") == "cowork"):
                break
            ctx.log(f"QA {a['name']}…")
            q = qa_assets(pid, [aid]).get(aid)
            if not q or q["score"] >= th or attempt >= retries:
                break
            attempt += 1
            hint = q.get("hint") or "; ".join(q.get("issues", []))
            ctx.log(f"{a['name']}: {q['score']}/10 → thử lại ({attempt}/{retries})")
            proj = store.load(pid)
            a = store.get_asset(proj, aid)
    ctx.progress(1, 1, f"Đã tạo lại {len(ids)} asset")
