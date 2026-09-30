"""Manual ChatGPT redraw: pack selected assets into ONE reference sheet + prompt,
the user redraws it in ChatGPT (app/web, no API), then the returned image is sliced
back into per-asset versions."""
import io
import math
import time

import cv2
import numpy as np
from PIL import Image

from .. import store
from .imgutil import has_real_alpha, trim_alpha

MAX_PER_SHEET = 9
CANVASES = [(1024, 1024), (1536, 1024), (1024, 1536)]   # ChatGPT image output sizes
REF_SCALE = 2                                           # reference sheet drawn at 2x
KEYS = [("magenta", (255, 0, 255)), ("pure green", (0, 255, 0)),
        ("pure blue", (0, 0, 255)), ("cyan", (0, 255, 255))]


# ---------------- layout ----------------

def _pack(aspects, CW, CH, h):
    """Justified rows at item height h. Returns rects [x,y,w,h] or None if it doesn't fit."""
    m = 0.4 * h
    rects, rows, x, y, row = [], [], m, m, []
    for ar in aspects:
        w = min(h * ar, CW - 2 * m)
        ih = w / ar
        if row and x + w + m > CW:
            rows.append(row)
            y += h + m
            x, row = m, []
        row.append([x, y, w, ih])
        x += w + m
    rows.append(row)
    if y + h + m > CH:
        return None
    # centre rows horizontally and the whole block vertically
    total_h = y + h + m
    dy = (CH - total_h) / 2
    for row in rows:
        right = row[-1][0] + row[-1][2] + m
        dx = (CW - right) / 2
        for r in row:
            r[0] += dx
            r[1] += dy + (h - r[3]) / 2
            rects.append(r)
    return rects


def layout(aspects):
    best = None
    for CW, CH in CANVASES:
        lo, hi = 8.0, float(max(CW, CH))
        for _ in range(40):
            mid = (lo + hi) / 2
            if _pack(aspects, CW, CH, mid):
                lo = mid
            else:
                hi = mid
        rects = _pack(aspects, CW, CH, lo)
        area = sum(r[2] * r[3] for r in rects)
        if best is None or area > best[0] * 1.05:
            best = (area, (CW, CH), rects, lo)
    return best[1], best[2], best[3]


def _pick_key(crops):
    px = np.concatenate([np.array(c.convert("RGB")).reshape(-1, 3) for c in crops]).astype(np.int32)
    if len(px) > 200000:
        px = px[np.random.default_rng(0).choice(len(px), 200000, replace=False)]
    best = None
    for name, col in KEYS:
        n = (np.linalg.norm(px - np.array(col), axis=1) < 110).sum()
        if best is None or n < best[0]:
            best = (n, name, col)
    return best[1], best[2]


# ---------------- build ----------------

def _orient(CW, CH):
    return "square 1:1" if CW == CH else ("landscape 3:2" if CW > CH else "portrait 2:3")


PROMPT = """The attached image is a sprite reference sheet from a mobile game UI: {n} separate UI elements \
laid out in rows on a transparent background. Redraw it as a clean, production-ready sprite sheet.

Rules:
- Output ONE {orient} image with EXACTLY the same layout: same number of elements, same order, same positions and \
same sizes as the reference. Do not add, remove, merge or move elements; keep the empty space between them.
- Output a PNG with a TRANSPARENT background (real alpha channel) around every element. No backdrop, \
no gradient, no checkerboard pattern, no shadows cast onto the background, no grid lines, no frames around cells. \
Only if a transparent background is impossible, use a flat solid {key} ({hexc}) background instead.
- Remove the dark game backdrop visible behind each reference element; draw only the element itself.
- If other UI pieces overlap an element in the reference (icons, labels, neighbouring buttons), leave them out \
and draw only the element described in its line below.
- Every element complete and fully visible, crisp clean edges, nothing cut off.
- No text, letters or numbers, except logo lettering that is marked "keep its logo lettering" below.
- Never use the {key} color inside an element.
- Match the reference art exactly: same silhouette, proportions, colors, outlines, highlights and shading.
- Art style: {style}

Elements, left to right, top to bottom:
{items}"""


def _item_line(i, a):
    t = f"{i}. {a['name'].replace('_', ' ')} ({a['type']}): {a.get('description') or ''}".strip()
    extra = []
    if a.get("occluded", 0) > 0.01:
        extra.append("reconstruct parts hidden by overlapping elements in the reference")
    if a.get("remove_text"):
        extra.append("remove all text/numbers drawn on it")
    elif "letter" in (a.get("description") or "").lower() or "logo" in (a.get("description") or "").lower():
        extra.append("keep its logo lettering exactly as in the reference")
    q = a.get("qa") or {}
    hint = q.get("hint") if q.get("recommend") == "regenerate" else ""
    if hint:
        extra.append(hint.rstrip("."))
    if extra:
        t += " — " + "; ".join(extra) + "."
    return t


def build(pid, ids):
    proj = store.load(pid)
    src = Image.open(store.pdir(pid) / "source.png").convert("RGB")
    assets = [a for a in (store.get_asset(proj, i) for i in ids) if a and a["type"] != "background"]
    if not assets:
        raise ValueError("Chọn ít nhất 1 asset (không tính background).")
    style = proj.get("screen", {}).get("style") or "polished casual mobile game UI"
    out_dir = store.pdir(pid) / "gptsheet"
    out_dir.mkdir(exist_ok=True)
    sheets = []
    nchunk = math.ceil(len(assets) / MAX_PER_SHEET)
    per = math.ceil(len(assets) / nchunk)
    for c in range(nchunk):
        chunk = assets[c * per:(c + 1) * per]
        crops = []
        for a in chunk:
            x, y, w, h = a["bbox"]
            crops.append(src.crop((x, y, x + w, y + h)))
        key, col = _pick_key(crops)
        (CW, CH), rects, rowh = layout([cr.width / max(1, cr.height) for cr in crops])
        sheet = Image.new("RGBA", (CW * REF_SCALE, CH * REF_SCALE), (0, 0, 0, 0))
        slots = []
        for a, cr, r in zip(chunk, crops, rects):
            x, y, w, h = [v * REF_SCALE for v in r]
            im = cr.resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
            sheet.paste(im.convert("RGBA"), (round(x), round(y)))
            slots.append({"aid": a["id"], "name": a["name"],
                          "rect": [r[0] / CW, r[1] / CH, r[2] / CW, r[3] / CH]})
        sid = store.new_id("s")
        fn = f"gptsheet/{sid}.png"
        sheet.save(store.pdir(pid) / fn)
        prompt = PROMPT.format(n=len(chunk), key=key, hexc="#%02X%02X%02X" % col,
                               orient=_orient(CW, CH), style=style,
                               items="\n".join(_item_line(i + 1, a) for i, a in enumerate(chunk)))
        sheets.append({"id": sid, "file": fn, "size": [CW, CH], "margin": 0.4 * rowh,
                       "key": list(col),
                       "key_name": key, "slots": slots, "prompt": prompt,
                       "created": time.time(), "imported": None})

    def f(p):
        p.setdefault("gpt_sheets", []).extend(sheets)
        p["gpt_sheets"] = p["gpt_sheets"][-30:]
    store.update(pid, f)
    return sheets


# ---------------- import ----------------

def _key_alpha(rgb, key):
    """Soft chroma key + un-mix of the key colour from edge pixels."""
    f = rgb.astype(np.float32)
    k = np.array(key, np.float32)
    d = np.linalg.norm(f - k, axis=2)
    t0, t1 = 45.0, 110.0
    a = np.clip((d - t0) / (t1 - t0), 0, 1)
    safe = np.maximum(a, 0.05)[..., None]
    col = np.clip((f - (1 - a[..., None]) * k) / safe, 0, 255)
    return col.astype(np.uint8), (a * 255).astype(np.uint8)


def _clean(alpha):
    """Keep the main blob(s); drop specks and bits of neighbours entering from the border."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 128).astype(np.uint8), 8)
    if n <= 1:
        return alpha
    areas = st[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    H, W = alpha.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        if ar == big or (ar >= 0.03 * big and not touches):
            keep[i] = True
    solid = keep[lab]
    solid = cv2.dilate(solid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(solid, alpha, 0).astype(np.uint8)


def _matte(im):
    """Whole image -> (rgb, alpha, key). Real alpha if ChatGPT gave a transparent PNG,
    otherwise key out the flat backdrop colour sampled from the image border."""
    arr = np.array(im.convert("RGBA"))
    if has_real_alpha(im):
        return arr[..., :3], arr[..., 3], None
    rgb = arr[..., :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
    key = np.median(border, axis=0)
    col, al = _key_alpha(rgb, key)
    return col, al, key


def _blobs(alpha):
    H, W = alpha.shape
    solid = (alpha > 128).astype(np.uint8)
    k = max(3, int(0.012 * max(W, H)) | 1)          # bridge small gaps inside one element
    merged = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, 8)
    if n <= 1:
        return [], lab
    big = st[1:, cv2.CC_STAT_AREA].max()
    out = []
    for i in range(1, n):
        x, y, w, h, ar = [int(v) for v in st[i]]
        if ar >= 0.02 * big:
            out.append({"labs": [i], "box": [x, y, x + w, y + h], "c": (x + w / 2, y + h / 2)})
    return out, lab


def _norm(pts):
    pts = np.array(pts, np.float64)
    lo, hi = pts.min(0), pts.max(0)
    return (pts - lo) / np.maximum(hi - lo, 1e-6)


def _assign(slot_c, blob_c):
    """Match slots to blobs by layout order (centres normalised to their own extents,
    so ChatGPT rescaling / shifting the whole layout does not matter)."""
    S, B = _norm(slot_c), _norm(blob_c)
    cost = np.linalg.norm(S[:, None, :] - B[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return {int(a): int(b) for a, b in zip(r, c)}
    except Exception:
        pass
    out, us, ub = {}, set(), set()
    for _, i, j in sorted((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])):
        if i not in us and j not in ub:
            out[i] = j
            us.add(i)
            ub.add(j)
    return out


def import_result(pid, sid, data: bytes):
    proj = store.load(pid)
    sh = next((s for s in proj.get("gpt_sheets", []) if s["id"] == sid), None)
    if not sh:
        raise ValueError("Không tìm thấy sheet.")
    im = Image.open(io.BytesIO(data))
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    W, H = im.size
    CW, CH = sh["size"]
    warn = []
    col, al, key = _matte(im)
    use_alpha = key is None
    blobs, lab = _blobs(al)
    slots = sh["slots"]
    if len(blobs) < len(slots):
        warn.append(f"Chỉ tìm thấy {len(blobs)} hình cho {len(slots)} ô – ChatGPT có thể đã gộp/bỏ bớt, "
                    "kiểm tra lại từng asset.")
    m = {}
    if blobs:
        sc = [((s["rect"][0] + s["rect"][2] / 2) * W, (s["rect"][1] + s["rect"][3] / 2) * H) for s in slots]
        if len(slots) == 1:          # one element: take the biggest shape
            m = {0: int(np.argmax([(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
                                   for b in blobs]))}
        elif len(blobs) == 1:        # one shape: give it to the closest slot
            dist = [math.dist(c, blobs[0]["c"]) for c in sc]
            m = {int(np.argmin(dist)): 0}
        else:
            m = _assign(sc, [b["c"] for b in blobs])
        # stray pieces (bells, sparkles…) not matched: attach to the element whose box contains them
        used = set(m.values())
        for j, b in enumerate(blobs):
            if j in used:
                continue
            for i, bj in m.items():
                x0, y0, x1, y1 = blobs[bj]["box"]
                px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
                if x0 - px <= b["c"][0] <= x1 + px and y0 - py <= b["c"][1] <= y1 + py:
                    blobs[bj]["labs"] += b["labs"]
                    bx = blobs[bj]["box"]
                    blobs[bj]["box"] = [min(bx[0], b["box"][0]), min(bx[1], b["box"][1]),
                                        max(bx[2], b["box"][2]), max(bx[3], b["box"][3])]
                    break
    d = store.pdir(pid)
    (d / "regen").mkdir(exist_ok=True)
    res = []
    for i, s in enumerate(slots):
        if i not in m:
            res.append({"aid": s["aid"], "name": s["name"], "ok": False,
                        "error": "Không tìm thấy hình tương ứng"})
            continue
        b = blobs[m[i]]
        x0, y0, x1, y1 = b["box"]
        sub = np.isin(lab[y0:y1, x0:x1], b["labs"])
        a = np.where(sub, al[y0:y1, x0:x1], 0).astype(np.uint8)
        out, _ = trim_alpha(Image.fromarray(np.dstack([col[y0:y1, x0:x1], a])))
        if out.width < 4 or out.height < 4:
            res.append({"aid": s["aid"], "name": s["name"], "ok": False, "error": "Hình quá nhỏ"})
            continue
        fn = f"regen/{s['aid']}_c{int(time.time() * 1000) % 10 ** 8}.png"
        out.save(d / fn)
        res.append({"aid": s["aid"], "name": s["name"], "ok": True, "file": fn,
                    "w": out.width, "h": out.height})
    (d / "gptsheet" / f"{sid}_result.png").write_bytes(data)

    def f(p):
        for r in res:
            if not r["ok"]:
                continue
            a = store.get_asset(p, r["aid"])
            if not a:
                continue
            a["versions"].append({"file": r["file"], "kind": "regen", "source": "chatgpt",
                                  "w": r["w"], "h": r["h"],
                                  "scale": round(r["w"] / max(1, a["bbox"][2]), 3),
                                  "created": time.time()})
            a["active"] = len(a["versions"]) - 1
            a["status"] = "regenerated"
            a["qa"] = None
            a["error"] = None
        for s2 in p.get("gpt_sheets", []):
            if s2["id"] == sid:
                s2["imported"] = time.time()
    store.update(pid, f)
    return {"results": res, "warnings": warn, "alpha": bool(use_alpha)}
