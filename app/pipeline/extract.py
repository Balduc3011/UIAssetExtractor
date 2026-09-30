"""Step 3 – local extraction: matte, remove occluders, remove text, trim, 9-slice, dedupe."""
import time

import cv2
import numpy as np
from PIL import Image

from .. import store
from ..config import load_settings
from ..gpu import models
from . import refine
from .imgutil import background_estimate, inter_frac, pad_box, trim_alpha

PAD = 3


def pad_of(bbox) -> int:
    """Crop padding around an element (tight AI boxes often clip glyphs/outlines)."""
    return max(PAD, int(round(0.035 * max(bbox[2], bbox[3]))))


def _dist_to_centers(lab, centers):
    return np.min(np.linalg.norm(lab[:, :, None, :] - centers[None, None], axis=3), axis=2)


def _kmeans_centers(px, k=4):
    px = px.astype(np.float32)
    k = max(1, min(k, len(px)))
    _, _, centers = cv2.kmeans(px, k, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               2, cv2.KMEANS_PP_CENTERS)
    return centers


def border_distance(crop: np.ndarray, ring=2, k=4) -> np.ndarray:
    """Per-pixel Lab distance to the nearest backdrop colour (sampled on the border)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
                           lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    return _dist_to_centers(lab, _kmeans_centers(edge, k))


def backdrop_distance(src: np.ndarray, box, k=5) -> np.ndarray:
    """Distance of each crop pixel to the backdrop colours, sampled on a ring just
    OUTSIDE the crop (so parts of the element touching the crop edge aren't mistaken
    for backdrop)."""
    H, W = src.shape[:2]
    x, y, w, h = box
    r = max(3, min(w, h) // 20)
    X0, Y0, X1, Y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
    big = src[Y0:Y1, X0:X1]
    ringm = np.ones(big.shape[:2], bool)
    ringm[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
    crop = src[y:y + h, x:x + w]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    if ringm.sum() < 20:          # element touches the screenshot edge
        return border_distance(crop)
    blab = cv2.cvtColor(big, cv2.COLOR_RGB2LAB).astype(np.float32)[ringm]
    return _dist_to_centers(lab, _kmeans_centers(blab, k))


def matte_grabcut(crop: np.ndarray, diff: np.ndarray | None = None) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return np.ones((h, w), bool)
    if diff is None:
        diff = border_distance(crop)
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 50, 140),
                       np.ones((3, 3), np.uint8))
    mask[(diff < 10) & (edges == 0)] = cv2.GC_PR_BGD
    # crop border is backdrop only where it looks like backdrop
    b = 2
    border = np.zeros((h, w), bool)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    sure_bg = border & (diff < 14)
    if sure_bg.sum() < 0.2 * border.sum():
        sure_bg = border  # nothing looks like backdrop: fall back to the whole border
    mask[sure_bg] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.ones((h, w), bool)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def matte_ai(crop: np.ndarray, s) -> np.ndarray:
    from rembg import remove
    sess = models.rembg(s["ai_matte_model"], s["gpu_mode"])
    out = remove(Image.fromarray(crop), session=sess, only_mask=True)
    return np.array(out.convert("L"))


def extract_one(pid, proj, a, src, bg, s):
    W, H = src.shape[1], src.shape[0]
    P = pad_of(a["bbox"])
    x, y, w, h = pad_box(a["bbox"], P, W, H)
    crop = src[y:y + h, x:x + w].copy()
    d = store.pdir(pid)
    Image.fromarray(crop).save(d / "crops" / f"{a['id']}.png")

    # ---- 1. binary silhouette
    method = a.get("method") or s["matte_method"]
    dist = backdrop_distance(src, (x, y, w, h)) if min(h, w) >= 8 else None
    if a["type"] == "background":
        mask = np.ones((h, w), bool)
    elif method == "ai":
        mask = matte_ai(crop, s) > 128
    elif method == "none":
        mask = np.zeros((h, w), bool)
        mask[P:h - P, P:w - P] = True
    else:
        mask = matte_grabcut(crop, dist)
    if a["type"] != "background" and method != "none":
        # the padding margin around our box: pixels there that fall inside another
        # element's box belong to that neighbour, not to us
        bx, by, bw, bh = a["bbox"]
        own = np.zeros((h, w), bool)
        own[by - y:by - y + bh, bx - x:bx - x + bw] = True
        for o in proj["assets"]:
            if o["id"] == a["id"] or o["type"] == "background":
                continue
            if inter_frac(a["bbox"], o["bbox"]) > 0.9:
                continue  # parent
            qx, qy, qw, qh = o["bbox"]
            nb = np.zeros((h, w), bool)
            nb[max(0, qy - y):max(0, qy - y + qh), max(0, qx - x):max(0, qx - x + qw)] = True
            mask &= ~(nb & ~own)
        mask = refine.drop_border_strips(mask)
        mask = refine.keep_main(mask)
        mask = refine.fill_holes(mask, bg_dist=dist)
        mask = refine.smooth_mask(mask)
    Image.fromarray((mask * 255).astype(np.uint8)).save(d / "mattes" / f"{a['id']}.png")

    # ---- 2. overlays stacked on top (badges, plates...) hide part of this element
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
        ox, oy, ow, oh = pad_box(o["bbox"], pad_of(o["bbox"]), W, H)
        x0, y0 = max(ox - x, 0), max(oy - y, 0)
        x1, y1 = min(ox + ow - x, w), min(oy + oh - y, h)
        if x1 <= x0 or y1 <= y0:
            continue
        omask = _asset_alpha(pid, o)
        if omask is not None and omask.shape == (oh, ow):
            # only trust the overlay's matte inside its own box (padding picks up neighbours)
            om = omask > 128
            keep = np.zeros_like(om)
            bx, by, bw, bh = o["bbox"]
            keep[max(0, by - oy - 1):by - oy + bh + 1, max(0, bx - ox - 1):bx - ox + bw + 1] = True
            om &= keep
            sub = om[y0 + y - oy:y1 + y - oy, x0 + x - ox:x1 + x - ox]
            occl[y0:y1, x0:x1] |= sub
        else:
            bx, by, bw, bh = o["bbox"]
            occl[max(by - y, 0):max(0, min(by + bh - y, h)),
                 max(bx - x, 0):max(0, min(bx + bw - x, w))] = True
    occl = cv2.dilate(occl.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    occ_ratio = float((occl & mask).sum() / max(1, mask.sum()))

    rgb = crop.copy()
    frozen = None
    if occl.any():
        vis = mask & ~occl
        # (a) holes fully enclosed by the element (icon sitting on a panel): rebuild surface
        hole = occl & fill_enclosed(vis)
        if hole.any():
            rgb, _ = refine.fill_region(rgb, vis, hole)
            vis = vis | hole
        # (b) the rest: mirror from the symmetric visible half when possible
        rest = occl & ~hole
        if rest.any():
            rgb, vis, _ = refine.symmetric_complete(rgb, vis, rest)
        # (c) still-missing pixels inside the element's convex outline (e.g. the middle of
        #     a ring frame hidden by its icon): rebuild with a row/column/radial model
        rest = occl & ~vis
        if rest.any() and vis.sum() > 50:
            hull = np.zeros((h, w), np.uint8)
            cnts, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(hull, [cv2.convexHull(np.vstack(cnts))], -1, 1, -1)
            inner = rest & (cv2.erode(hull, np.ones((3, 3), np.uint8)) > 0)
            # only small gaps are rebuilt locally; big hidden areas are left for GPT
            if inner.any() and inner.sum() <= 0.15 * vis.sum():
                rgb, _ = refine.fill_region(rgb, vis, inner, center=refine.centroid(vis))
                vis = vis | inner
        mask = refine.smooth_mask(vis) if a["type"] != "background" else vis
        frozen = cv2.dilate(occl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    # ---- 3. erase text (text assets lying on it always; OCR text when remove_text)
    text_assets = [o["bbox"] for o in proj["assets"]
                   if o["type"] == "text" and o["id"] != a["id"]
                   and inter_frac(o["bbox"], a["bbox"]) > 0.6]
    strip = (a["remove_text"] or s.get("strip_all_text", True)) and a["type"] != "text"
    tboxes = []
    if strip:
        # OCR boxes + the text boxes Claude marked (catches single digits OCR misses)
        tboxes += list(a["text_boxes"]) + list(a.get("ai_text_boxes") or [])
    if a["type"] not in ("button", "label_plate", "panel", "bar"):
        # on icons, text that fills most of the element is the artwork itself
        # (an "ADS" logo), not a label: keep it
        area = max(1, a["bbox"][2] * a["bbox"][3])
        ai = [list(t) for t in (a.get("ai_text_boxes") or [])]
        tboxes = [t for t in tboxes if list(t) in ai or t[2] * t[3] < 0.4 * area]
    if a["type"] != "text":
        tboxes += text_assets
    if tboxes:
        local = [[bx - x, by - y, bw, bh] for bx, by, bw, bh in tboxes]
        rgb, mask = refine.remove_text(rgb, mask, local)

    # ---- 4. soft anti-aliased edge + remove backdrop colour bleeding
    if a["type"] == "background" or method == "none":
        alpha = (mask * 255).astype(np.uint8)
    else:
        alpha, rgb_edge = refine.refine_alpha(crop, mask, frozen)
        band = (alpha > 0) & (alpha < 255)
        rgb[band] = rgb_edge[band]

    im = Image.fromarray(np.dstack([rgb, alpha]))
    im.save(d / "mattes" / f"{a['id']}_full.png")  # untrimmed, padded-bbox coords (debug/QA)
    if a["type"] != "background":
        im, _ = trim_alpha(im)
    fn = f"{a['id']}_x{int(time.time() * 1000) % 10 ** 8}.png"
    im.save(d / "assets" / fn)
    borders = None
    if a["nine_slice"]:
        b = refine.nine_slice(np.array(im))
        if b:
            borders = {k: b[k] for k in ("l", "t", "r", "b")}
    return {"file": f"assets/{fn}", "kind": "extract", "w": im.width, "h": im.height,
            "scale": 1.0, "created": time.time()}, occ_ratio, borders


def fill_enclosed(vis: np.ndarray) -> np.ndarray:
    """Pixels not in `vis` that are fully enclosed by it."""
    inv = (~vis).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = vis.shape
    out = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        out |= lab == i
    return out


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
