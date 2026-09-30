"""Step 1 – local CV: candidate UI groups + OCR text boxes. No API calls."""
import cv2
import numpy as np

from .imgutil import background_estimate, foreground_map, iou


def _fgmap(rgb: np.ndarray) -> np.ndarray:
    """Pixels that differ from the (large-scale) backdrop or sit on strong edges.
    The backdrop is estimated on a downscaled copy so big panels/gradients/patterns
    are treated as backdrop, not foreground."""
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (max(8, W // 4), max(8, H // 4)), interpolation=cv2.INTER_AREA)
    k = max(15, (min(small.shape[:2]) // 6) | 1)
    bg = cv2.resize(cv2.medianBlur(small, min(k, 255)), (W, H), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(lab - lb, axis=2)
    edges = cv2.Canny(cv2.GaussianBlur(rgb, (5, 5), 0), 90, 200)
    return ((diff > 30) | (edges > 0)).astype(np.uint8)


def _components(m, W, H, min_frac, off=(0, 0)):
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if w * h < min_frac * W * H or w < 6 or h < 6:
            continue
        out.append([int(x) + off[0], int(y) + off[1], int(w), int(h)])
    return out


def _ell(k):
    k = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def detect_groups(rgb: np.ndarray, min_frac=0.0006, max_frac=0.6) -> list[list[int]]:
    """Candidate UI regions. Big regions (panels, bars, overlapping stacks) are kept and
    their inner elements are added as separate candidates, so the AI step can label
    both levels."""
    H, W = rgb.shape[:2]
    s = min(W, H)
    m = _fgmap(rgb)
    k = max(3, int(s / 150)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    # solid silhouettes (holes filled)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
    solid = m | (ff == 0).astype(np.uint8)
    kb = max(5, int(s * 0.025))
    # opening breaks thin connectors (rails, lines) between elements
    boxes = _components(cv2.morphologyEx(solid, cv2.MORPH_OPEN, _ell(kb)), W, H, min_frac)
    out = []
    for b in boxes:
        x, y, w, h = b
        if w * h > max_frac * W * H:
            continue
        out.append(b)
        if w * h > 0.04 * W * H:
            # large region: also propose what's inside it
            sub_s = solid[y:y + h, x:x + w]
            kids = _components(cv2.morphologyEx(sub_s, cv2.MORPH_OPEN, _ell(max(kb + 2, s * 0.05))),
                               W, H, min_frac, (x, y))
            sub_m = m[y:y + h, x:x + w]
            kids += _components(cv2.morphologyEx(sub_m, cv2.MORPH_OPEN, _ell(kb)),
                                W, H, min_frac, (x, y))
            for c in kids:
                if c[2] * c[3] < 0.6 * w * h:
                    out.append(c)
    out.sort(key=lambda b: -b[2] * b[3])
    res = []
    for b in out:
        if all(iou(b, o) < 0.8 for o in res):
            res.append(b)
    res.sort(key=lambda b: (b[1] // 40, b[0]))
    return res[:80]


def run_ocr(rgb: np.ndarray) -> list[dict]:
    from ..gpu import ocr
    try:
        r = ocr()(rgb[:, :, ::-1].copy())
    except Exception:
        return []
    res = []
    if r is None or r.boxes is None:
        return res
    for box, txt, sc in zip(r.boxes, r.txts, r.scores):
        # low-confidence reads are usually shapes (a padlock read as "8")
        if float(sc) < 0.8 or not any(ch.isalnum() for ch in str(txt)):
            continue
        pts = np.array(box)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        res.append({"text": str(txt), "score": float(sc),
                    "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return res


def snap_box(rgb: np.ndarray, box, margin=6) -> list[int]:
    """Tighten a rough box (e.g. from Claude) to the actual pixels."""
    H, W = rgb.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return box
    region = rgb[y0:y1, x0:x1]
    bg = background_estimate(rgb)[y0:y1, x0:x1]
    m = foreground_map(region, bg)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(m > 0)
    if len(xs) < 20:
        return box
    nx0, nx1 = np.percentile(xs, [0.5, 99.5])
    ny0, ny1 = np.percentile(ys, [0.5, 99.5])
    nb = [x0 + int(nx0), y0 + int(ny0), int(nx1 - nx0) + 1, int(ny1 - ny0) + 1]
    # only accept if it is a reasonable refinement
    if nb[2] * nb[3] < 0.35 * w * h:
        return box
    return nb
