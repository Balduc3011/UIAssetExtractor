"""Step 1 – local CV: candidate UI groups + OCR text boxes. No API calls."""
import cv2
import numpy as np

from .imgutil import background_estimate, foreground_map, iou


def detect_groups(rgb: np.ndarray, min_frac=0.0006, max_frac=0.5) -> list[list[int]]:
    H, W = rgb.shape[:2]
    bg = background_estimate(rgb)
    m = foreground_map(rgb, bg)
    k = max(3, round(min(W, H) / 120)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h
        if a < min_frac * W * H or a > max_frac * W * H:
            continue
        if w < 6 or h < 6:
            continue
        boxes.append([x, y, w, h])
    # drop near-duplicates
    boxes.sort(key=lambda b: -b[2] * b[3])
    out = []
    for b in boxes:
        if all(iou(b, o) < 0.85 for o in out):
            out.append(b)
    out.sort(key=lambda b: (b[1] // 40, b[0]))
    return out


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
