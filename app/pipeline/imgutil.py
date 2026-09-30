import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def b64png(im: Image.Image) -> str:
    return base64.b64encode(to_png_bytes(im)).decode()


def clamp_box(b, W, H):
    x, y, w, h = [int(round(v)) for v in b]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [x, y, w, h]


def pad_box(b, pad, W, H):
    x, y, w, h = b
    return clamp_box([x - pad, y - pad, w + 2 * pad, h + 2 * pad], W, H)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def inter_frac(inner, outer):
    """Fraction of `inner` area lying inside `outer`."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / float(aw * ah + 1e-6)


def background_estimate(rgb: np.ndarray) -> np.ndarray:
    k = max(31, (min(rgb.shape[:2]) // 12) | 1)
    k = min(k, 99)
    return cv2.medianBlur(rgb, k)


def foreground_map(rgb: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    """Rough 0/255 map of 'things that are not the backdrop'."""
    if bg is None:
        bg = background_estimate(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    labb = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(lab - labb, axis=2)
    edges = cv2.Canny(cv2.GaussianBlur(rgb, (3, 3), 0), 60, 160)
    m = ((diff > 28) | (edges > 0)).astype(np.uint8) * 255
    return m


def checker(w, h, s=8) -> Image.Image:
    yy, xx = np.indices((h, w))
    c = (((yy // s) + (xx // s)) % 2).astype(np.uint8)
    arr = np.where(c[..., None] == 1, 205, 245).astype(np.uint8).repeat(3, 2)
    return Image.fromarray(arr).convert("RGBA")


def on_checker(im: Image.Image) -> Image.Image:
    im = im.convert("RGBA")
    bg = checker(im.width, im.height)
    bg.alpha_composite(im)
    return bg


def font(size=14):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_marks(im: Image.Image, boxes: list, labels: list) -> Image.Image:
    """Set-of-Mark overlay: numbered boxes."""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    fs = max(12, min(im.width, im.height) // 45)
    f = font(fs)
    colors = [(255, 0, 80), (0, 200, 255), (0, 220, 0), (255, 170, 0), (200, 0, 255)]
    for i, (b, lab) in enumerate(zip(boxes, labels)):
        x, y, w, h = b
        c = colors[i % len(colors)]
        d.rectangle([x, y, x + w, y + h], outline=c, width=2)
        t = str(lab)
        tw = d.textlength(t, font=f)
        d.rectangle([x, y, x + tw + 6, y + fs + 4], fill=c)
        d.text((x + 3, y + 1), t, fill=(0, 0, 0) if c[1] > 150 else (255, 255, 255), font=f)
    return im


def draw_grid(im: Image.Image, step=100) -> Image.Image:
    """Grid with 0..1000 normalized ticks to help the model give coordinates."""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    f = font(max(10, min(W, H) // 40))
    for v in range(0, 1001, step):
        x = v * (W - 1) / 1000
        y = v * (H - 1) / 1000
        d.line([(x, 0), (x, H)], fill=(0, 255, 255, 90), width=1)
        d.line([(0, y), (W, y)], fill=(0, 255, 255, 90), width=1)
        if 0 < v < 1000:
            d.text((x + 2, 2), str(v), fill=(0, 255, 255, 230), font=f)
            d.text((2, y + 2), str(v), fill=(0, 255, 255, 230), font=f)
    return im


def upscale_for_view(im: Image.Image, target=768, min_short=None, max_long=1600) -> Image.Image:
    s = target / max(im.size)
    if min_short:
        s = max(s, min_short / min(im.size))
        s = min(s, max_long / max(im.size))
    if s <= 1:
        return im
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def trim_alpha(im: Image.Image, thr=8):
    """Returns (trimmed, (dx, dy))."""
    im = im.convert("RGBA")
    a = np.array(im)[:, :, 3]
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return im, (0, 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return im.crop((x0, y0, x1, y1)), (int(x0), int(y0))


def has_real_alpha(im: Image.Image) -> bool:
    if im.mode != "RGBA":
        return False
    a = np.array(im)[:, :, 3]
    return (a < 250).mean() > 0.02
