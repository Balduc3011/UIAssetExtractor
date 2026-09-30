"""Quality helpers for extraction: clean masks, soft edges (alpha matting), colour
decontamination, gradient-aware fills, symmetry completion and 9-slice compaction."""
import cv2
import numpy as np

K3 = np.ones((3, 3), np.uint8)


def _disk(r):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


# ---------------------------------------------------------------- masks

def drop_border_strips(mask: np.ndarray) -> np.ndarray:
    """Remove rails / lines / bars that run behind the element and enter the crop from
    its border. Only long, thin pieces lying along the border normal are dropped, so
    parts of the element that merely touch a tight crop are kept."""
    h, w = mask.shape
    k = max(3, int(round(min(h, w) * 0.09)))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(k // 2))
    thin = (mask > 0) & (opened == 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    out = mask.copy().astype(bool)
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        side = x <= 4 or x + ww >= w - 4
        tb = y <= 4 or y + hh >= h - 4
        rail_h = side and hh <= 0.14 * h and ww >= 1.5 * hh
        rail_v = tb and ww <= 0.14 * w and hh >= 1.5 * ww
        if rail_h or rail_v:
            out[lab == i] = False
    return out


def keep_main(mask: np.ndarray, min_frac=0.02) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask.astype(bool)
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    out = np.zeros_like(mask, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(12, min_frac * biggest):
            out |= lab == i
    return out


def fill_holes(mask: np.ndarray, rgb=None, bg_dist=None, max_frac=0.01) -> np.ndarray:
    """Fill enclosed holes that are small or don't look like the backdrop."""
    inv = (~mask.astype(bool)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = mask.shape
    out = mask.astype(bool).copy()
    for i in range(1, n):
        x, y, ww, hh, ar = stats[i]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        reg = lab == i
        if ar < max_frac * h * w or (bg_dist is not None and float(np.median(bg_dist[reg])) > 16):
            out[reg] = True
    return out


def smooth_mask(mask: np.ndarray, sigma=1.2) -> np.ndarray:
    """Remove stair-steps / jaggies from a binary mask."""
    f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return f > 0.5


# ---------------------------------------------------------------- soft edges

def refine_alpha(rgb: np.ndarray, mask: np.ndarray, frozen: np.ndarray | None = None,
                 band=2):
    """Closed-form alpha matting on a thin band around the mask edge + colour
    decontamination. Returns (alpha uint8, rgb uint8)."""
    m = mask.astype(bool)
    h, w = m.shape
    if m.sum() < 30 or h < 8 or w < 8:
        return (m * 255).astype(np.uint8), rgb
    fg = cv2.erode(m.astype(np.uint8), _disk(band)) > 0
    bgm = cv2.dilate(m.astype(np.uint8), _disk(band + 1)) == 0
    tri = np.full((h, w), 0.5)
    tri[fg] = 1.0
    tri[bgm] = 0.0
    if frozen is not None:          # e.g. cut edges next to an occluder: keep hard
        tri[frozen & m] = 1.0
        tri[frozen & ~m] = 0.0
    img = rgb.astype(np.float64) / 255.0
    try:
        from pymatting import estimate_alpha_cf, estimate_foreground_ml
        a = estimate_alpha_cf(img, tri)
        a = np.clip(a, 0, 1)
        # matting can't create new solid regions far from the mask
        a[bgm] = 0
        a[fg] = 1
        f = estimate_foreground_ml(img, a)
        band_m = (tri == 0.5)
        out = rgb.copy()
        out[band_m] = np.clip(f[band_m] * 255, 0, 255).astype(np.uint8)
        alpha = (a * 255).round().astype(np.uint8)
        # tiny speckles -> 0
        alpha[alpha < 8] = 0
        return alpha, out
    except Exception:
        a = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
        return (a * 255).astype(np.uint8), rgb


# ---------------------------------------------------------------- fills

def _row_model(rgb, known, axis):
    """Median colour per row (axis=0) or per column (axis=1) of known pixels."""
    h, w = known.shape
    n = h if axis == 0 else w
    vals = np.full((n, 3), np.nan, np.float32)
    for i in range(n):
        k = known[i] if axis == 0 else known[:, i]
        if k.sum() >= 3:
            px = rgb[i][k] if axis == 0 else rgb[:, i][k]
            vals[i] = np.median(px, axis=0)
    idx = np.arange(n)
    ok = ~np.isnan(vals[:, 0])
    if ok.sum() < 2:
        return None
    for c in range(3):
        vals[:, c] = np.interp(idx, idx[ok], vals[ok, c])
    return np.repeat(vals[:, None, :], w, 1) if axis == 0 else np.repeat(vals[None], h, 0)


def _radial_model(rgb, known, center, h, w):
    cy, cx = center
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ri = r.astype(np.int32)
    n = ri.max() + 1
    vals = np.full((n, 3), np.nan, np.float32)
    kr = ri[known]
    kp = rgb[known]
    for i in np.unique(kr):
        sel = kp[kr == i]
        if len(sel) >= 3:
            vals[i] = np.median(sel, axis=0)
    ok = ~np.isnan(vals[:, 0])
    if ok.sum() < 3:
        return None
    idx = np.arange(n)
    for c in range(3):
        vals[:, c] = np.interp(idx, idx[ok], vals[ok, c])
    return vals[ri]


def _flat_dominant(rgb, px):
    if len(px) < 4:
        return None
    k = 2 if len(px) >= 8 else 1
    _, lbl, cen = cv2.kmeans(px.astype(np.float32), k, None,
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    big = int(np.argmax(np.bincount(lbl.ravel(), minlength=k)))
    return np.broadcast_to(cen[big], rgb.shape).astype(np.float32)


def _models(f, known, focus, center=None, radius=None):
    """Candidate surface models built from known pixels near `focus`."""
    h, w = known.shape
    if radius is None:
        ys, xs = np.where(focus)
        radius = max(4, int(0.35 * min(np.ptp(xs) + 1, np.ptp(ys) + 1))) if len(xs) else 8
    near = (cv2.dilate(focus.astype(np.uint8), _disk(radius)) > 0) & known
    if near.sum() < 10:
        near = known
    out = []
    for axis in (0, 1):
        mdl = _row_model(f, near, axis)
        if mdl is not None:
            out.append(("row" if axis == 0 else "col", mdl))
    ring = (cv2.dilate(focus.astype(np.uint8), _disk(5)) > 0) & known
    fl = _flat_dominant(f, f[ring] if ring.sum() >= 8 else f[near])
    if fl is not None:
        out.append(("flat", fl))
    if center is not None:
        mdl = _radial_model(f, known, center, h, w)
        if mdl is not None:
            out.append(("radial", mdl))
    return out


def _pick(models, f, ring):
    best, err = None, 1e9
    for name, mdl in models:
        if not ring.any():
            break
        e = float(np.median(np.abs(mdl[ring] - f[ring]).mean(axis=1)))
        if e < err:
            best, err = (name, mdl), e
    return best, err


def fill_region(rgb: np.ndarray, known: np.ndarray, region: np.ndarray, focus=None,
                center=None):
    """Rebuild `region` smoothly from the surrounding known surface. Returns (rgb, 0)."""
    return membrane_fill(rgb, region.astype(bool) & ~known.astype(bool) | region.astype(bool)), 0.0


def centroid(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return ((ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0)


def membrane_fill(rgb: np.ndarray, region: np.ndarray, iters=150, valid=None,
                  init=None) -> np.ndarray:
    """Smooth fill of `region` from its boundary (Laplace relaxation): reproduces flat
    colours and gradients without smearing. Only `valid` pixels (default: everything
    outside the region) act as sources, so outlines next to the region don't bleed in."""
    region = region.astype(bool)
    if not region.any():
        return rgb
    if init is None:
        out = cv2.inpaint(rgb, region.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
    else:
        out = rgb.astype(np.float32).copy()
        out[region] = init[region]
    src_ok = (~region) if valid is None else (valid.astype(bool) & ~region)
    ys, xs = np.where(region)
    y0, y1 = max(0, ys.min() - 1), min(rgb.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(rgb.shape[1], xs.max() + 2)
    sub = out[y0:y1, x0:x1]
    rm = region[y0:y1, x0:x1]
    wgt = (rm | src_ok[y0:y1, x0:x1]).astype(np.float32)
    wp = np.pad(wgt, 1)
    wsum = wp[:-2, 1:-1] + wp[2:, 1:-1] + wp[1:-1, :-2] + wp[1:-1, 2:]
    ok = rm & (wsum > 0)
    for _ in range(iters):
        p = np.pad(sub * wgt[..., None], ((1, 1), (1, 1), (0, 0)))
        acc = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        sub[ok] = acc[ok] / wsum[ok][:, None]
    out[y0:y1, x0:x1] = sub
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_text(rgb: np.ndarray, mask: np.ndarray, boxes: list):
    """Erase glyphs (fill, outline, shadow) inside the element. The surface colour is
    learned from the perimeter of each text box (glyphs sit inside the box), glyph
    pixels are those that differ from it, and they are rebuilt with a smooth membrane
    fill from the clean surrounding surface. Text floating over transparent parts of
    the element is made transparent. Returns (rgb, mask)."""
    h, w = mask.shape
    m = mask.astype(bool)
    out = rgb.copy()
    newm = m.copy()
    L = to_lab(rgb)
    strokes = np.zeros((h, w), bool)
    dt = None
    valid = np.zeros((h, w), bool)
    init = rgb.astype(np.float32).copy()
    for x, y, bw, bh in boxes:
        pd = max(3, int(round(0.15 * min(bw, bh))))  # boxes are rarely tight on outlines
        x0, y0 = max(0, x - pd), max(0, y - pd)
        x1, y1 = min(w, x + bw + pd), min(h, y + bh + pd)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        b = np.zeros((h, w), bool)
        b[y0:y1, x0:x1] = True
        inside = b & m
        if inside.sum() < 0.5 * b.sum():
            # mostly outside the element's body: text over a transparent area -> drop it
            newm[b & ~(cv2.erode(m.astype(np.uint8), _disk(3)) > 0)] = False
            newm[b] = newm[b] & ~(np.linalg.norm(L[b] - L[b].mean(0), axis=1) > 25)
            continue
        if dt is None:
            dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        # surface colour: the biggest colour cluster inside the text box (glyph fill and
        # outline together usually cover < 50%), unless the side margins of the text line
        # (plain surface next to the glyphs) clearly agree on another cluster
        boxpx = L[b & m]
        if len(boxpx) < 12:
            continue
        k = 3
        _, lbl, cen = cv2.kmeans(boxpx.astype(np.float32), k, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(lbl.ravel(), minlength=k)
        surf = cen[int(np.argmax(counts))]
        side = np.zeros((h, w), bool)
        ym0, ym1 = y + int(0.2 * bh), y + int(0.8 * bh) + 1
        side[max(0, ym0):ym1, max(0, x - 5):max(0, x - 1)] = True
        side[max(0, ym0):ym1, min(w, x + bw + 1):min(w, x + bw + 5)] = True
        sp = L[side & m & (dt >= 2)]
        if len(sp) >= 8:
            dd = np.linalg.norm(sp[:, None, :] - cen[None], axis=2)
            near_c = dd.argmin(1)
            ok = dd.min(1) < 20
            if ok.mean() > 0.6:
                c = np.bincount(near_c[ok], minlength=k)
                if c.max() > 0.6 * len(sp):
                    surf = cen[int(c.argmax())]
        px = boxpx[np.linalg.norm(boxpx - surf, axis=1) < 25]
        if len(px) < 8:
            px = boxpx
        # tolerance adapts to how much the surface itself varies (gradients)
        spread = float(np.percentile(np.linalg.norm(px - surf, axis=1), 60))
        thr = max(14.0, min(40.0, 2.2 * spread + 8))
        dist = np.linalg.norm(L - surf, axis=2)
        st = b & m & (dist > thr)
        # glyphs sit inside the (padded) box; other surfaces crossing the box edge
        # (e.g. a frame around a panel) are not text
        n_, lab_, _, _ = cv2.connectedComponentsWithStats(st.astype(np.uint8), 8)
        edge_ids = np.unique(np.concatenate([lab_[y0, x0:x1], lab_[y1 - 1, x0:x1],
                                             lab_[y0:y1, x0], lab_[y0:y1, x1 - 1]]))
        for i in edge_ids:
            if not i:
                continue
            comp = lab_ == i
            sides = [comp[y0, x0:x1].mean(), comp[y1 - 1, x0:x1].mean(),
                     comp[y0:y1, x0].mean(), comp[y0:y1, x1 - 1].mean()]
            # a surface crossing the box runs along a whole side; glyphs only graze it
            if max(sides) > 0.6:
                st &= ~comp
        st = (cv2.dilate(st.astype(np.uint8), _disk(3)) > 0) & b & m
        strokes |= st
        near = cv2.dilate(b.astype(np.uint8), _disk(6)) > 0
        valid |= near & m & (dist <= thr)
        srgb = cv2.cvtColor(surf.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        init[st] = srgb
    if strokes.any():
        out = membrane_fill(out, strokes, valid=valid, init=init)
    return out, newm


# ---------------------------------------------------------------- symmetry

def _transforms(h, w, V):
    ys, xs = np.where(V)
    if len(xs) < 30:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    out = []
    # the visible part may be truncated by the overlay, so search a wide axis range
    for c in np.linspace(x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0), 21):
        out.append(("lr", c))
    for c in np.linspace(y0 + 0.25 * (y1 - y0), y0 + 0.75 * (y1 - y0), 21):
        out.append(("ud", c))
    out.append(("rot", ((x0 + x1) / 2, (y0 + y1) / 2)))
    return out


def _apply(arr, kind, c, h, w):
    if kind == "lr":
        M = np.float32([[-1, 0, 2 * c], [0, 1, 0]])
    elif kind == "ud":
        M = np.float32([[1, 0, 0], [0, -1, 2 * c]])
    else:
        cx, cy = c
        M = np.float32([[-1, 0, 2 * cx], [0, -1, 2 * cy]])
    flags = cv2.INTER_NEAREST
    return cv2.warpAffine(arr, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def symmetric_complete(rgb, mask, occl, min_iou=0.9, max_err=18):
    """Reconstruct pixels hidden by an overlay by mirroring the visible part, when the
    element is (left-right / up-down / point) symmetric. Returns (rgb, mask, filled_px)."""
    h, w = mask.shape
    V = mask.astype(bool) & ~occl
    O = occl.astype(bool)
    if not O.any() or V.sum() < 50:
        return rgb, mask, 0
    best = None
    Vu = V.astype(np.uint8)
    Ou = O.astype(np.uint8)
    L = to_lab(rgb)
    for kind, c in _transforms(h, w, V):
        TV = _apply(Vu, kind, c, h, w) > 0
        TO = _apply(Ou, kind, c, h, w) > 0
        inb = _apply(np.ones((h, w), np.uint8), kind, c, h, w) > 0
        # judge symmetry near the hidden part (and its mirror), not over the whole
        # element: text/logos in the middle often break global symmetry
        near = cv2.dilate((O | TO).astype(np.uint8), _disk(max(4, 0.2 * min(h, w)))) > 0
        dom = ~O & ~TO & inb & near
        a, b = V & dom, TV & dom
        uni = (a | b).sum()
        if uni < 50:
            continue
        iou = (a & b).sum() / uni
        if iou < min_iou:
            continue
        TL = _apply(L, kind, c, h, w)
        both = a & b
        err = float(np.abs(TL[both] - L[both]).mean()) if both.any() else 99
        # strong shape symmetry tolerates shading differences (top-lit vs bottom)
        if err > max_err and not (iou >= 0.94 and err <= 30):
            continue
        gain = int((O & TV).sum())
        # prefer the axis that reconstructs the most hidden pixels, then the best fit
        score = (gain / max(1, O.sum())) + (iou - err / 200) * 0.25
        if gain and (best is None or score > best[0]):
            best = (score, kind, c)
    if best is None:
        return rgb, mask, 0
    _, kind, c = best
    TV = _apply(Vu, kind, c, h, w) > 0
    Trgb = _apply(rgb, kind, c, h, w)
    fill = O & TV
    out = rgb.copy()
    out[fill] = Trgb[fill]
    m = mask.astype(bool).copy()
    m[fill] = True
    return out, m, int(fill.sum())


# ---------------------------------------------------------------- 9-slice

def nine_slice(rgba: np.ndarray, thr=5.0):
    """Stretchable middle band -> borders {l,t,r,b} or None."""
    a = rgba.astype(np.float32)
    h, w = a.shape[:2]
    if w < 12 or h < 12:
        return None

    def band(arr, n):
        c = n // 2
        ref = arr[c]
        lo = c
        while lo > 1 and np.abs(arr[lo - 1] - ref).mean() < thr:
            lo -= 1
        hi = c
        while hi < n - 2 and np.abs(arr[hi + 1] - ref).mean() < thr:
            hi += 1
        return lo, hi

    l, r = band(np.transpose(a, (1, 0, 2)), w)
    t, b = band(a, h)
    res = {"l": int(l), "r": int(w - 1 - r), "t": int(t), "b": int(h - 1 - b),
           "sx": (r - l) >= max(6, w * 0.25), "sy": (b - t) >= max(6, h * 0.25)}
    if not res["sx"] and not res["sy"]:
        return None
    return res


def compact_nine_slice(rgba: np.ndarray, keep=4):
    """Collapse the stretchable middle band(s) like an artist would deliver a 9-slice
    sprite. Returns (rgba, borders or None)."""
    b = nine_slice(rgba)
    if not b:
        return rgba, None
    h, w = rgba.shape[:2]
    out = rgba
    if b["sx"]:
        mid = b["l"] + (w - b["l"] - b["r"]) // 2
        out = np.concatenate([out[:, :b["l"]], out[:, mid - keep // 2: mid - keep // 2 + keep],
                              out[:, w - b["r"]:]], axis=1)
    if b["sy"]:
        hh = out.shape[0]
        mid = b["t"] + (hh - b["t"] - b["b"]) // 2
        out = np.concatenate([out[:b["t"]], out[mid - keep // 2: mid - keep // 2 + keep],
                              out[hh - b["b"]:]], axis=0)
    return np.ascontiguousarray(out), {"l": b["l"] if b["sx"] else 0, "r": b["r"] if b["sx"] else 0,
                                       "t": b["t"] if b["sy"] else 0, "b": b["b"] if b["sy"] else 0}
