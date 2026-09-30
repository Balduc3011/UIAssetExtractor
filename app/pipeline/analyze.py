"""Step 2 – Claude: label groups (Set-of-Mark) and split each group into assets."""
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .. import store
from ..config import load_settings
from .ai import claude_json, img_block
from .detect import snap_box
from .imgutil import clamp_box, draw_grid, draw_marks, inter_frac, iou, pad_box, upscale_for_view

TYPES = ["icon", "button", "panel", "label_plate", "badge", "frame", "bar", "currency",
         "avatar", "decoration", "background", "text", "other"]

SYSTEM = (
    "You are a senior game UI artist who slices mobile game UI mockups into production "
    "sprite assets (like a TexturePacker sprite sheet). You answer ONLY with JSON."
)

GLOBAL_PROMPT = """This is a game UI screenshot. Candidate regions are drawn as numbered boxes.
Tasks:
1. For each numbered box decide keep=true if it contains game UI art worth exporting,
   keep=false for ads, OS/system bars, pure backdrop, boxes that contain only text, and
   boxes that are just a duplicate of another box. Boxes can be nested (a bar and the
   icons on it): keep both levels when both hold art.
2. Give each kept box a short snake_case label (e.g. "shop_gift_button").
3. List clearly visible UI elements NOT covered by any box in "missing", with bbox as
   [x0,y0,x1,y1] normalized 0-1000 over the full image.
4. Describe the overall art style in one sentence (used later to redraw assets).

Return JSON:
{"style": "...", "background": "short description of the backdrop",
 "groups": [{"id": 0, "keep": true, "label": "..."}],
 "missing": [{"label": "...", "bbox": [x0,y0,x1,y1]}]}"""

GROUP_PROMPT = """This crop shows ONE region of a game UI ("{label}"). Image 1 is the clean crop,
image 2 is the same crop with a 0-1000 normalized coordinate grid.
Split it into separate reusable sprite assets, the way an artist would deliver them:
- a badge/sticker ("!", "NEW", "200%") overlapping an icon is its OWN asset
- an icon and the name plate below it are separate assets
- NO asset keeps text, letters or numbers drawn on it (labels, counters, the "4" on a heart,
  "x2" on a badge...): set remove_text=true and give each text line's box in text_boxes;
  text is rendered by the engine. Exception: lettering that IS the artwork of a logo icon
  (a big "ADS" logo) stays
- plates/panels/bars that stretch should be nine_slice=true
- pure text on the backdrop: type "text", export=false
- identical repeated pieces still get listed (duplicates are merged later)
- if this region is a container (bar/panel) whose inner elements are clearly separate
  pieces, list the container's own art (the bar/panel itself, with nine_slice) AND the
  inner elements; duplicates across regions are merged later
Coordinates: bbox = [x0,y0,x1,y1] normalized 0-1000 relative to THIS crop, tight around
the element's visible pixels. z = stacking order (higher = drawn on top).
description = precise visual description (shape, colors, outline, material) so an
artist could redraw it identically, WITHOUT mentioning the text content.

Return JSON: {{"parts": [{{"name": "snake_case", "type": one of {types},
 "bbox": [x0,y0,x1,y1], "z": 0, "export": true, "has_text": false, "text": "",
 "remove_text": false, "text_boxes": [[x0,y0,x1,y1]], "nine_slice": false,
 "description": "..."}}]}}"""


def _norm_to_px(nb, region):
    rx, ry, rw, rh = region
    x0, y0, x1, y1 = [max(0, min(1000, float(v))) for v in nb]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [rx + x0 / 1000 * rw, ry + y0 / 1000 * rh,
            max(2, (x1 - x0) / 1000 * rw), max(2, (y1 - y0) / 1000 * rh)]


def run(pid: str, params: dict, ctx):
    s = load_settings()
    if s.get("ai_provider") == "cowork":
        return run_cowork(pid, params, ctx)
    return run_api(pid, params, ctx)


def _prepare(pid):
    import numpy as np
    proj = store.load(pid)
    src = Image.open(store.pdir(pid) / "source.png").convert("RGB")
    rgb = np.array(src)
    groups = [a for a in proj["assets"] if a["type"] == "group"] or \
        [{"bbox": b} for b in proj.get("groups", [])]
    if not groups:
        raise RuntimeError("Chưa có vùng nào. Chạy bước Detect trước.")
    return proj, src, rgb, groups


def _marked(src, boxes):
    W, H = src.size
    scale = min(1.0, 1568 / max(W, H))
    view = src if scale == 1 else src.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    return draw_marks(view, [[v * scale for v in b] for b in boxes], list(range(len(boxes))))


def _region(bbox, W, H):
    return pad_box(bbox, max(4, int(0.06 * max(bbox[2:]))), W, H)


def run_api(pid: str, params: dict, ctx):
    s = load_settings()
    proj, src, rgb, groups = _prepare(pid)
    W, H = src.size
    boxes = [g["bbox"] for g in groups]

    # ---- 1. global pass ----
    ctx.progress(0, 1, "Claude: đọc toàn màn hình…")
    g = claude_json(pid, [img_block(_marked(src, boxes)), {"type": "text", "text": GLOBAL_PROMPT}],
                    SYSTEM)
    info = {x.get("id"): x for x in g.get("groups", []) if isinstance(x, dict)}
    regions = []
    for i, gr in enumerate(groups):
        gi = info.get(i, {"keep": True, "label": f"group_{i}"})
        if gi.get("keep", True):
            regions.append({"label": gi.get("label") or f"group_{i}", "bbox": gr["bbox"]})
    for m in g.get("missing", []) or []:
        try:
            b = _norm_to_px(m["bbox"], [0, 0, W, H])
            regions.append({"label": m.get("label", "missing"),
                            "bbox": clamp_box(snap_box(rgb, clamp_box(b, W, H)), W, H)})
        except Exception:
            pass

    # ---- 2. per-region decomposition (parallel) ----
    def work(reg):
        region = _region(reg["bbox"], W, H)
        rx, ry, rw, rh = region
        crop = upscale_for_view(src.crop((rx, ry, rx + rw, ry + rh)), 640, min_short=320)
        prompt = GROUP_PROMPT.format(label=reg["label"], types="|".join(TYPES))
        r = claude_json(pid, [img_block(crop), img_block(draw_grid(crop)),
                              {"type": "text", "text": prompt}], SYSTEM)
        return reg, region, r.get("parts", [])

    total = len(regions)
    done = 0
    results = []
    with ThreadPoolExecutor(max_workers=int(s["claude_concurrency"])) as ex:
        for res in ex.map(lambda r: _safe(work, r), regions):
            done += 1
            ctx.progress(done, total + 1, f"Claude: tách vùng {done}/{total}")
            results.append(res)
    _build(pid, src, rgb, results, g, ctx)


COWORK_TASK = """# UIAssetExtractor – task phân tích (analyze)

Bạn là Claude trong app Cowork. Người dùng chạy tool UIAssetExtractor và cần bạn
tách screenshot UI game thành các sprite asset. Làm ĐÚNG các bước:

1. Xem `screen_marked.png`: ảnh toàn màn hình, các vùng ứng viên được đánh số 0..{last}.
2. Với MỖI vùng `i`, xem `group_{{i}}.png`: crop phóng to của vùng đó, có lưới toạ độ
   chuẩn hoá 0–1000 (vạch mỗi 100) để bạn đọc toạ độ.
3. Ghi file `result.json` vào CHÍNH thư mục này (UTF-8, chỉ JSON, không markdown).
   Ghi một lần, đầy đủ. Tool đang chờ file này và sẽ tự chạy tiếp.

## Quy tắc tách (như artist giao sprite sheet)
- keep=false cho: quảng cáo, thanh hệ thống, vùng chỉ là nền, vùng chỉ có chữ, phần tử bị cắt
  mép ảnh, vùng trùng hệt vùng khác.
- Các vùng có thể lồng nhau (thanh bar và các icon trên nó): liệt kê bản thân container
  (bar/panel, nine_slice) và các phần tử con; trùng lặp giữa các vùng sẽ được gộp sau.
- Badge/sticker đè lên icon ("!", "NEW", "200%") là asset RIÊNG, z cao hơn icon.
- Icon và bảng tên (name plate) bên dưới là 2 asset riêng.
- KHÔNG asset nào được giữ chữ hoặc số vẽ trên nó (nhãn, bộ đếm, số "4" trên trái tim, "x2" trên
  badge...): remove_text=true và ghi khung từng dòng chữ vào text_boxes (chuẩn hoá 0–1000 theo
  ảnh group_i.png, như bbox; với "missing" thì theo toàn ảnh), khung rộng hơn chữ một chút.
  Ngoại lệ: chữ là chính hình vẽ của logo (icon "ADS") thì giữ. Chữ do engine render.
- Plate/bar/panel kéo giãn được: nine_slice=true.
- Chữ nằm trực tiếp trên nền: type "text", export=false.
- Các mảnh giống nhau lặp lại vẫn liệt kê (tool tự gộp trùng).
- bbox = [x0,y0,x1,y1] chuẩn hoá 0–1000 THEO ẢNH group_i.png, ôm sát pixel của phần tử.
- z = thứ tự lớp (cao hơn = nằm trên). description = mô tả hình ảnh chi tiết (hình dạng,
  màu, viền, chất liệu) đủ để vẽ lại y hệt, KHÔNG nhắc nội dung chữ.
- type thuộc: {types}

## Định dạng result.json
{{
  "style": "1 câu mô tả art style chung",
  "background": "mô tả ngắn nền phía sau",
  "groups": [
    {{"id": 0, "keep": true, "label": "snake_case_label",
      "parts": [
        {{"name": "snake_case", "type": "icon", "bbox": [x0,y0,x1,y1], "z": 0,
          "export": true, "has_text": false, "text": "", "remove_text": false,
          "text_boxes": [[x0,y0,x1,y1]],
          "nine_slice": false, "description": "..."}}
      ]}}
  ],
  "missing": [
    {{"name": "snake_case", "type": "icon", "bbox": [x0,y0,x1,y1], "z": 0,
      "remove_text": false, "text_boxes": [], "nine_slice": false,
      "description": "phần tử UI rõ ràng KHÔNG nằm trong vùng nào; bbox chuẩn hoá 0–1000 theo screen_marked.png"}}
  ]
}}
Phải có đủ mọi id từ 0 đến {last} trong "groups".
"""


def run_cowork(pid: str, params: dict, ctx):
    from . import cowork
    proj, src, rgb, groups = _prepare(pid)
    W, H = src.size
    boxes = [g["bbox"] for g in groups]
    images = {"screen_marked.png": _marked(src, boxes)}
    regions = []
    for i, gr in enumerate(groups):
        region = _region(gr["bbox"], W, H)
        rx, ry, rw, rh = region
        crop = upscale_for_view(src.crop((rx, ry, rx + rw, ry + rh)), 512, min_short=280)
        images[f"group_{i}.png"] = draw_grid(crop)
        regions.append(region)
    cowork.cleanup_stale(pid)
    d = cowork.create_task(pid, "analyze", images,
                           COWORK_TASK.format(last=len(groups) - 1, types="|".join(TYPES)),
                           {"project_name": proj["name"], "groups": len(groups),
                            "regions": regions})
    ctx.log(f"Đã tạo task: {d}")
    g = cowork.wait_result(d, ctx, "phân tích ảnh")
    cowork.add_usage(pid)
    info = {x.get("id"): x for x in g.get("groups", []) if isinstance(x, dict)}
    results = []
    for i, gr in enumerate(groups):
        gi = info.get(i, {"keep": True, "label": f"group_{i}", "parts": []})
        if not gi.get("keep", True):
            continue
        reg = {"label": gi.get("label") or f"group_{i}", "bbox": gr["bbox"]}
        results.append((reg, regions[i], gi.get("parts") or []))
    for m in g.get("missing", []) or []:
        if isinstance(m, dict) and m.get("bbox"):
            results.append(({"label": m.get("name", "missing"), "bbox": [0, 0, W, H]},
                             [0, 0, W, H], [m]))
    _build(pid, src, rgb, results, g, ctx)


def _build(pid, src, rgb, results, g, ctx):
    W, H = src.size
    assets = []
    used = set()
    for res in results:
        if isinstance(res, Exception):
            ctx.log(f"Lỗi 1 vùng: {res}")
            continue
        reg, region, parts = res
        if not parts:
            parts = [{"name": reg["label"], "type": "other", "bbox": [0, 0, 1000, 1000]}]
        for p in parts:
            if not isinstance(p, dict):
                continue
            try:
                b = clamp_box(_norm_to_px(p["bbox"], region), W, H)
            except Exception:
                continue
            b = clamp_box(snap_box(rgb, b), W, H)
            # nested regions can yield the same element twice
            if any(iou(b, o["bbox"]) > 0.8 for o in assets):
                continue
            name = _uniq(str(p.get("name") or "asset"), used)
            t = p.get("type") if p.get("type") in TYPES else "other"
            ai_tb = []
            for tb in p.get("text_boxes") or []:
                try:
                    ai_tb.append(clamp_box(_norm_to_px(tb, region), W, H))
                except Exception:
                    pass
            a = store.make_asset(
                name=name, type=t, bbox=b, z=int(p.get("z") or 0),
                keep=bool(p.get("export", t != "text")) and t != "text",
                remove_text=bool(p.get("remove_text")) or bool(ai_tb) or bool(p.get("has_text")),
                nine_slice=bool(p.get("nine_slice")), ai_text_boxes=ai_tb,
                description=str(p.get("description") or ""), text=str(p.get("text") or ""),
                group=reg["label"],
            )
            assets.append(a)
    # background pseudo-asset
    assets.append(store.make_asset(name="background", type="background", bbox=[0, 0, W, H],
                                   z=-100, keep=False,
                                   description=g.get("background", "")))

    def f(p):
        _attach_text(p, assets)
        p["assets"] = assets
        p["screen"] = {"style": g.get("style", ""), "background": g.get("background", "")}
        p["stage"] = "analyzed"
    store.update(pid, f)
    ctx.progress(1, 1, f"Xong: {len(assets) - 1} asset")


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as e:
        return e


def _uniq(name, used):
    base = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower()).strip("_") or "asset"
    n, i = base, 2
    while n in used:
        n = f"{base}_{i}"
        i += 1
    used.add(n)
    return n


def _attach_text(proj, assets):
    for a in assets:
        if a["type"] == "background":
            continue
        a["text_boxes"] = [t["bbox"] for t in proj.get("ocr", [])
                           if inter_frac(t["bbox"], a["bbox"]) > 0.6]
        if a["text_boxes"] and not a["text"]:
            a["text"] = " ".join(t["text"] for t in proj["ocr"]
                                 if inter_frac(t["bbox"], a["bbox"]) > 0.6)
