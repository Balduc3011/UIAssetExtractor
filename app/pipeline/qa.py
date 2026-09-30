"""Step 4 – Claude QA: compare original crop vs produced asset, score 0-10."""
from PIL import Image

from .. import store
from ..config import load_settings
from .ai import claude_json, img_block
from .imgutil import on_checker

SYSTEM = "You are a strict game-art QA reviewer. Answer ONLY with JSON."

PROMPT = """Each image shows one sprite asset: LEFT = original crop from the UI screenshot,
RIGHT = the extracted/redrawn asset on a checkerboard (checkerboard = transparent).
Expected per asset is given in its caption. Judge each RIGHT image as a production sprite:
- same shape / colors / style as LEFT (ignore other UI pieces that overlap in LEFT)
- complete (nothing cut off, no holes), clean transparent edges, no leftover backdrop
- no text if caption says "no text"; no extra elements (badges etc.) unless expected
Score 0-10 (10 = ship it). recommend: "ok" | "reextract" (local cleanup could fix) |
"regenerate" (needs redraw). hint = one short instruction to fix it when redrawing.
Return JSON: {"results": [{"id": "...", "score": 0, "issues": ["..."], "recommend": "ok", "hint": ""}]}"""


def panel(pid, a) -> Image.Image | None:
    d = store.pdir(pid)
    if a["active"] is None:
        return None
    try:
        crop = Image.open(d / "crops" / f"{a['id']}.png").convert("RGBA")
        res = Image.open(d / a["versions"][a["active"]]["file"]).convert("RGBA")
    except Exception:
        return None
    H = 256

    def fit(im):
        s = H / im.height
        w = max(1, round(im.width * s))
        if w > 384:
            s = 384 / im.width
        return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)

    l, r = fit(crop), on_checker(fit(res))
    out = Image.new("RGBA", (l.width + r.width + 12, max(l.height, r.height)), (40, 40, 40, 255))
    out.alpha_composite(l, (0, 0))
    out.alpha_composite(r, (l.width + 12, 0))
    return out.convert("RGB")


COWORK_TASK = """# UIAssetExtractor – task QA

Bạn là Claude trong app Cowork. Chấm điểm các sprite asset do tool trích xuất.
Mỗi file `qa_<id>.png`: TRÁI = crop gốc từ screenshot, PHẢI = asset đã tách trên nền
caro (caro = trong suốt). Danh sách asset và yêu cầu mong đợi:

{items}

Tiêu chí: giống bản gốc về hình/màu/style (bỏ qua phần UI khác đè lên ở bản gốc),
đầy đủ không bị cắt/thủng, viền trong suốt sạch, không còn nền, không chữ nếu yêu cầu
"no text", không thừa phần tử khác. Điểm 0–10 (10 = dùng được ngay).
recommend: "ok" | "reextract" (xử lý local sửa được) | "regenerate" (cần vẽ lại).
hint: 1 câu ngắn (tiếng Anh) hướng dẫn sửa khi vẽ lại.

Ghi `result.json` vào CHÍNH thư mục này (chỉ JSON):
{{"results": [{{"id": "<id>", "score": 8, "issues": ["..."], "recommend": "ok", "hint": ""}}]}}
Phải có đủ mọi id trong danh sách.
"""


def _items(pid, ids):
    proj = store.load(pid)
    out = []
    for a in proj["assets"]:
        if a["id"] not in ids or a["active"] is None:
            continue
        p = panel(pid, a)
        if p is None:
            continue
        exp = [a["type"]]
        if a["remove_text"]:
            exp.append("no text")
        cap = (f'id={a["id"]} name={a["name"]} expected: {", ".join(exp)}; '
               f'{a["description"][:200]}')
        out.append((a, p, cap))
    return out


def _store_results(pid, results):
    def f(p):
        for a in p["assets"]:
            if a["id"] in results:
                a["qa"] = results[a["id"]]
    store.update(pid, f)


def _norm(x):
    return {"score": float(x.get("score", 0)), "issues": list(x.get("issues") or [])[:5],
            "recommend": x.get("recommend", ""), "hint": x.get("hint", "")}


def qa_assets(pid: str, ids: list[str], ctx=None) -> dict:
    if load_settings().get("ai_provider") == "cowork":
        return qa_cowork(pid, ids, ctx)
    items = _items(pid, ids)
    results = {}
    B = 6
    for i in range(0, len(items), B):
        batch = items[i:i + B]
        if ctx:
            ctx.progress(i, len(items), f"Claude QA {i + 1}-{i + len(batch)}/{len(items)}")
        content = []
        for a, p, cap in batch:
            content.append({"type": "text", "text": cap})
            content.append(img_block(p))
        content.append({"type": "text", "text": PROMPT})
        r = claude_json(pid, content, SYSTEM, max_tokens=2500)
        for x in r.get("results", []):
            if isinstance(x, dict) and x.get("id"):
                results[x["id"]] = _norm(x)
    _store_results(pid, results)
    return results


def qa_cowork(pid, ids, ctx):
    from . import cowork
    if ctx is None:
        return {}
    items = _items(pid, ids)
    if not items:
        return {}
    images = {f"qa_{a['id']}.png": p for a, p, _ in items}
    lines = "\n".join(f"- `qa_{a['id']}.png` — {cap}" for a, _, cap in items)
    cowork.cleanup_stale(pid)
    d = cowork.create_task(pid, "qa", images, COWORK_TASK.format(items=lines),
                           {"assets": [a["id"] for a, _, _ in items]})
    r = cowork.wait_result(d, ctx, f"chấm QA {len(items)} asset")
    cowork.add_usage(pid)
    results = {x["id"]: _norm(x) for x in r.get("results", [])
               if isinstance(x, dict) and x.get("id")}
    _store_results(pid, results)
    return results


def run(pid, params, ctx):
    proj = store.load(pid)
    ids = params.get("asset_ids") or [a["id"] for a in proj["assets"]
                                      if a["keep"] and not a["dup_of"] and a["active"] is not None
                                      and a["type"] != "background"]
    r = qa_assets(pid, ids, ctx)
    th = load_settings()["qa_threshold"]
    bad = sum(1 for v in r.values() if v["score"] < th)
    ctx.progress(1, 1, f"QA xong: {len(r)} asset, {bad} dưới ngưỡng {th}")
