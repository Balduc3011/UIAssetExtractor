"""Registers pipeline steps as background jobs."""
import numpy as np
from PIL import Image

from .. import store
from ..jobs import handler
from . import analyze, export, extract, qa, regen
from .detect import detect_groups, run_ocr


@handler("detect")
def _detect(pid, params, ctx):
    d = store.pdir(pid)
    rgb = np.array(Image.open(d / "source.png").convert("RGB"))
    ctx.progress(0, 2, "Tìm vùng UI (OpenCV)…")
    boxes = detect_groups(rgb)
    ctx.progress(1, 2, "Đọc chữ (OCR)…")
    ocr = run_ocr(rgb)

    def f(p):
        p["assets"] = [store.make_asset(name=f"group_{i}", type="group", bbox=b)
                       for i, b in enumerate(boxes)]
        p["ocr"] = ocr
        p["groups"] = boxes
        p["screen"] = {}
        p["stage"] = "detected"
    store.update(pid, f)
    ctx.progress(1, 1, f"Tìm thấy {len(boxes)} vùng, {len(ocr)} dòng chữ")


handler("analyze")(analyze.run)
handler("extract")(extract.run)
handler("qa")(qa.run)
handler("regen")(regen.run)
handler("export")(export.run)


@handler("auto")
def _auto(pid, params, ctx):
    """Detect → Claude analyze → extract → QA in one go."""
    steps = [("detect", _detect), ("analyze", analyze.run), ("extract", extract.run)]
    if params.get("qa", True):
        steps.append(("qa", qa.run))
    for i, (name, fn) in enumerate(steps):
        ctx.log(f"== {name} ==")
        sub = _Sub(ctx, i, len(steps))
        fn(pid, {}, sub)
    ctx.progress(1, 1, "Hoàn tất pipeline tự động")


class _Sub:
    def __init__(self, ctx, i, n):
        self.ctx, self.i, self.n = ctx, i, n

    def progress(self, done, total, msg=""):
        frac = (done / total) if total else 0
        self.ctx.progress(self.i + frac, self.n, f"[{self.i + 1}/{self.n}] {msg}" if msg else "")

    def log(self, msg):
        self.ctx.log(msg)
