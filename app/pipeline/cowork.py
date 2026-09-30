"""'Claude app (Cowork)' mode: instead of calling the Claude API, write a task folder
(images + TASK.md) and wait for Claude in the Cowork app to write result.json."""
import json
import time

from .. import store
from ..config import DATA

TASKS = DATA / "cowork_tasks"

HOWTO = ("Mở Claude (Cowork) có kết nối thư mục chứa tool và nhắn: "
         "\"Làm task UIAssetExtractor\"")


def create_task(pid: str, kind: str, images: dict, task_md: str, meta: dict):
    """images: {filename: PIL.Image}. Returns task folder Path."""
    TASKS.mkdir(parents=True, exist_ok=True)
    tid = f"{kind}_{pid}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = TASKS / tid
    d.mkdir()
    for fn, im in images.items():
        im.save(d / fn)
    (d / "meta.json").write_text(json.dumps({"task": tid, "kind": kind, "project": pid,
                                             "created": time.time(), **meta},
                                            indent=1, ensure_ascii=False), "utf-8")
    (d / "TASK.md").write_text(task_md, "utf-8")
    return d


def wait_result(d, ctx, what: str):
    """Block the job until result.json appears (user can cancel the job)."""
    res = d / "result.json"
    err = d / "result_error.txt"
    t0 = time.time()
    try:
        return _wait(d, res, err, t0, ctx, what)
    except BaseException:
        try:
            d.rename(d.parent / ("old_" + d.name))  # cancelled -> not pending anymore
        except OSError:
            pass
        raise


def _wait(d, res, err, t0, ctx, what):
    while True:
        mins = int((time.time() - t0) // 60)
        ctx.progress(0, 1, f"Chờ Claude (Cowork) {what}… {mins} phút. {HOWTO}")
        if res.exists():
            time.sleep(0.5)  # let the writer finish
            try:
                from .ai import parse_json
                data = parse_json(res.read_text("utf-8-sig"))
                if err.exists():
                    err.unlink()
                done = d.parent / ("done_" + d.name)
                try:
                    d.rename(done)
                except OSError:
                    pass
                return data
            except Exception as e:
                err.write_text(f"result.json không đọc được: {e}. Hãy ghi lại file.", "utf-8")
                res.rename(d / f"result_bad_{int(time.time())}.json")
        time.sleep(2)


def pending() -> list[dict]:
    out = []
    if not TASKS.exists():
        return out
    for d in sorted(TASKS.iterdir()):
        if d.is_dir() and not d.name.startswith("done_") and not (d / "result.json").exists():
            try:
                out.append(json.loads((d / "meta.json").read_text("utf-8")) | {"folder": str(d)})
            except Exception:
                pass
    return out


def cleanup_stale(pid: str):
    """Mark older unfinished tasks of this project as abandoned (a new one replaces them)."""
    if not TASKS.exists():
        return
    for d in TASKS.iterdir():
        if d.is_dir() and not d.name.startswith(("done_", "old_")) and f"_{pid}_" in d.name:
            try:
                d.rename(d.parent / ("old_" + d.name))
            except OSError:
                pass


def add_usage(pid):
    store.add_usage(pid, claude_calls=1)
