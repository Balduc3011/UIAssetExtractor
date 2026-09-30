"""Single background worker: one job at a time (keeps GPU/CPU load low)."""
import queue
import threading
import time
import traceback

from . import store
from .gpu import models

_q: "queue.Queue[dict]" = queue.Queue()
JOBS: dict[str, dict] = {}
HANDLERS = {}


class Cancelled(Exception):
    pass


class Ctx:
    def __init__(self, job):
        self.job = job

    def progress(self, done, total, msg=""):
        self.job["progress"] = round(done / total, 3) if total else 0
        if msg:
            self.job["message"] = msg
        if self.job.get("cancel"):
            raise Cancelled()

    def log(self, msg):
        self.job["log"].append(msg)
        self.job["log"] = self.job["log"][-200:]
        self.job["message"] = msg


def handler(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn
    return deco


def submit(kind: str, pid: str, params: dict | None = None) -> dict:
    if kind not in HANDLERS:
        raise ValueError(f"Unknown job: {kind}")
    job = {
        "id": store.new_id("j"), "kind": kind, "project": pid,
        "params": params or {}, "status": "queued", "progress": 0,
        "message": "Đang chờ…", "log": [], "error": None,
        "created": time.time(), "finished": None,
    }
    JOBS[job["id"]] = job
    _q.put(job)
    return job


def public(job):
    return {k: v for k, v in job.items() if k != "cancel"}


def _worker():
    while True:
        job = _q.get()
        if job.get("cancel"):
            job["status"] = "cancelled"
            continue
        job["status"] = "running"
        try:
            HANDLERS[job["kind"]](job["project"], job["params"], Ctx(job))
            job["status"] = "done"
            job["progress"] = 1
            job["message"] = job["message"] or "Xong"
        except Cancelled:
            job["status"] = "cancelled"
            job["message"] = "Đã huỷ"
        except Exception as e:
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = str(e)
            job["message"] = f"Lỗi: {e}"
        finally:
            job["finished"] = time.time()
            models.release()     # free VRAM between jobs


threading.Thread(target=_worker, daemon=True).start()
