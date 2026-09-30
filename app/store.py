"""Project storage: data/projects/<id>/project.json + image folders."""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from .config import PROJECTS

_locks: dict[str, threading.RLock] = {}
_glock = threading.Lock()

ASSET_DEFAULTS = {
    "name": "asset",
    "type": "other",
    "bbox": [0, 0, 1, 1],        # x, y, w, h in source pixels
    "z": 0,
    "keep": True,                # export it
    "remove_text": False,
    "nine_slice": False,
    "method": None,              # matte method override (None = settings)
    "description": "",
    "text": "",
    "text_boxes": [],
    "ai_text_boxes": [],
    "status": "new",             # new | extracted | regenerated | error
    "versions": [],              # [{file, kind: extract|regen, w, h, scale, created}]
    "active": None,              # index into versions
    "qa": None,                  # {score, issues, recommend, hint}
    "dup_of": None,
    "occluded": 0.0,
    "borders": None,             # 9-slice {l,t,r,b} in source pixels
    "group": None,
    "error": None,
}


def lock(pid: str) -> threading.RLock:
    with _glock:
        return _locks.setdefault(pid, threading.RLock())


def pdir(pid: str) -> Path:
    return PROJECTS / pid


def new_id(prefix="") -> str:
    return prefix + uuid.uuid4().hex[:8]


def create(name: str, image_bytes: bytes) -> dict:
    from PIL import Image
    import io

    pid = new_id("p")
    d = pdir(pid)
    for sub in ("crops", "assets", "regen", "export"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    im = Image.open(io.BytesIO(image_bytes))
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    im.save(d / "source.png")
    th = im.copy()
    th.thumbnail((320, 320))
    th.save(d / "thumb.png")
    proj = {
        "id": pid,
        "name": name,
        "created": time.time(),
        "updated": time.time(),
        "width": im.width,
        "height": im.height,
        "stage": "uploaded",
        "screen": {},          # Claude's global description / style
        "ocr": [],
        "assets": [],
        "usage": {"claude_in": 0, "claude_out": 0, "claude_calls": 0,
                  "images": 0, "image_calls": 0},
        "exports": [],
    }
    save(proj)
    return proj


def load(pid: str) -> dict:
    with lock(pid):
        return json.loads((pdir(pid) / "project.json").read_text("utf-8"))


def save(proj: dict):
    pid = proj["id"]
    with lock(pid):
        proj["updated"] = time.time()
        p = pdir(pid) / "project.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(proj, indent=1, ensure_ascii=False), "utf-8")
        tmp.replace(p)


def update(pid: str, fn):
    """Atomic read-modify-write. fn(proj) mutates in place."""
    with lock(pid):
        proj = load(pid)
        r = fn(proj)
        save(proj)
        return r if r is not None else proj


def list_all() -> list[dict]:
    out = []
    for d in PROJECTS.iterdir():
        f = d / "project.json"
        if f.exists():
            try:
                p = json.loads(f.read_text("utf-8"))
                out.append({k: p[k] for k in ("id", "name", "created", "updated",
                                              "width", "height", "stage")}
                           | {"assets": len(p["assets"])})
            except Exception:
                pass
    return sorted(out, key=lambda x: -x["updated"])


def delete(pid: str):
    shutil.rmtree(pdir(pid), ignore_errors=True)


def make_asset(**kw) -> dict:
    a = json.loads(json.dumps(ASSET_DEFAULTS))
    a.update(kw)
    a["id"] = kw.get("id") or new_id("a")
    return a


def get_asset(proj: dict, aid: str) -> dict | None:
    return next((a for a in proj["assets"] if a["id"] == aid), None)


def add_usage(pid: str, **inc):
    def f(p):
        for k, v in inc.items():
            p["usage"][k] = p["usage"].get(k, 0) + v
    update(pid, f)
