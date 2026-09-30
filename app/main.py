"""UI Asset Extractor – local web server (backend + frontend on one port)."""
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import gpu, jobs, secrets_store, store
from .config import DEFAULTS, WEB, load_settings, save_settings
from .pipeline import tasks  # noqa: F401  (registers job handlers)
from .pipeline.ai import check_anthropic, check_openai

APP_NAME = "UI Asset Extractor"
VERSION = "1.0.0"
app = FastAPI(title=APP_NAME)

KEY_LINKS = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
}


# ---------------- misc ----------------

@app.get("/api/ping")
def ping():
    return {"app": APP_NAME, "version": VERSION}


@app.get("/api/settings")
def get_settings():
    return {
        "settings": load_settings(),
        "defaults": DEFAULTS,
        "keys": {n: secrets_store.masked(n) for n in secrets_store.NAMES},
        "key_links": KEY_LINKS,
        "gpu": gpu.status(),
    }


@app.put("/api/settings")
def put_settings(patch: dict = Body(...)):
    return save_settings(patch)


@app.post("/api/keys/{name}")
def set_key(name: str, body: dict = Body(...)):
    if name not in secrets_store.NAMES:
        raise HTTPException(404)
    key = (body.get("key") or "").strip()
    if not key:
        raise HTTPException(400, "Key trống")
    try:
        info = check_anthropic(key) if name == "anthropic" else check_openai(key)
    except Exception as e:
        raise HTTPException(400, f"Key không hợp lệ hoặc không gọi được API: {e}")
    where = secrets_store.set_key(name, key)
    _auto_pick_model(name, info.get("models") or [])
    return {"stored": where, "masked": secrets_store.masked(name), **info}


def _auto_pick_model(name, models):
    """If the configured model isn't available on this key, pick a sensible one."""
    s = load_settings()
    field = "claude_model" if name == "anthropic" else "openai_image_model"
    if not models or s[field] in models:
        return
    if name == "anthropic":
        pref = [m for m in models if "sonnet" in m] or models
    else:
        order = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"]
        pref = [m for m in order if m in models] or [m for m in models if "image" in m] or models
    save_settings({field: pref[0]})


@app.post("/api/keys/{name}/test")
def test_key(name: str):
    key = secrets_store.get_key(name)
    if not key:
        raise HTTPException(400, "Chưa có key")
    try:
        return check_anthropic(key) if name == "anthropic" else check_openai(key)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/keys/{name}")
def del_key(name: str):
    secrets_store.delete_key(name)
    return {"ok": True}


# ---------------- projects ----------------

@app.get("/api/projects")
def projects():
    return store.list_all()


@app.post("/api/projects")
async def create_project(file: UploadFile = File(...), name: str = Form("")):
    data = await file.read()
    try:
        p = store.create(name or Path(file.filename or "screen").stem, data)
    except Exception as e:
        raise HTTPException(400, f"Không đọc được ảnh: {e}")
    return p


def _load(pid):
    try:
        return store.load(pid)
    except FileNotFoundError:
        raise HTTPException(404, "Không tìm thấy project")


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _load(pid)


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, body: dict = Body(...)):
    _load(pid)
    return store.update(pid, lambda p: p.update({k: v for k, v in body.items() if k in ("name",)}))


@app.delete("/api/projects/{pid}")
def del_project(pid: str):
    store.delete(pid)
    return {"ok": True}


EDITABLE = {"name", "type", "bbox", "z", "keep", "remove_text", "nine_slice", "method",
            "description", "text", "active", "dup_of", "borders"}


@app.patch("/api/projects/{pid}/assets/{aid}")
def patch_asset(pid: str, aid: str, body: dict = Body(...)):
    _load(pid)

    def f(p):
        a = store.get_asset(p, aid)
        if not a:
            raise HTTPException(404)
        for k, v in body.items():
            if k in EDITABLE:
                a[k] = v
        if "bbox" in body:
            W, H = p["width"], p["height"]
            from .pipeline.imgutil import clamp_box, inter_frac
            a["bbox"] = clamp_box(a["bbox"], W, H)
            a["text_boxes"] = [t["bbox"] for t in p.get("ocr", [])
                               if inter_frac(t["bbox"], a["bbox"]) > 0.6]
        return a
    return store.update(pid, f)


@app.post("/api/projects/{pid}/assets/bulk")
def bulk_asset(pid: str, body: dict = Body(...)):
    ids = set(body.get("ids", []))
    patch = {k: v for k, v in body.get("patch", {}).items() if k in EDITABLE - {"bbox", "name"}}

    def f(p):
        if body.get("delete"):
            p["assets"] = [a for a in p["assets"] if a["id"] not in ids]
        else:
            for a in p["assets"]:
                if a["id"] in ids:
                    a.update(patch)
    store.update(pid, f)
    return _load(pid)


@app.post("/api/projects/{pid}/assets")
def add_asset(pid: str, body: dict = Body(...)):
    def f(p):
        from .pipeline.imgutil import clamp_box, inter_frac
        b = clamp_box(body.get("bbox", [0, 0, 32, 32]), p["width"], p["height"])
        names = {a["name"] for a in p["assets"]}
        n, i = body.get("name") or "asset", 1
        base = n
        while n in names:
            i += 1
            n = f"{base}_{i}"
        a = store.make_asset(name=n, type=body.get("type", "other"), bbox=b,
                             z=int(body.get("z", 0)))
        a["text_boxes"] = [t["bbox"] for t in p.get("ocr", []) if inter_frac(t["bbox"], b) > 0.6]
        p["assets"].append(a)
        return a
    return store.update(pid, f)


@app.delete("/api/projects/{pid}/assets/{aid}")
def del_asset(pid: str, aid: str):
    store.update(pid, lambda p: p.__setitem__(
        "assets", [a for a in p["assets"] if a["id"] != aid]))
    return {"ok": True}


# ---------------- jobs ----------------

@app.post("/api/projects/{pid}/jobs")
def start_job(pid: str, body: dict = Body(...)):
    _load(pid)
    try:
        j = jobs.submit(body.get("kind", ""), pid, body.get("params") or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return jobs.public(j)


@app.get("/api/jobs")
def list_jobs(project: str | None = None):
    js = [jobs.public(j) for j in jobs.JOBS.values() if not project or j["project"] == project]
    return sorted(js, key=lambda j: -j["created"])[:30]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return jobs.public(j)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    j = jobs.JOBS.get(jid)
    if j:
        j["cancel"] = True
    return {"ok": True}


# ---------------- files ----------------

@app.get("/files/{pid}/{path:path}")
def files(pid: str, path: str):
    base = store.pdir(pid).resolve()
    f = (base / path).resolve()
    if base not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.post("/api/projects/{pid}/open-folder")
def open_folder(pid: str, body: dict = Body(default={})):
    base = store.pdir(pid).resolve()
    f = (base / body.get("path", "")).resolve()
    if f != base and base not in f.parents:
        raise HTTPException(400)
    if not f.exists():
        raise HTTPException(404)
    try:
        if sys.platform == "win32":
            os.startfile(str(f))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(f)])
        else:
            subprocess.Popen(["xdg-open", str(f)])
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "path": str(f)}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/web", StaticFiles(directory=WEB), name="web")


@app.exception_handler(Exception)
async def err(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=500)


# ---------------- launcher ----------------

def _port_free(p):
    """Free = nothing is listening on it (connect test; ignores TIME_WAIT leftovers)."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", p)) != 0


def _is_us(p):
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/ping", timeout=1) as r:
            return json.loads(r.read()).get("app") == APP_NAME
    except Exception:
        return False


def main():
    import uvicorn
    base = int(os.environ.get("UIAE_PORT", "8765"))
    no_browser = "--no-browser" in sys.argv
    if _is_us(base):
        print(f"Tool dang chay san tai http://127.0.0.1:{base}")
        if not no_browser:
            webbrowser.open(f"http://127.0.0.1:{base}")
        return
    port = next(p for p in range(base, base + 50) if _port_free(p))
    url = f"http://127.0.0.1:{port}"

    def opener():
        for _ in range(100):
            if _is_us(port):
                break
            time.sleep(0.2)
        if not no_browser:
            webbrowser.open(url)
        print(f"\n  {APP_NAME} dang chay: {url}\n  Dong cua so nay de tat tool.\n")

    threading.Thread(target=opener, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
