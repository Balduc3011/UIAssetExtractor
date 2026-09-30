"""Paths and user settings."""
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROJECTS = DATA / "projects"
MODELS = ROOT / "models"
WEB = ROOT / "web"
SETTINGS_FILE = DATA / "settings.json"

for p in (DATA, PROJECTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# Keep downloaded AI models inside the project folder (portable).
os.environ.setdefault("U2NET_HOME", str(MODELS / "rembg"))

DEFAULTS = {
    "ai_provider": "cowork",            # cowork (Claude app, no API key) | api
    "claude_model": "claude-sonnet-5-5",
    "openai_image_model": "gpt-image-2",
    "image_quality": "medium",          # low | medium | high
    "gpu_mode": "eco",                  # off | eco | full
    "matte_method": "grabcut",          # grabcut | ai | none
    "ai_matte_model": "isnet-general-use",
    "qa_threshold": 7,                  # 0..10
    "auto_qa_after_regen": True,
    "max_regen_retries": 1,
    "max_images_per_project": 40,       # hard budget stop for GPT images
    "claude_concurrency": 4,
    "export_scale": 2,
    "atlas_max_size": 2048,
    "atlas_padding": 2,
    "skip_duplicates": True,
    "compact_nine_slice": True,
    "strip_all_text": True,             # never keep text / numbers on exported art         # export 9-slice sprites with the middle collapsed
}

_lock = threading.Lock()


def load_settings() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text("utf-8")))
        except Exception:
            pass
    return s


def save_settings(patch: dict) -> dict:
    with _lock:
        s = load_settings()
        for k, v in patch.items():
            if k in DEFAULTS:
                s[k] = v
        SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), "utf-8")
        return s
