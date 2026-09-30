"""API keys: stored in the OS credential store (Windows Credential Manager).
Falls back to data/secrets.json only if the OS store is unavailable."""
import json

from .config import DATA

SERVICE = "UIAssetExtractor"
FALLBACK = DATA / "secrets.json"
NAMES = ("anthropic", "openai")


def _kr():
    try:
        import keyring
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _fb_read() -> dict:
    try:
        return json.loads(FALLBACK.read_text("utf-8"))
    except Exception:
        return {}


def get_key(name: str) -> str | None:
    kr = _kr()
    if kr:
        try:
            v = kr.get_password(SERVICE, name)
            if v:
                return v
        except Exception:
            pass
    return _fb_read().get(name)


def set_key(name: str, value: str) -> str:
    """Returns where the key was stored: 'keyring' or 'file'."""
    kr = _kr()
    if kr:
        try:
            kr.set_password(SERVICE, name, value)
            return "keyring"
        except Exception:
            pass
    d = _fb_read()
    d[name] = value
    FALLBACK.write_text(json.dumps(d), "utf-8")
    return "file"


def delete_key(name: str):
    kr = _kr()
    if kr:
        try:
            kr.delete_password(SERVICE, name)
        except Exception:
            pass
    d = _fb_read()
    if name in d:
        d.pop(name)
        FALLBACK.write_text(json.dumps(d), "utf-8")


def masked(name: str) -> str | None:
    v = get_key(name)
    if not v:
        return None
    return v[:7] + "…" + v[-4:]
