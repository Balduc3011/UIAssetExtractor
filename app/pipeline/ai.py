"""Thin wrappers around the Claude and OpenAI APIs (with usage tracking)."""
import base64
import io
import json
import re

from PIL import Image

from .. import store
from ..config import load_settings
from ..secrets_store import get_key
from .imgutil import b64png, to_png_bytes


class ApiError(Exception):
    pass


# ---------------- Claude ----------------

def claude_client():
    key = get_key("anthropic")
    if not key:
        raise ApiError("Chưa có Claude API key (vào Settings).")
    import anthropic
    return anthropic.Anthropic(api_key=key, max_retries=3, timeout=180)


def img_block(im: Image.Image) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64png(im)}}


def parse_json(text: str):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    raise ApiError("Claude trả về không phải JSON hợp lệ.")


def claude_json(pid: str, content: list, system: str, max_tokens=4000):
    s = load_settings()
    c = claude_client()
    r = c.messages.create(
        model=s["claude_model"], max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": content}],
    )
    u = getattr(r, "usage", None)
    store.add_usage(pid, claude_calls=1,
                    claude_in=getattr(u, "input_tokens", 0) or 0,
                    claude_out=getattr(u, "output_tokens", 0) or 0)
    text = "".join(getattr(b, "text", "") for b in r.content)
    return parse_json(text)


# ---------------- OpenAI ----------------

def openai_client():
    key = get_key("openai")
    if not key:
        raise ApiError("Chưa có OpenAI API key (vào Settings).")
    import openai
    return openai.OpenAI(api_key=key, max_retries=2, timeout=300)


def pick_size(w, h) -> str:
    r = w / max(1, h)
    if r > 1.3:
        return "1536x1024"
    if r < 1 / 1.3:
        return "1024x1536"
    return "1024x1024"


def image_edit(pid: str, images: list[Image.Image], prompt: str, size: str,
               mask: Image.Image | None = None, transparent=True) -> Image.Image:
    s = load_settings()
    proj = store.load(pid)
    if proj["usage"].get("images", 0) >= int(s["max_images_per_project"]):
        raise ApiError(f"Đã chạm giới hạn {s['max_images_per_project']} ảnh GPT cho project này "
                       "(chỉnh trong Settings).")
    c = openai_client()
    files = []
    for i, im in enumerate(images):
        f = io.BytesIO(to_png_bytes(im))
        f.name = f"ref{i}.png"
        files.append(f)
    kw = dict(model=s["openai_image_model"], image=files if len(files) > 1 else files[0],
              prompt=prompt, size=size, quality=s["image_quality"], n=1)
    if transparent:
        kw["background"] = "transparent"
        kw["output_format"] = "png"
    if mask is not None:
        mf = io.BytesIO(to_png_bytes(mask))
        mf.name = "mask.png"
        kw["mask"] = mf
    store.add_usage(pid, image_calls=1)
    try:
        r = c.images.edit(**kw)
    except Exception as e:
        msg = str(e)
        # some models don't support transparent background / output_format
        if transparent and ("background" in msg or "output_format" in msg):
            for f in files:
                f.seek(0)
            kw.pop("background", None)
            kw.pop("output_format", None)
            r = c.images.edit(**kw)
        else:
            raise ApiError(f"OpenAI: {msg}")
    store.add_usage(pid, images=1)
    d = r.data[0]
    if getattr(d, "b64_json", None):
        raw = base64.b64decode(d.b64_json)
    else:
        import urllib.request
        raw = urllib.request.urlopen(d.url).read()
    return Image.open(io.BytesIO(raw)).convert("RGBA")


# ---------------- key checks ----------------

def check_anthropic(key: str) -> dict:
    import anthropic
    c = anthropic.Anthropic(api_key=key, timeout=30)
    ids = [m.id for m in c.models.list(limit=100).data]
    return {"ok": True, "models": ids}


def check_openai(key: str) -> dict:
    import openai
    c = openai.OpenAI(api_key=key, timeout=30)
    ids = [m.id for m in c.models.list().data]
    img = sorted([i for i in ids if "image" in i or i.startswith("dall-e")])
    return {"ok": True, "models": img,
            "note": None if img else "Key hợp lệ nhưng chưa thấy model tạo ảnh. "
                                     "Có thể tổ chức chưa được Verify trên platform.openai.com."}
