"""Thin OpenRouter (OpenAI-compatible) chat wrapper, keyed by role.

Every lane calls chat(role=...) so competition-day model swaps are one edit in config.
"""
from . import config

_client = None


def _get():
    global _client
    if _client is None:
        from openai import OpenAI
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "No OpenRouter API key — set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY in .env"
            )
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)
    return _client


def chat(messages, role="synth", model=None, temperature=0.0, max_tokens=1500, **kw) -> str:
    """One chat completion. `role` indexes config.MODELS; `model` overrides it."""
    m = model or config.MODELS.get(role, role)
    r = _get().chat.completions.create(
        model=m, messages=messages, temperature=temperature, max_tokens=max_tokens, **kw
    )
    return (r.choices[0].message.content or "").strip()


def _img_data_url(path) -> str:
    import base64
    import mimetypes
    mt = mimetypes.guess_type(str(path))[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mt};base64,{b64}"


def vlm(prompt, images, role="vlm", model=None, temperature=0.0, max_tokens=1000, **kw) -> str:
    """Vision call: `images` is a path or list of paths, sent alongside the text prompt."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _img_data_url(img)}})
    m = model or config.MODELS.get(role, role)
    r = _get().chat.completions.create(
        model=m, messages=[{"role": "user", "content": content}],
        temperature=temperature, max_tokens=max_tokens, **kw,
    )
    return (r.choices[0].message.content or "").strip()
