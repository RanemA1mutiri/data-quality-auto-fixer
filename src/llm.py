"""LLM client — Gemini via REST (stdlib only, model-agnostic by design).

The LLM's only job in this system: read profiles, propose plans, explain.
It never receives the full dataset and never transforms data itself.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# AI-only by design (project owner's decision): if every model in this chain
# fails, the system STOPS with a clear message — it never falls back to
# non-AI planning. The chain keeps the AI available through free-tier
# congestion: same API, different serving pools.
MODEL_CHAIN = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash"]


def _endpoint(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def get_api_key() -> str:
    """Resolve the API key: Streamlit secrets → env var → local secrets.toml."""
    try:
        import streamlit as st  # noqa: PLC0415

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]

    secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError("GEMINI_API_KEY not found (Streamlit secrets, env, or .streamlit/secrets.toml)")


def _call_once(model: str, body: dict, api_key: str) -> str:
    req = urllib.request.Request(
        _endpoint(model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates") or []
    if not candidates or "content" not in candidates[0]:
        raise RuntimeError("empty response (safety block or exhausted quota)")
    return candidates[0]["content"]["parts"][0]["text"]


def generate(prompt: str, temperature: float = 0.0) -> str:
    """Single-turn text generation. temperature=0 → reproducible plans.

    Walks the model chain: each model gets one retry with backoff on
    transient errors (429/500/503), then the next model is tried. If the
    whole chain fails, raises a clear error — the system stops (AI-only,
    no silent fallback)."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    api_key = get_api_key()
    errors: list[str] = []
    for model in MODEL_CHAIN:
        for attempt in range(2):
            try:
                return _call_once(model, body, api_key)
            except urllib.error.HTTPError as e:
                errors.append(f"{model}: HTTP {e.code}")
                if e.code in (429, 500, 503) and attempt == 0:
                    time.sleep(2)
                    continue
                break  # this model is down — try the next one
            except Exception as e:
                errors.append(f"{model}: {e}")
                break
    raise RuntimeError(
        "⛔ خدمة الذكاء الاصطناعي غير متاحة حالياً (ربما ازدحام أو اكتمال الحد اليومي المجاني) — "
        "أعيدي المحاولة بعد دقائق. هذا النظام يعمل حصريًا بالتخطيط الذكي.\n"
        f"AI service unavailable — tried: {' | '.join(errors)}"
    )
