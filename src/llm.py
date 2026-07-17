"""LLM client — Gemini via REST (stdlib only, model-agnostic by design).

The LLM's only job in this system: read profiles, propose plans, explain.
It never receives the full dataset and never transforms data itself.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

MODEL = "gemini-flash-latest"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


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


def generate(prompt: str, temperature: float = 0.2) -> str:
    """Single-turn text generation."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": get_api_key()},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]
