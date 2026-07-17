"""Rule Planner agent — the LLM proposes, the validator disposes.

Sends the LLM only: the computed profile (aggregate stats) + a few sample
rows. Receives back a cleaning plan as strict JSON. Every plan item is
validated against the closed op registry before anything can run.
"""

from __future__ import annotations

import json

import pandas as pd

from .llm import generate
from .ops import REGISTRY

PROMPT_TEMPLATE = """You are the Rule Planner agent in a Data Quality Auto-Fixer system.
Based on the data profile below, propose a cleaning plan.

STRICT RULES:
- Reply with a JSON array ONLY (no prose, no markdown fences).
- Each item: {{"op": "<op_name>", "column": "<column or null>", "params": {{}}, "reason": "<short reason in Arabic>"}}
- Allowed ops (use ONLY these): {allowed_ops}
- Op semantics: {op_descriptions}
- Order matters: standardize_nulls and unify_numerals before parsing; drop_exact_duplicates last.
- Only propose ops that the profile actually justifies. Do not invent problems.
- For map_values, only propose it when the sample clearly shows variants of the same real-world value, and put the exact mapping in params.mapping.

DATA PROFILE (computed with pandas — trustworthy):
{profile_json}

SAMPLE ROWS (first {n_samples} rows only):
{sample_json}
"""


def build_plan(profile: dict, df: pd.DataFrame, n_samples: int = 5) -> tuple[list[dict], list[str]]:
    """Ask the LLM for a plan, validate it, return (valid_plan, rejected_notes)."""
    prompt = PROMPT_TEMPLATE.format(
        allowed_ops=", ".join(REGISTRY.keys()),
        op_descriptions="; ".join(f"{k}: {v['desc']}" for k, v in REGISTRY.items()),
        profile_json=json.dumps(profile, ensure_ascii=False, default=str),
        n_samples=n_samples,
        sample_json=df.head(n_samples).to_json(orient="records", force_ascii=False),
    )
    raw = generate(prompt)
    return validate_plan(parse_json_array(raw), df)


def parse_json_array(raw: str) -> list[dict]:
    """Extract the first JSON array from the LLM reply (tolerates fences/prose)."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in LLM reply: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def validate_plan(plan: list[dict], df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Whitelist gate: reject unknown ops, missing columns, bad params."""
    valid, rejected = [], []
    for item in plan:
        op = item.get("op")
        if op not in REGISTRY:
            rejected.append(f"عملية غير معروفة رُفضت: {op}")
            continue
        if REGISTRY[op]["needs_column"]:
            col = item.get("column")
            if col not in df.columns:
                rejected.append(f"{op}: العمود '{col}' غير موجود — رُفضت")
                continue
        if op == "map_values" and not isinstance((item.get("params") or {}).get("mapping"), dict):
            rejected.append("map_values بدون mapping صريح — رُفضت")
            continue
        valid.append(item)
    return valid, rejected
