"""Rule Planner agent — the LLM proposes, the validator disposes.

Sends the LLM only: the computed profile (aggregate stats) + a few
PII-masked sample rows. Receives back a cleaning plan as strict JSON.
Every plan item is validated against the closed op registry (op name,
target column AND params keys) before anything can run.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .llm import generate
from .ops import REGISTRY

# Columns whose values are personally identifying — masked before any value
# leaves the machine for the LLM. The LLM only needs the *shape* of a value
# (which digits/letters vary), never the real name or phone number.
_PII_HINTS = ("name", "اسم", "mobile", "phone", "جوال", "هاتف", "email", "بريد",
              "id", "هوية", "iban", "حساب", "address", "عنوان")


def _mask_sample(df: pd.DataFrame, n: int) -> str:
    """Return n sample rows as JSON with PII columns shape-masked
    (digits→#, Arabic/Latin letters→x) so no real identifier is sent."""
    sample = df.head(n).copy()
    for col in sample.columns:
        if any(h in str(col).lower() for h in _PII_HINTS):
            sample[col] = sample[col].map(
                lambda v: v if pd.isna(v) else re.sub(r"[A-Za-zء-ي]", "x",
                                                      re.sub(r"\d", "#", str(v)))
            )
    return sample.to_json(orient="records", force_ascii=False)

PROMPT_TEMPLATE = """You are the Rule Planner agent in a Data Quality Auto-Fixer system.
Based on the data profile below, propose a cleaning plan.

STRICT RULES:
- Reply with a JSON array ONLY (no prose, no markdown fences).
- Each item: {{"op": "<op_name>", "column": "<column or null>", "params": {{}}, "reason": "<short reason in Arabic>"}}
- Allowed ops (use ONLY these): {allowed_ops}
- Op semantics: {op_descriptions}
- Order matters: standardize_nulls and unify_numerals before parsing; drop_exact_duplicates last.
- Only propose ops that the profile actually justifies. Do not invent problems.
- For map_values, only propose it when the sample clearly shows variants of the same real-world value, and put the exact string-to-string mapping in params.mapping.

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
        sample_json=_mask_sample(df, n_samples),
    )
    raw = generate(prompt)
    return validate_plan(parse_json_array(raw), df)


# NOTE (project owner's decision, 17 Jul 2026): this system is AI-only.
# There is deliberately NO non-AI fallback planner — if the AI is
# unavailable, the app stops with a clear message. The showcase IS the
# agentic pattern; resilience lives in the model chain (src/llm.py).


OPTIMIZER_TEMPLATE = """You are the Optimizer agent in a Data Quality Auto-Fixer (evaluator-optimizer loop).
A cleaning plan was applied to the raw data, then the Quality Judge MEASURED the remaining
weaknesses below. Produce an IMPROVED, COMPLETE plan — it fully REPLACES the previous plan
and will be applied to the ORIGINAL raw data from scratch.

STRICT RULES:
- Reply with a JSON array ONLY (no prose, no markdown fences).
- Each item: {{"op": "<op_name>", "column": "<column or null>", "params": {{}}, "reason": "<short reason in Arabic>"}}
- Allowed ops (use ONLY these): {allowed_ops}
- Op semantics: {op_descriptions}
- Keep the useful ops from the previous plan, then add/adjust ops that target the measured weaknesses.
- Order matters: standardize_nulls and unify_numerals before parsing; drop_exact_duplicates last.
- If a weakness cannot be fixed by any allowed op, skip it — never invent operations.

MEASURED REMAINING WEAKNESSES (computed with pandas — trustworthy):
{weaknesses_json}

PREVIOUS PLAN:
{plan_json}

DATA PROFILE:
{profile_json}
"""


def optimize_plan(prev_plan: list[dict], weaknesses: list[dict], profile: dict, df: pd.DataFrame) -> list[dict]:
    """Optimizer agent: turn the Judge's weakness vector into an improved plan."""
    prompt = OPTIMIZER_TEMPLATE.format(
        allowed_ops=", ".join(REGISTRY.keys()),
        op_descriptions="; ".join(f"{k}: {v['desc']}" for k, v in REGISTRY.items()),
        weaknesses_json=json.dumps(weaknesses, ensure_ascii=False, default=str),
        plan_json=json.dumps(prev_plan, ensure_ascii=False, default=str),
        profile_json=json.dumps(profile, ensure_ascii=False, default=str),
    )
    return parse_json_array(generate(prompt))


def parse_json_array(raw: str) -> list[dict]:
    """Extract the JSON array from the LLM reply, tolerant of fences/prose.

    Scans for a bracket-balanced [...] (ignoring brackets inside strings)
    rather than naive first-[/last-], so prose containing '[' doesn't break it.
    """
    text = raw.strip()
    if text.startswith("```"):  # strip markdown fences
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    depth = start = 0
    in_str = escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                # Return the first balanced array that is valid JSON — prose
                # like "[step 1]" scans as balanced but won't parse, so skip it.
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"No JSON array in LLM reply: {raw[:200]}")


def validate_plan(plan: list[dict], df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Whitelist gate: reject unknown ops, missing columns, and unknown params keys."""
    valid, rejected = [], []
    for item in plan:
        if not isinstance(item, dict):
            rejected.append("عنصر خطة مشوّه — رُفض")
            continue
        op = item.get("op")
        if op not in REGISTRY:
            rejected.append(f"عملية غير معروفة رُفضت: {op}")
            continue
        spec = REGISTRY[op]
        if spec["needs_column"]:
            col = item.get("column")
            if col not in df.columns:
                rejected.append(f"{op}: العمود '{col}' غير موجود — رُفضت")
                continue
        params = item.get("params") or {}
        unknown = [k for k in params if k not in spec["allowed_params"]]
        if unknown:
            rejected.append(f"{op}: معاملات غير مسموحة {unknown} — رُفضت")
            continue
        if op == "map_values":
            mapping = params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                rejected.append("map_values بدون mapping صريح — رُفضت")
                continue
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                rejected.append("map_values: الـ mapping يجب أن يكون نص→نص فقط — رُفضت")
                continue
        valid.append(item)
    return valid, rejected
