"""Rule Planner agent — the LLM proposes, the validator disposes.

Sends the LLM only: the computed profile (aggregate stats) + a few sample
rows. Receives back a cleaning plan as strict JSON. Every plan item is
validated against the closed op registry (op name, target column AND
params keys) before anything can run.

If the LLM is unavailable (free-tier quota, network), build_heuristic_plan
produces a deterministic rule-based plan from the profile so the demo
never dies.
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
        sample_json=df.head(n_samples).to_json(orient="records", force_ascii=False),
    )
    raw = generate(prompt)
    return validate_plan(parse_json_array(raw), df)


def build_heuristic_plan(profile: dict, df: pd.DataFrame) -> list[dict]:
    """Deterministic fallback plan (no LLM) — and a STRONG one:
    column kinds (phone/date/numeric/text) are detected deterministically
    by the Quality Judge, so the fallback can safely propose the semantic
    fixes too. The demo never depends on the AI being up."""
    from .quality import _detect_kind

    plan = []
    for col in profile["columns"]:
        name = col["column"]
        series = df[name]
        kind = _detect_kind(name, series)
        is_texty = pd.api.types.is_string_dtype(series) or series.dtype == object

        if col["hidden_nulls"]:
            plan.append({"op": "standardize_nulls", "column": name, "params": {},
                         "reason": "قيم فارغة مخفية (N/A، -، غير معروف) تتحول لفراغات حقيقية"})
        if col["hindi_numerals"]:
            plan.append({"op": "unify_numerals", "column": name, "params": {},
                         "reason": "توحيد الأرقام الهندية (٠-٩) مع الأرقام العربية (0-9)"})

        if kind == "phone":
            plan.append({"op": "normalize_phone_sa", "column": name, "params": {},
                         "reason": "توحيد أرقام الجوال السعودية إلى الصيغة القياسية +966"})
        elif kind == "date" and is_texty:
            plan.append({"op": "parse_dates", "column": name, "params": {},
                         "reason": "توحيد صيغ التواريخ المختلطة إلى ISO"})
        elif kind == "numeric" and is_texty:
            plan.append({"op": "to_numeric", "column": name, "params": {},
                         "reason": "تحويل القيم النصية الرقمية إلى أرقام حقيقية"})
        elif kind == "text" and is_texty:
            if col["alef_variants"]:
                plan.append({"op": "normalize_arabic_text", "column": name, "params": {},
                             "reason": "توحيد أشكال الألف والهمزة لمنع التطابقات الخاطئة"})
            plan.append({"op": "trim_whitespace", "column": name, "params": {},
                         "reason": "إزالة المسافات الزائدة"})

    if profile["duplicate_rows"]:
        plan.append({"op": "drop_exact_duplicates", "column": None, "params": {},
                     "reason": "إزالة الصفوف المكررة تماماً"})
    return plan


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
    """Extract the first JSON array from the LLM reply (tolerates fences/prose)."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in LLM reply: {raw[:200]}")
    return json.loads(raw[start : end + 1])


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
