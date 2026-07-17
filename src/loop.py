"""The Evaluator-Optimizer loop — the heart of the system (Phase 2b).

Planner proposes → Executor applies (on a copy) → Judge measures → if the
score is below threshold, the Optimizer receives the Judge's weakness
vector and produces a targeted improved plan → repeat.

Stop conditions (any): threshold reached · diminishing returns ·
iteration cap · optimizer stagnation (same plan again).
The best-so-far plan is always kept — a bad iteration can never make the
final result worse. The loop only explores on copies; nothing touches the
user's data until they approve the winning plan in the UI.

AI-only by design: if the initial Planner call fails, the loop raises —
the app stops with a clear message instead of planning without AI.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from .ops import apply_plan
from .planner import build_plan, optimize_plan, validate_plan
from .quality import quality_score, weakness_report


def _signature(plan: list[dict]) -> str:
    return json.dumps(plan, sort_keys=True, ensure_ascii=False, default=str)


def run_loop(
    df: pd.DataFrame,
    profile: dict,
    threshold: float = 95.0,
    max_iters: int = 3,
    min_gain: float = 1.0,
    on_event: Callable[[str], None] | None = None,
    planner_fn: Callable | None = None,
    optimizer_fn: Callable | None = None,
) -> tuple[dict | None, list[dict], list[str]]:
    """Run the full loop. Returns (best, history, rejected_notes).
    Raises if the AI planner is unavailable — this system never plans without AI."""
    emit = on_event or (lambda msg: None)
    planner_fn = planner_fn or build_plan
    optimizer_fn = optimizer_fn or optimize_plan

    rejected_notes: list[str] = []
    emit("🤖 Rule Planner: proposing the initial plan...")
    plan, rejected = planner_fn(profile, df)
    rejected_notes += rejected

    best: dict | None = None
    history: list[dict] = []
    seen = {_signature(plan)}
    prev_score: float | None = None

    for iteration in range(1, max_iters + 1):
        clean, log = apply_plan(df, plan)
        score, dims = quality_score(clean)
        emit(f"⚖️ Iteration {iteration}: {len(plan)} ops → score {score:.1f}/100")
        history.append({"iteration": iteration, "ops": len(plan), "score": round(score, 1)})

        if best is None or score > best["score"]:
            best = {"plan": plan, "clean": clean, "log": log,
                    "score": score, "dims": dims, "iteration": iteration}

        if score >= threshold:
            emit(f"✅ Threshold {threshold:.0f} reached — stopping")
            break
        if prev_score is not None and score - prev_score < min_gain:
            emit("🛑 Diminishing returns — keeping the best plan so far")
            break
        prev_score = score
        if iteration == max_iters:
            emit("🛑 Iteration cap reached — keeping the best plan so far")
            break

        weaknesses = weakness_report(clean)
        if not weaknesses:
            emit("🛑 No measurable weaknesses left")
            break

        emit(f"🔧 Optimizer: targeting {len(weaknesses)} remaining weakness group(s)...")
        try:
            improved = optimizer_fn(plan, weaknesses, profile, df)
        except Exception as e:
            emit(f"⚠️ Optimizer unavailable ({e}) — keeping the best plan so far")
            break
        valid, rejected = validate_plan(improved, df)
        rejected_notes += rejected
        signature = _signature(valid)
        if not valid or signature in seen:
            emit("🛑 Optimizer stagnated (no new plan) — keeping the best plan so far")
            break
        seen.add(signature)
        plan = valid

    return best, history, rejected_notes
