"""Loop-mechanics tests — no LLM involved (planner/optimizer are injected)."""

import pandas as pd
import pytest

from src.loop import run_loop
from src.profiler import profile_dataframe

GOLDEN_PLAN = [
    {"op": "standardize_nulls", "column": "order_date", "params": {}},
    {"op": "standardize_nulls", "column": "amount_sar", "params": {}},
    {"op": "unify_numerals", "column": "mobile", "params": {}},
    {"op": "unify_numerals", "column": "amount_sar", "params": {}},
    {"op": "normalize_arabic_text", "column": "name", "params": {}},
    {"op": "normalize_arabic_text", "column": "city", "params": {}},
    {"op": "trim_whitespace", "column": "city", "params": {}},
    {"op": "normalize_phone_sa", "column": "mobile", "params": {}},
    {"op": "parse_dates", "column": "order_date", "params": {}},
    {"op": "to_numeric", "column": "amount_sar", "params": {}},
]

PARTIAL_PLAN = GOLDEN_PLAN[:2]  # nulls only — leaves plenty of weaknesses


@pytest.fixture
def sample():
    df = pd.read_csv("data/samples/messy_customers_ar.csv")
    return df, profile_dataframe(df)


def test_loop_stops_at_threshold_without_optimizer(sample):
    """A good first plan that passes the threshold must stop after 1 iteration."""
    df, profile = sample

    def never_called(*args, **kwargs):
        raise AssertionError("optimizer must not be called when threshold is reached")

    best, history, rejected, source = run_loop(
        df, profile, threshold=90.0,
        planner_fn=lambda p, d: (GOLDEN_PLAN, []),
        optimizer_fn=never_called,
    )
    assert best is not None and best["score"] >= 90
    assert len(history) == 1
    assert source == "ai"


def test_loop_optimizer_improves_score(sample):
    """Weak first plan → optimizer returns the golden plan → score must climb."""
    df, profile = sample
    best, history, rejected, source = run_loop(
        df, profile, threshold=99.5,
        planner_fn=lambda p, d: (PARTIAL_PLAN, []),
        optimizer_fn=lambda plan, weaknesses, p, d: GOLDEN_PLAN,
    )
    assert len(history) >= 2
    assert history[1]["score"] > history[0]["score"]
    assert best["iteration"] == 2


def test_loop_keeps_best_on_stagnation(sample):
    """Optimizer that repeats the same plan → loop stops, best-so-far kept."""
    df, profile = sample
    best, history, rejected, source = run_loop(
        df, profile, threshold=100.0,
        planner_fn=lambda p, d: (GOLDEN_PLAN, []),
        optimizer_fn=lambda plan, weaknesses, p, d: GOLDEN_PLAN,  # same plan again
    )
    assert len(history) == 1
    assert round(best["score"], 1) == history[0]["score"]


def test_loop_falls_back_to_heuristic_when_ai_dies(sample):
    """Planner raising (quota out) must not kill the loop — heuristic plan used."""
    df, profile = sample

    def dead_planner(p, d):
        raise RuntimeError("quota exceeded")

    best, history, rejected, source = run_loop(
        df, profile, threshold=99.0,
        planner_fn=dead_planner,
        optimizer_fn=lambda *a, **k: GOLDEN_PLAN,
    )
    assert source == "heuristic"
    assert best is not None and len(history) == 1  # heuristic mode can't optimize further
