"""Core safety tests — every test here pins a real bug found in review.

Run: pytest tests/ -v
"""

import pandas as pd
import pytest

from src.ops import (
    apply_plan,
    normalize_phone_sa,
    parse_dates,
    standardize_nulls,
    to_numeric,
    trim_whitespace,
    unify_numerals,
)
from src.planner import validate_plan
from src.profiler import profile_dataframe
from src.quality import quality_score


# --- 1. parse_dates must NEVER corrupt correct dates -----------------------

def test_parse_dates_preserves_iso():
    """The killer bug: dayfirst+mixed used to flip 2026-02-01 into 2026-01-02."""
    df = pd.DataFrame({"d": ["2026-02-01", "2026-02-10", "2026-03-08", "2026-12-31"]})
    out, _ = parse_dates(df, "d")
    assert list(out["d"]) == ["2026-02-01", "2026-02-10", "2026-03-08", "2026-12-31"]


def test_parse_dates_handles_mixed_formats():
    df = pd.DataFrame({"d": ["15/01/2026", "2026/02/05", "05-02-2026"]})
    out, affected = parse_dates(df, "d")
    assert list(out["d"]) == ["2026-01-15", "2026-02-05", "2026-02-05"]
    assert affected == 3


def test_parse_dates_never_turns_numbers_into_dates():
    """"1200" is an amount, not the year 1200."""
    df = pd.DataFrame({"d": ["1200", "450.5", "2026-01-15"]})
    out, _ = parse_dates(df, "d")
    assert list(out["d"])[:2] == ["1200", "450.5"]
    assert list(out["d"])[2] == "2026-01-15"


def test_parse_dates_leaves_unparseable_untouched():
    df = pd.DataFrame({"d": ["غير معروف", "2026-01-15"]})
    out, _ = parse_dates(df, "d")
    assert list(out["d"]) == ["غير معروف", "2026-01-15"]


# --- 2. to_numeric must never raise ----------------------------------------

def test_to_numeric_never_raises_on_mixed_column():
    """Used to crash on pandas 3.x when a column mixed numbers and text."""
    df = pd.DataFrame({"amount": ["450.50", "-", "N/A", "١٢٠٠", "abc", None]})
    out, affected = to_numeric(df, "amount")
    assert out["amount"][0] == 450.5
    assert out["amount"][3] == 1200.0
    assert out["amount"][1] == "-"      # unconvertible left as-is
    assert out["amount"][4] == "abc"
    assert affected == 2


def test_text_ops_skip_numeric_columns():
    """Text ops must not silently destroy numeric dtypes."""
    df = pd.DataFrame({"n": [450.5, 120.0]})
    for op in (trim_whitespace, unify_numerals, standardize_nulls):
        out, affected = op(df, "n")
        assert affected == 0
        assert pd.api.types.is_numeric_dtype(out["n"])


# --- 3. profiler must survive edge inputs ----------------------------------

def test_profiler_edge_inputs():
    """Used to crash (int('') ValueError) on all-null columns."""
    all_null = pd.DataFrame({"a": [None, None], "b": ["x", "y"]})
    profile_dataframe(all_null)

    headers_only = pd.DataFrame({"a": [], "b": []})
    profile_dataframe(headers_only)

    empty = pd.DataFrame()
    p = profile_dataframe(empty)
    assert p["rows"] == 0


# --- 4. validator must reject malformed plans ------------------------------

def test_validator_rejects_malformed_plans():
    df = pd.DataFrame({"mobile": ["0501234567"]})
    plan = [
        {"op": "hack_the_db", "column": "mobile"},                          # unknown op
        {"op": "trim_whitespace", "column": "nope"},                        # missing column
        {"op": "trim_whitespace", "column": "mobile", "params": {"x": 1}},  # unknown param
        {"op": "map_values", "column": "mobile", "params": {}},             # no mapping
        {"op": "map_values", "column": "mobile",
         "params": {"mapping": {"a": {"nested": 1}}}},                      # non-string mapping
        {"op": "normalize_phone_sa", "column": "mobile", "params": {}},     # valid ✓
    ]
    valid, rejected = validate_plan(plan, df)
    assert len(valid) == 1 and valid[0]["op"] == "normalize_phone_sa"
    assert len(rejected) == 5


# --- 5. golden end-to-end on the demo sample -------------------------------

def test_end_to_end_sample_file():
    df = pd.read_csv("data/samples/messy_customers_ar.csv")
    plan = [
        {"op": "standardize_nulls", "column": "amount_sar", "params": {}},
        {"op": "standardize_nulls", "column": "order_date", "params": {}},
        {"op": "unify_numerals", "column": "mobile", "params": {}},
        {"op": "normalize_arabic_text", "column": "name", "params": {}},
        {"op": "normalize_phone_sa", "column": "mobile", "params": {}},
        {"op": "parse_dates", "column": "order_date", "params": {}},
        {"op": "to_numeric", "column": "amount_sar", "params": {}},
    ]
    valid, rejected = validate_plan(plan, df)
    assert not rejected
    clean, log = apply_plan(df, valid)

    # ISO dates stayed ISO (the corruption bug is dead)
    assert clean["order_date"][0] == "2026-01-15"
    assert clean["order_date"][4] == "2026-02-01"   # was flipped to 2026-01-02 by the old bug
    # Hindi-numeral phone normalized to +966
    assert clean["mobile"][2] == "+966559876543"
    # 11-digit invalid phone left untouched (never guess)
    assert str(clean["mobile"][8]) == "05044455566"
    # alef unified
    assert clean["name"][0] == clean["name"][1] == "احمد العتيبي"

    # THE LAW: cleaning must RAISE the quality score (validity dimension live)
    score_before, dims_before = quality_score(df)
    score_after, dims_after = quality_score(clean)
    assert score_after > score_before
    assert dims_after["validity"] > dims_before["validity"]
    assert dims_after["consistency"] >= dims_before["consistency"]


def test_kind_detection():
    from src.quality import _detect_kind

    df = pd.read_csv("data/samples/messy_customers_ar.csv")
    assert _detect_kind("mobile", df["mobile"]) == "phone"
    assert _detect_kind("order_date", df["order_date"]) == "date"
    assert _detect_kind("amount_sar", df["amount_sar"]) == "numeric"
    assert _detect_kind("name", df["name"]) == "text"


def test_phone_float_artifact():
    """501234567.0 (float column with NaN) must still normalize."""
    df = pd.DataFrame({"mobile": [501234567.0, None]})
    out, affected = normalize_phone_sa(df, "mobile")
    assert out["mobile"][0] == "+966501234567"
    assert affected == 1
