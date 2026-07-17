"""Deterministic data profiler — pure pandas, no LLM.

Produces the structured profile that later feeds the Semantic Inferrer
and the Rule Planner (LLM agents). Numbers here are always computed,
never generated.
"""

from __future__ import annotations

import re

import pandas as pd

# Arabic-specific patterns
ARABIC_CHARS = re.compile(r"[؀-ۿ]")
HINDI_DIGITS = re.compile(r"[٠-٩]")
ALEF_VARIANTS = re.compile(r"[أإآٱ]")


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profile a dataframe: per-column stats + dataset-level issues."""
    columns = []
    issues = []

    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues.append(f"🔁 {duplicate_rows} duplicate rows found ({duplicate_rows / total_rows:.1%} of data)")

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        # Hidden nulls: strings that mean "missing" but aren't NaN
        hidden_nulls = 0
        mixed_numerals = 0
        alef_variants = 0
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            as_str = series.dropna().astype(str)
            hidden_nulls = int(as_str.str.strip().isin(["", "-", "N/A", "NA", "null", "NULL", "غير معروف", "لا يوجد"]).sum())
            mixed_numerals = int(as_str.apply(lambda v: bool(HINDI_DIGITS.search(v))).sum())
            alef_variants = int(as_str.apply(lambda v: bool(ALEF_VARIANTS.search(v))).sum())

        columns.append(
            {
                "column": str(col),
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": f"{missing / total_rows:.1%}" if total_rows else "0%",
                "hidden_nulls": hidden_nulls,
                "unique_values": int(series.nunique(dropna=True)),
                "hindi_numerals": mixed_numerals,
                "alef_variants": alef_variants,
            }
        )

        if total_rows and missing / total_rows > 0.2:
            issues.append(f"🕳️ Column '{col}': {missing / total_rows:.0%} missing values")
        if hidden_nulls:
            issues.append(f"👻 Column '{col}': {hidden_nulls} hidden nulls (e.g. 'N/A', '-', 'غير معروف')")
        if mixed_numerals:
            issues.append(f"🔢 Column '{col}': {mixed_numerals} values contain Hindi numerals (٠-٩) — mixed with Arabic numerals (0-9)")
        if alef_variants:
            issues.append(f"✍️ Column '{col}': {alef_variants} values contain alef variants (أ/إ/آ) — may cause false mismatches")

    return {"rows": total_rows, "duplicate_rows": duplicate_rows, "columns": columns, "issues": issues}
