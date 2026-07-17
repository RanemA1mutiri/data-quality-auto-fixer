"""Quality Judge — deterministic scoring engine.

Computes the composite data-quality score over standard dimensions
(DAMA-style). Every number is computed with pandas; the LLM only
narrates weaknesses. Non-applicable dimensions are dropped and weights
renormalized.

Dimensions live in Phase 2:
- completeness: share of non-empty cells
- uniqueness:   share of non-duplicate rows
- validity:     share of values matching the STRICT target format for the
                column's detected kind (phone → +9665XXXXXXXX, date → ISO,
                numeric → real numbers). Messy-but-fixable values count as
                invalid — which is exactly why cleaning RAISES this score.
- consistency:  share of text cells free of representation noise
                (alef variants, Hindi numerals, untrimmed whitespace)

Column kinds are detected deterministically (name hints + content shape).
No LLM involvement anywhere in this file.
"""

from __future__ import annotations

import re

import pandas as pd

from .ops import HINDI_TO_ARABIC

DEFAULT_WEIGHTS = {
    "completeness": 0.25,
    "validity": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.15,
    "accuracy": 0.15,   # Phase 3+ (reference checks)
    "timeliness": 0.05,  # Phase 3+ (needs an SLA)
}

_PHONE_HINTS = ("mobile", "phone", "جوال", "هاتف", "تليفون")
_DATE_HINTS = ("date", "تاريخ", "يوم")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_E164_SA = re.compile(r"\+9665\d{8}")
_NUMERIC_ONLY = re.compile(r"-?\d+(\.\d+)?")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_NOISE = re.compile(r"[أإآٱ]|[٠-٩۰-۹]|^\s|\s$")


def _detect_kind(name: str, series: pd.Series) -> str:
    """Deterministic column-kind detection: 'phone' | 'date' | 'numeric' | 'text'."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    lowered = str(name).lower()
    values = series.dropna().astype("string").str.translate(HINDI_TO_ARABIC)
    if not len(values):
        return "text"

    if any(h in lowered for h in _PHONE_HINTS):
        return "phone"
    if any(h in lowered for h in _DATE_HINTS):
        return "date"

    digits = values.str.replace(r"\D", "", regex=True)
    phone_like = (digits.str.len().between(9, 12) & (
        digits.str.startswith("05") | digits.str.startswith("9665") | digits.str.startswith("5")
    ))
    if phone_like.mean() >= 0.6:
        return "phone"

    parsed = pd.Series(False, index=values.index)
    for fmt in _DATE_FORMATS:
        parsed = parsed | pd.to_datetime(values, format=fmt, errors="coerce").notna()
    if parsed.mean() >= 0.5:
        return "date"

    cleaned = values.str.replace(r"[^\d.\-]", "", regex=True)
    if (cleaned.str.fullmatch(_NUMERIC_ONLY.pattern).fillna(False) & (cleaned.str.len() > 0)).mean() >= 0.7:
        return "numeric"
    return "text"


def _validity_of(series: pd.Series, kind: str) -> tuple[int, int]:
    """Return (valid_count, checked_count) for one column. NA cells are
    completeness's business, not validity's — they're excluded here."""
    values = series.dropna()
    if not len(values):
        return 0, 0

    if kind == "numeric":
        if pd.api.types.is_numeric_dtype(series):
            return len(values), len(values)
        valid = values.map(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
        return int(valid.sum()), len(values)

    as_str = values.astype("string")
    if kind == "phone":
        return int(as_str.str.fullmatch(_E164_SA.pattern).fillna(False).sum()), len(values)
    if kind == "date":
        if pd.api.types.is_datetime64_any_dtype(series):
            return len(values), len(values)
        return int(as_str.str.fullmatch(_ISO_DATE.pattern).fillna(False).sum()), len(values)
    return len(values), len(values)  # plain text: validity not applicable → all pass


def _consistency_of(series: pd.Series) -> tuple[int, int]:
    """Representation noise in text cells: alef variants, Hindi digits,
    untrimmed whitespace. Returns (clean_count, checked_count)."""
    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return 0, 0
    values = series.dropna().astype("string")
    if not len(values):
        return 0, 0
    noisy = values.str.contains(_NOISE, regex=True).fillna(False)
    return int((~noisy).sum()), len(values)


def quality_score(df: pd.DataFrame, weights: dict | None = None) -> tuple[float, dict]:
    """Return (score_0_100, per-dimension scores in [0, 1])."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    dims: dict[str, float] = {}

    total_cells = df.shape[0] * df.shape[1]
    if total_cells:
        dims["completeness"] = 1.0 - (int(df.isna().sum().sum()) / total_cells)

    if len(df):
        dims["uniqueness"] = 1.0 - (int(df.duplicated().sum()) / len(df))

    valid_total = checked_total = 0
    clean_total = text_total = 0
    for col in df.columns:
        kind = _detect_kind(col, df[col])
        v, c = _validity_of(df[col], kind)
        valid_total += v
        checked_total += c
        cl, ct = _consistency_of(df[col])
        clean_total += cl
        text_total += ct

    if checked_total:
        dims["validity"] = valid_total / checked_total
    if text_total:
        dims["consistency"] = clean_total / text_total

    # Drop non-applicable dimensions and renormalize weights
    applicable = {k: v for k, v in weights.items() if k in dims}
    total_w = sum(applicable.values())
    if not total_w:
        return 0.0, dims

    score = sum(dims[k] * (w / total_w) for k, w in applicable.items())
    return max(0.0, min(1.0, score)) * 100, dims
