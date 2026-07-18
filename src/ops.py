"""Closed operation registry — the ONLY code allowed to transform data.

Every operation is deterministic pandas, written and tested ahead of time.
The LLM can only pick from this whitelist (by name + params); anything
outside it is rejected by the plan validator. Ops never destroy data:
values that can't be safely converted are left unchanged, and text ops
never touch numeric columns.

Each op: fn(df, column=None, **params) -> (new_df, affected_count)
"""

from __future__ import annotations

import re

import pandas as pd

# --- Arabic helpers -------------------------------------------------------

HINDI_TO_ARABIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ALEF_RE = re.compile(r"[أإآٱ]")
DIACRITICS_RE = re.compile(r"[ً-ٰٟ]")
TATWEEL = "ـ"
NULL_TOKENS = {"", "-", "—", "N/A", "NA", "n/a", "null", "NULL", "None", "غير معروف", "لا يوجد", "غير متوفر"}


def _as_str(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _is_texty(series: pd.Series) -> bool:
    """Text ops must never silently destroy numeric/datetime dtypes."""
    return pd.api.types.is_string_dtype(series) or series.dtype == object


# --- Operations -----------------------------------------------------------

def trim_whitespace(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column])
    new = s.str.strip().str.replace(r"\s+", " ", regex=True)
    affected = int((s != new).fillna(False).sum())
    df = df.copy()
    df[column] = new.where(s.notna(), df[column])
    return df, affected


def normalize_arabic_text(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Unify alef variants, strip diacritics and tatweel. Meaning-preserving only."""
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column])
    new = (
        s.str.replace(ALEF_RE, "ا", regex=True)
        .str.replace(DIACRITICS_RE, "", regex=True)
        .str.replace(TATWEEL, "")
    )
    affected = int((s != new).fillna(False).sum())
    df = df.copy()
    df[column] = new.where(s.notna(), df[column])
    return df, affected


def unify_numerals(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Convert Hindi/Persian digits (٠-٩ / ۰-۹) to Arabic digits (0-9)."""
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column])
    new = s.str.translate(HINDI_TO_ARABIC)
    affected = int((s != new).fillna(False).sum())
    df = df.copy()
    df[column] = new.where(s.notna(), df[column])
    return df, affected


def standardize_nulls(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Turn hidden-null tokens ('N/A', '-', 'غير معروف', …) into real NA."""
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column])
    mask = s.str.strip().isin(NULL_TOKENS).fillna(False)
    df = df.copy()
    df.loc[mask, column] = pd.NA
    return df, int(mask.sum())


def normalize_phone_sa(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Normalize Saudi mobile numbers to +9665XXXXXXXX. Unconvertible values are left as-is."""

    def fix(v):
        if pd.isna(v):
            return v
        raw = str(v)
        if isinstance(v, float) and v.is_integer():
            raw = str(int(v))  # 501234567.0 → "501234567"
        digits = re.sub(r"\D", "", raw.translate(HINDI_TO_ARABIC))
        if digits.startswith("9665") and len(digits) == 12:
            return "+" + digits
        if digits.startswith("05") and len(digits) == 10:
            return "+966" + digits[1:]
        if digits.startswith("5") and len(digits) == 9:
            return "+966" + digits
        return v  # leave unchanged — never guess

    s = df[column]
    new = s.map(fix)
    affected = int((s.astype("string") != new.astype("string")).fillna(False).sum())
    df = df.copy()
    df[column] = new
    return df, affected


# Explicit formats tried in order BEFORE any ambiguous parsing —
# ISO first so 2026-02-01 can never be flipped into 2026-01-02.
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_NUMERIC_ONLY = re.compile(r"-?\d+(\.\d+)?")


def parse_dates(df: pd.DataFrame, column: str, dayfirst: bool = True) -> tuple[pd.DataFrame, int]:
    """Parse mixed-format dates to ISO YYYY-MM-DD.

    Safety rules: ISO/explicit formats are parsed first (never reinterpreted),
    pure numbers are never treated as dates, and unparseable values are left as-is.
    """
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column]).str.translate(HINDI_TO_ARABIC).str.strip()

    numeric_like = s.str.fullmatch(_NUMERIC_ONLY.pattern).fillna(False)
    candidates = s.where(~numeric_like)

    parsed = pd.Series(pd.NaT, index=s.index)
    for fmt in _DATE_FORMATS:
        remaining = parsed.isna() & candidates.notna()
        if not remaining.any():
            break
        attempt = pd.to_datetime(candidates.where(remaining), format=fmt, errors="coerce")
        parsed = parsed.fillna(attempt)

    # Free-form leftovers (e.g. "Jan 20 2026") — explicit formats already consumed ISO,
    # so dayfirst here can no longer flip unambiguous dates.
    remaining = parsed.isna() & candidates.notna()
    if remaining.any():
        attempt = pd.to_datetime(candidates.where(remaining), errors="coerce",
                                 dayfirst=dayfirst, format="mixed")
        parsed = parsed.fillna(attempt)

    iso = parsed.dt.strftime("%Y-%m-%d")
    new = iso.where(parsed.notna(), df[column])
    affected = int((parsed.notna() & (s != iso)).fillna(False).sum())
    df = df.copy()
    df[column] = new
    return df, affected


def to_numeric(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Convert numeric-looking strings (incl. Hindi digits, currency symbols) to numbers.
    Values that don't parse are left as-is; the result column is object (mixed) by design."""
    if pd.api.types.is_numeric_dtype(df[column]):
        return df, 0
    if not _is_texty(df[column]):
        return df, 0
    s = _as_str(df[column]).str.translate(HINDI_TO_ARABIC)
    cleaned = s.str.replace(r"[^\d.\-]", "", regex=True)
    cleaned = cleaned.where(cleaned.str.fullmatch(_NUMERIC_ONLY.pattern).fillna(False))
    nums = pd.to_numeric(cleaned, errors="coerce")

    mask = nums.notna()
    affected = int(mask.sum())
    df = df.copy()
    if bool(mask.all()) and affected:
        df[column] = nums.astype("float64")  # all converted → clean numeric column
    else:
        new = df[column].astype(object).copy()
        new[mask] = nums[mask].astype(float)
        df[column] = new
    return df, affected


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def map_values(df: pd.DataFrame, column: str, mapping: dict) -> tuple[pd.DataFrame, int]:
    """Replace values by an explicit mapping (e.g. unify city-name variants)."""
    s = df[column]
    new = s.replace(mapping)
    affected = int((s.astype("string") != new.astype("string")).fillna(False).sum())
    df = df.copy()
    df[column] = new
    return df, affected


# --- Registry (the whitelist) ---------------------------------------------
# allowed_params: the validator rejects any plan item carrying params
# outside this list (LLM output is never trusted blindly).

REGISTRY = {
    "trim_whitespace": {"fn": trim_whitespace, "needs_column": True, "allowed_params": [],
                        "desc": "Trim/collapse whitespace"},
    "normalize_arabic_text": {"fn": normalize_arabic_text, "needs_column": True, "allowed_params": [],
                              "desc": "Unify alef variants, strip diacritics/tatweel"},
    "unify_numerals": {"fn": unify_numerals, "needs_column": True, "allowed_params": [],
                       "desc": "Hindi/Persian digits → Arabic digits"},
    "standardize_nulls": {"fn": standardize_nulls, "needs_column": True, "allowed_params": [],
                          "desc": "Hidden null tokens → real NA"},
    "normalize_phone_sa": {"fn": normalize_phone_sa, "needs_column": True, "allowed_params": [],
                           "desc": "Saudi mobiles → +9665XXXXXXXX"},
    "parse_dates": {"fn": parse_dates, "needs_column": True, "allowed_params": ["dayfirst"],
                    "desc": "Mixed date formats → ISO YYYY-MM-DD (ISO parsed first, numbers never become dates)"},
    "to_numeric": {"fn": to_numeric, "needs_column": True, "allowed_params": [],
                   "desc": "Numeric-looking strings → numbers"},
    "drop_exact_duplicates": {"fn": drop_exact_duplicates, "needs_column": False, "allowed_params": [],
                              "desc": "Remove exact duplicate rows"},
    "map_values": {"fn": map_values, "needs_column": True, "allowed_params": ["mapping"],
                   "desc": "Explicit value mapping (e.g. city variants)"},
}


def _kwargs_for(item: dict, spec: dict) -> dict:
    kwargs = {k: v for k, v in (item.get("params") or {}).items() if k in spec["allowed_params"]}
    if spec["needs_column"]:
        kwargs["column"] = item["column"]
    return kwargs


def apply_plan(df: pd.DataFrame, plan: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply approved plan items in order. Returns (clean_df, change_log)."""
    log = []
    for item in plan:
        spec = REGISTRY[item["op"]]
        df, affected = spec["fn"](df, **_kwargs_for(item, spec))
        log.append({
            "op": item["op"],
            "column": item.get("column") or "—",
            "affected": affected,
            "reason": item.get("reason", ""),
        })
    return df, log


def dry_run(df: pd.DataFrame, plan: list[dict], max_examples: int = 3) -> list[dict]:
    """Simulate the full plan on a copy and capture, per operation:
    the affected count and real before→after examples — so the human
    approves each op knowing exactly what it will do. Nothing is written."""
    previews = []
    work = df
    for item in plan:
        spec = REGISTRY[item["op"]]
        before = work
        work, affected = spec["fn"](work, **_kwargs_for(item, spec))

        examples: list[dict] = []
        note = ""
        if spec["needs_column"]:
            col = item["column"]
            b_str = before[col].astype("string").fillna("␀")
            a_str = work[col].astype("string").fillna("␀")
            changed = b_str != a_str
            for idx in list(changed[changed].index[:max_examples]):
                b_val = before[col][idx]
                a_val = work[col][idx]
                examples.append({
                    "before": "∅" if pd.isna(b_val) else str(b_val),
                    "after": "∅ (empty)" if pd.isna(a_val) else str(a_val),
                })
        elif affected:
            note = f"{affected} duplicate row(s) will be removed"

        previews.append({
            "op": item["op"],
            "column": item.get("column") or "—",
            "affected": affected,
            "examples": examples,
            "note": note,
        })
    return previews
