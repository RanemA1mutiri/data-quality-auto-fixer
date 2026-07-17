"""Closed operation registry — the ONLY code allowed to transform data.

Every operation is deterministic pandas, written and tested ahead of time.
The LLM can only pick from this whitelist (by name + params); anything
outside it is rejected by the plan validator. Ops never destroy data:
values that can't be safely converted are left unchanged.

Each op: fn(df, column=None, **params) -> (new_df, affected_count)
"""

from __future__ import annotations

import re

import pandas as pd

# --- Arabic helpers -------------------------------------------------------

HINDI_TO_ARABIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ALEF_RE = re.compile(r"[أإآٱ]")
DIACRITICS_RE = re.compile(r"[ً-ٰٟ]")
TATWEEL = "ـ"
NULL_TOKENS = {"", "-", "—", "N/A", "NA", "n/a", "null", "NULL", "None", "غير معروف", "لا يوجد", "غير متوفر"}


def _as_str(series: pd.Series) -> pd.Series:
    return series.astype("string")


# --- Operations -----------------------------------------------------------

def trim_whitespace(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    s = _as_str(df[column])
    new = s.str.strip().str.replace(r"\s+", " ", regex=True)
    affected = int((s != new).fillna(False).sum())
    df = df.copy()
    df[column] = new.where(s.notna(), df[column])
    return df, affected


def normalize_arabic_text(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Unify alef variants, strip diacritics and tatweel. Meaning-preserving only."""
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
    s = _as_str(df[column])
    new = s.str.translate(HINDI_TO_ARABIC)
    affected = int((s != new).fillna(False).sum())
    df = df.copy()
    df[column] = new.where(s.notna(), df[column])
    return df, affected


def standardize_nulls(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Turn hidden-null tokens ('N/A', '-', 'غير معروف', …) into real NA."""
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
        digits = re.sub(r"\D", "", str(v).translate(HINDI_TO_ARABIC))
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


def parse_dates(df: pd.DataFrame, column: str, dayfirst: bool = True) -> tuple[pd.DataFrame, int]:
    """Parse mixed-format dates to ISO YYYY-MM-DD. Unparseable values are left as-is."""
    s = _as_str(df[column]).str.translate(HINDI_TO_ARABIC)
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    iso = parsed.dt.strftime("%Y-%m-%d")
    new = iso.where(parsed.notna(), df[column])
    affected = int((parsed.notna() & (s != iso)).sum())
    df = df.copy()
    df[column] = new
    return df, affected


def to_numeric(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Convert numeric-looking strings (incl. Hindi digits, currency symbols) to numbers.
    Values that don't parse are left as-is."""
    s = _as_str(df[column]).str.translate(HINDI_TO_ARABIC)
    cleaned = s.str.replace(r"[^\d.\-]", "", regex=True)
    nums = pd.to_numeric(cleaned, errors="coerce")
    new = nums.where(nums.notna(), df[column])
    affected = int((nums.notna() & (s.str.strip() != nums.astype("string"))).fillna(False).sum())
    df = df.copy()
    df[column] = new
    return df, affected


def drop_exact_duplicates(df: pd.DataFrame, column: str | None = None) -> tuple[pd.DataFrame, int]:
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

REGISTRY = {
    "trim_whitespace": {"fn": trim_whitespace, "needs_column": True,
                        "desc": "Trim/collapse whitespace"},
    "normalize_arabic_text": {"fn": normalize_arabic_text, "needs_column": True,
                              "desc": "Unify alef variants, strip diacritics/tatweel"},
    "unify_numerals": {"fn": unify_numerals, "needs_column": True,
                       "desc": "Hindi/Persian digits → Arabic digits"},
    "standardize_nulls": {"fn": standardize_nulls, "needs_column": True,
                          "desc": "Hidden null tokens → real NA"},
    "normalize_phone_sa": {"fn": normalize_phone_sa, "needs_column": True,
                           "desc": "Saudi mobiles → +9665XXXXXXXX"},
    "parse_dates": {"fn": parse_dates, "needs_column": True,
                    "desc": "Mixed date formats → ISO YYYY-MM-DD"},
    "to_numeric": {"fn": to_numeric, "needs_column": True,
                   "desc": "Numeric-looking strings → numbers"},
    "drop_exact_duplicates": {"fn": drop_exact_duplicates, "needs_column": False,
                              "desc": "Remove exact duplicate rows"},
    "map_values": {"fn": map_values, "needs_column": True,
                   "desc": "Explicit value mapping (e.g. city variants)"},
}


def apply_plan(df: pd.DataFrame, plan: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply approved plan items in order. Returns (clean_df, change_log)."""
    log = []
    for item in plan:
        spec = REGISTRY[item["op"]]
        kwargs = dict(item.get("params") or {})
        if spec["needs_column"]:
            kwargs["column"] = item["column"]
        df, affected = spec["fn"](df, **kwargs)
        log.append({
            "op": item["op"],
            "column": item.get("column") or "—",
            "affected": affected,
            "reason": item.get("reason", ""),
        })
    return df, log
