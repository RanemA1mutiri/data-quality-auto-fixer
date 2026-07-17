"""Quality Judge — deterministic scoring engine.

Computes the composite data-quality score over standard dimensions
(DAMA-style). Every number is computed with pandas; the LLM only
narrates weaknesses (in a later phase). Non-applicable dimensions are
dropped and weights renormalized.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS = {
    "completeness": 0.25,
    "validity": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.15,
    "accuracy": 0.15,
    "timeliness": 0.05,
}


def quality_score(df: pd.DataFrame, weights: dict | None = None) -> tuple[float, dict]:
    """Return (score_0_100, per-dimension scores in [0, 1]).

    Phase 1 implements completeness + uniqueness (fully computable without
    column semantics). Validity/consistency/accuracy/timeliness activate in
    Phase 2 once the Semantic Inferrer provides per-column rules.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    dims: dict[str, float] = {}

    total_cells = df.shape[0] * df.shape[1]
    if total_cells:
        dims["completeness"] = 1.0 - (int(df.isna().sum().sum()) / total_cells)

    if len(df):
        dims["uniqueness"] = 1.0 - (int(df.duplicated().sum()) / len(df))

    # Drop non-applicable dimensions and renormalize weights
    applicable = {k: v for k, v in weights.items() if k in dims}
    total_w = sum(applicable.values())
    if not total_w:
        return 0.0, dims

    score = sum(dims[k] * (w / total_w) for k, w in applicable.items())
    return max(0.0, min(1.0, score)) * 100, dims
