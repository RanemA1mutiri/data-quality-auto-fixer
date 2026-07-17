"""Data Quality Auto-Fixer — Streamlit entry point.

Phase 1 (MVP): upload → profile → LLM cleaning plan → human approval →
deterministic apply (pandas) → before/after score → download.

Core principle: the LLM never touches the data — it proposes a plan from
a closed op registry; pandas executes; scores are always computed.
"""

import pandas as pd
import streamlit as st

from src.ops import apply_plan
from src.planner import build_plan
from src.profiler import profile_dataframe
from src.quality import quality_score

st.set_page_config(page_title="Data Quality Auto-Fixer", page_icon="🧹", layout="wide")

st.title("🧹 Data Quality Auto-Fixer")
st.caption(
    "AI multi-agent system (evaluator–optimizer) that repairs messy data — Arabic-first. "
    "The LLM proposes; deterministic pandas executes; you approve."
)

uploaded = st.file_uploader("Upload a messy CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("⬆️ Upload a file to begin. Try `data/samples/messy_customers_ar.csv` from the repo.")
    st.stop()

# --- Load (a copy — the original upload is never mutated) ---
df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

score_before, dims_before = quality_score(df)
profile = profile_dataframe(df)

st.subheader("1 · Profile")
c1, c2, c3 = st.columns(3)
c1.metric("Quality score (before)", f"{score_before:.0f} / 100")
c2.metric("Rows", len(df))
c3.metric("Issues detected", len(profile["issues"]))

with st.expander("Preview & detected issues", expanded=True):
    st.dataframe(df.head(15), use_container_width=True)
    for issue in profile["issues"]:
        st.warning(issue)

# --- Plan (LLM proposes — privacy: only aggregate profile + 5 sample rows are sent) ---
st.subheader("2 · Cleaning plan (AI-proposed, you approve)")
st.caption("🔒 Privacy: the AI sees only aggregate statistics and 5 sample rows — never your full dataset.")

if st.button("🤖 Generate cleaning plan", type="primary"):
    with st.spinner("Rule Planner agent is analyzing the profile..."):
        try:
            plan, rejected = build_plan(profile, df)
            st.session_state["plan"] = plan
            st.session_state["rejected"] = rejected
        except Exception as e:
            st.error(f"LLM call failed: {e}")

plan = st.session_state.get("plan")
if plan:
    for note in st.session_state.get("rejected", []):
        st.error(f"🛡️ Validator: {note}")

    st.write("**Review each proposed operation** — uncheck anything you don't approve:")
    approved = []
    for i, item in enumerate(plan):
        label = f"`{item['op']}` on **{item.get('column') or 'whole table'}** — {item.get('reason', '')}"
        if st.checkbox(label, value=True, key=f"op_{i}"):
            approved.append(item)

    # --- Apply (deterministic pandas only) ---
    st.subheader("3 · Apply & download")
    if st.button(f"✅ Apply {len(approved)} approved operations"):
        clean, log = apply_plan(df, approved)
        score_after, dims_after = quality_score(clean)
        issues_after = profile_dataframe(clean)["issues"]

        a, b, c = st.columns(3)
        a.metric("Quality score (after)", f"{score_after:.0f} / 100", delta=f"{score_after - score_before:.1f}")
        b.metric("Issues remaining", len(issues_after), delta=len(issues_after) - len(profile["issues"]))
        c.metric("Cells/rows affected", sum(l["affected"] for l in log))

        st.dataframe(clean.head(15), use_container_width=True)

        st.write("**Audit log** — every transformation, recorded:")
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download clean CSV",
            clean.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"clean_{uploaded.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
        )
        st.caption(
            "Note: exposing hidden nulls ('N/A', '-') can lower the completeness score — "
            "that's honesty, not regression. The validity dimension (Phase 2) reflects the true gain."
        )

st.divider()
st.caption("Roadmap: Phase 2 — full evaluator–optimizer loop · Phase 4 — Arabic executive report + audit export.")
