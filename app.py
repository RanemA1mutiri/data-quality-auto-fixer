"""Data Quality Auto-Fixer — Streamlit entry point.

Phase 1 (MVP in progress): upload → profile → quality score.
The LLM never touches the data: pandas computes, the LLM will only
propose cleaning plans and narrate (see README architecture).
"""

import streamlit as st
import pandas as pd

from src.profiler import profile_dataframe
from src.quality import quality_score

st.set_page_config(page_title="Data Quality Auto-Fixer", page_icon="🧹", layout="wide")

st.title("🧹 Data Quality Auto-Fixer")
st.caption(
    "AI multi-agent system (evaluator–optimizer) that repairs messy data — Arabic-first. "
    "🚧 Phase 1: profiling & quality score."
)

uploaded = st.file_uploader("Upload a messy CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("⬆️ Upload a file to see its data-quality profile. Try `data/samples/messy_customers_ar.csv` from the repo.")
    st.stop()

# --- Load (a copy — the original upload is never mutated) ---
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# --- Deterministic profile ---
profile = profile_dataframe(df)
score, dimensions = quality_score(df)

st.subheader("Data Quality Score")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Overall quality", f"{score:.0f} / 100")
with col2:
    st.dataframe(
        pd.DataFrame(
            [{"Dimension": name, "Score": f"{val*100:.1f}%"} for name, val in dimensions.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Column Profile")
st.dataframe(pd.DataFrame(profile["columns"]), use_container_width=True, hide_index=True)

st.subheader("Detected Issues")
if profile["issues"]:
    for issue in profile["issues"]:
        st.warning(issue)
else:
    st.success("No obvious issues detected — this file is unusually clean!")

st.divider()
st.caption("Next up (Phase 1): LLM-proposed cleaning plan → human approval → apply with pandas → download clean file.")
