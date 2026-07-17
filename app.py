"""Data Quality Auto-Fixer — Streamlit entry point.

Flow: upload (or one-click sample) → profile → AI cleaning plan (with a
deterministic fallback if the LLM is unavailable) → per-op human approval →
deterministic apply (pandas) → before/after score → audit log → download.

Core principle: the LLM never touches the data — it proposes a plan from
a closed op registry; pandas executes; scores are always computed.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loop import run_loop
from src.ops import apply_plan
from src.planner import build_heuristic_plan, build_plan
from src.profiler import profile_dataframe
from src.quality import quality_score

SAMPLE_PATH = Path(__file__).parent / "data" / "samples" / "messy_customers_ar.csv"

st.set_page_config(
    page_title="Data Quality Auto-Fixer",
    page_icon="🧹",
    layout="wide",
    menu_items={"About": "https://github.com/RanemA1mutiri/data-quality-auto-fixer"},
)

st.title("🧹 Data Quality Auto-Fixer")
st.markdown("**Turn messy data into a clean file in one minute — you approve every change.**")
st.markdown("📤 Upload → 🤖 Review the AI-proposed cleaning plan → ✅ Approve → ⬇️ Download")
st.caption("Under the hood: a multi-agent evaluator–optimizer system. The LLM proposes; deterministic pandas executes.")


# --- Safe file loading -----------------------------------------------------

def load_dataframe(uploaded) -> pd.DataFrame | None:
    """Read CSV/Excel defensively: encodings, empty files, bad extensions."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            for encoding in ("utf-8-sig", "utf-8", "cp1256"):
                try:
                    uploaded.seek(0)
                    return pd.read_csv(uploaded, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            st.error("تعذّرت قراءة ترميز الملف — جرّبي حفظه بترميز UTF-8 من Excel (CSV UTF-8).")
            return None
        uploaded.seek(0)
        return pd.read_excel(uploaded)
    except pd.errors.EmptyDataError:
        st.error("الملف فارغ — ما فيه بيانات تُقرأ.")
    except Exception as e:
        st.error(f"تعذّرت قراءة الملف: {e}")
    return None


# --- Input: upload OR one-click sample -------------------------------------

col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a messy CSV or Excel file", type=["csv", "xlsx"])
with col_sample:
    st.write("")
    st.write("")
    if st.button("✨ Try with sample data (messy Arabic customers)", type="primary"):
        st.session_state["use_sample"] = True

if uploaded is not None:
    st.session_state["use_sample"] = False
    df = load_dataframe(uploaded)
    file_id = f"{uploaded.name}:{uploaded.size}"
    source_name = uploaded.name
elif st.session_state.get("use_sample"):
    df = pd.read_csv(SAMPLE_PATH)
    file_id = "sample"
    source_name = "messy_customers_ar.csv (sample)"
else:
    st.info("⬆️ Upload a file — or click the sample button to see the system in action instantly.")
    st.stop()

if df is None or df.empty:
    if df is not None:
        st.error("الملف ما فيه صفوف بيانات.")
    st.stop()

# Reset stale state when the file changes (a plan for file A must never touch file B)
if st.session_state.get("file_id") != file_id:
    for key in ("plan", "rejected", "result"):
        st.session_state.pop(key, None)
    st.session_state["file_id"] = file_id

score_before, dims_before = quality_score(df)
profile = profile_dataframe(df)

DIM_LABELS = {
    "completeness": "Completeness — non-empty cells",
    "validity": "Validity — values matching their column's target format",
    "uniqueness": "Uniqueness — non-duplicate rows",
    "consistency": "Consistency — text free of representation noise",
}


def score_badge(score: float) -> str:
    return "🟢" if score >= 80 else "🟠" if score >= 50 else "🔴"


def render_dimensions(dims: dict) -> None:
    for key, value in dims.items():
        st.progress(min(max(value, 0.0), 1.0), text=f"{DIM_LABELS.get(key, key)}: {value:.0%}")


st.subheader("1 · Profile")
c1, c2, c3 = st.columns(3)
c1.metric(f"{score_badge(score_before)} Quality score (before)", f"{score_before:.0f} / 100")
c2.metric("Rows", len(df))
c3.metric("Issues detected", len(profile["issues"]))
with st.expander("Quality dimensions (computed, never generated)"):
    render_dimensions(dims_before)

with st.expander(f"Preview & detected issues — {source_name}", expanded=True):
    st.dataframe(df.head(15), use_container_width=True)
    for issue in profile["issues"]:
        st.warning(issue)

# --- Plan (LLM proposes; heuristic fallback keeps the demo alive) ----------

st.subheader("2 · Cleaning plan (AI-proposed, you approve)")
st.caption("🔒 Privacy: the AI sees only aggregate statistics and 5 sample rows — never your full dataset.")

threshold = st.slider(
    "🎯 Target quality score (for the auto-optimize loop)", 85, 100, 95,
    help="The evaluator–optimizer loop keeps improving the plan until the measured score passes this threshold (max 3 iterations, best plan always kept).",
)
col_loop, col_single = st.columns([2, 1])
with col_loop:
    run_auto = st.button("🔁 Auto-optimize (evaluator–optimizer loop)", type="primary")
with col_single:
    run_single = st.button("🤖 Single-pass plan")

if run_single:
    with st.spinner("Rule Planner agent is analyzing the profile..."):
        try:
            plan, rejected = build_plan(profile, df)
            st.session_state["plan"] = plan
            st.session_state["rejected"] = rejected
            st.session_state["plan_source"] = "ai"
        except Exception as e:
            st.warning(f"⚠️ {e}")
            st.session_state["plan"] = build_heuristic_plan(profile)
            st.session_state["rejected"] = []
            st.session_state["plan_source"] = "heuristic"
    st.session_state.pop("loop_history", None)
    st.session_state.pop("result", None)

if run_auto:
    with st.status("🔁 Evaluator–optimizer loop running...", expanded=True) as status:
        best, history, rejected, source = run_loop(
            df, profile, threshold=float(threshold), on_event=status.write,
        )
        status.update(label="🔁 Loop finished", state="complete")
    st.session_state["plan"] = best["plan"] if best else []
    st.session_state["rejected"] = rejected
    st.session_state["plan_source"] = source
    st.session_state["loop_history"] = history
    st.session_state.pop("result", None)

history = st.session_state.get("loop_history")
if history:
    st.caption("The loop explores on copies — nothing touches your data until you approve below.")
    st.dataframe(pd.DataFrame(history), hide_index=True)
    if len(history) > 1:
        st.success(
            f"📈 Score climbed {history[0]['score']:.0f} → {max(h['score'] for h in history):.0f} "
            f"across {len(history)} iterations — showing the winning plan below."
        )

plan = st.session_state.get("plan")
if plan is not None:
    if st.session_state.get("plan_source") == "heuristic":
        st.info("🛟 AI unavailable right now — showing a rule-based plan computed from the profile instead.")
    for note in st.session_state.get("rejected", []):
        st.error(f"🛡️ Validator: {note}")
    if not plan:
        st.success("The planner found nothing that needs fixing — unusually clean file!")

    st.write("**Review each proposed operation** — uncheck anything you don't approve:")
    approved = []
    for i, item in enumerate(plan):
        label = f"`{item['op']}` on **{item.get('column') or 'whole table'}** — {item.get('reason', '')}"
        checked = st.checkbox(label, value=True, key=f"op_{file_id}_{i}")
        mapping = (item.get("params") or {}).get("mapping")
        if mapping:
            st.dataframe(
                pd.DataFrame([{"from": k, "to": v} for k, v in mapping.items()]),
                hide_index=True,
            )
        if checked:
            approved.append(item)

    # --- Apply (deterministic pandas only) ---------------------------------
    st.subheader("3 · Apply & download")
    if st.button(f"✅ Apply {len(approved)} approved operations", disabled=not approved):
        clean, log = apply_plan(df, approved)
        score_after, dims_after = quality_score(clean)
        issues_after = profile_dataframe(clean)["issues"]
        st.session_state["result"] = {
            "clean": clean,
            "log": log,
            "score_after": score_after,
            "dims_after": dims_after,
            "issues_after": issues_after,
        }

result = st.session_state.get("result")
if result is not None:
    clean, log = result["clean"], result["log"]
    score_after, issues_after = result["score_after"], result["issues_after"]

    a, b, c = st.columns(3)
    a.metric(f"{score_badge(score_after)} Quality score (after)", f"{score_after:.0f} / 100",
             delta=f"{score_after - score_before:.1f}")
    b.metric("Issues remaining", len(issues_after),
             delta=len(issues_after) - len(profile["issues"]), delta_color="inverse")
    c.metric("Cells/rows affected", sum(entry["affected"] for entry in log))
    with st.expander("Quality dimensions after cleaning", expanded=True):
        render_dimensions(result["dims_after"])

    tab_after, tab_before = st.tabs(["✨ After", "Before"])
    with tab_after:
        st.dataframe(clean.head(15), use_container_width=True)
    with tab_before:
        st.dataframe(df.head(15), use_container_width=True)

    st.write("**Audit log** — every transformation, recorded:")
    st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download clean CSV",
        clean.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"clean_{source_name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )
    if score_after < score_before:
        st.caption(
            "Note: exposing hidden nulls ('N/A', '-') can lower the completeness score — "
            "that's honesty, not regression. The validity dimension (Phase 2) reflects the true gain."
        )

st.divider()
st.caption("Roadmap: Phase 3 — per-op dry-run preview · Phase 4 — Arabic executive report + audit export.")
