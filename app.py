"""Data Quality Auto-Fixer — Streamlit entry point.

Flow: upload (or one-click sample) → profile → AI cleaning plan (with a
deterministic fallback if the LLM is unavailable) → per-op human approval →
deterministic apply (pandas) → before/after score → audit log → download.

Core principle: the LLM never touches the data — it proposes a plan from
a closed op registry; pandas executes; scores are always computed.
"""

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loop import run_loop
from src.ops import apply_plan, dry_run
from src.planner import build_heuristic_plan, build_plan
from src.profiler import profile_dataframe
from src.quality import quality_score
from src.report import build_report

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

    # Dry-run preview: simulate the plan on a copy so every checkbox is an informed decision
    plan_sig = json.dumps(plan, sort_keys=True, ensure_ascii=False, default=str)
    if st.session_state.get("preview_sig") != (file_id, plan_sig):
        with st.spinner("Simulating the plan (dry-run) to preview each operation's impact..."):
            st.session_state["previews"] = dry_run(df, plan)
            st.session_state["preview_sig"] = (file_id, plan_sig)
    previews = st.session_state["previews"]

    st.write("**Review each proposed operation** — uncheck anything you don't approve:")
    st.caption("🔎 Impact previews below assume the full plan runs in order — unchecking earlier ops may change later effects.")
    approved = []
    for i, item in enumerate(plan):
        preview = previews[i]
        label = (
            f"`{item['op']}` on **{item.get('column') or 'whole table'}** — {item.get('reason', '')} "
            f"· 🎯 **{preview['affected']}** affected"
        )
        checked = st.checkbox(label, value=True, key=f"op_{file_id}_{i}")
        if preview["examples"]:
            st.caption("　　e.g. " + " · ".join(f"`{e['before']}` → `{e['after']}`" for e in preview["examples"]))
        elif preview["note"]:
            st.caption(f"　　{preview['note']}")
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
        st.toast(f"Done! Quality improved by {score_after - score_before:+.1f} points ✨")
        if score_after >= threshold:
            st.balloons()

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

    tab_after, tab_before = st.tabs(["✨ After (changed cells highlighted)", "Before"])
    with tab_after:
        after_head = clean.head(15).reset_index(drop=True)
        before_head = df.head(15).reset_index(drop=True)
        rows = min(len(after_head), len(before_head))
        shared_cols = [c for c in after_head.columns if c in before_head.columns]
        a_str = after_head.loc[: rows - 1, shared_cols].astype("string").fillna("␀")
        b_str = before_head.loc[: rows - 1, shared_cols].astype("string").fillna("␀")
        changed_mask = a_str != b_str

        def _style_changes(frame: pd.DataFrame) -> pd.DataFrame:
            style = pd.DataFrame("", index=frame.index, columns=frame.columns)
            style.loc[changed_mask.index, changed_mask.columns] = changed_mask.map(
                lambda hit: "background-color: #d3f9d8" if hit else ""
            )
            return style

        st.dataframe(after_head.style.apply(_style_changes, axis=None), use_container_width=True)
        if len(clean) != len(df):
            st.caption(f"↳ {len(df) - len(clean)} duplicate row(s) removed — highlighting compares rows positionally.")
    with tab_before:
        st.dataframe(df.head(15), use_container_width=True)

    st.write("**Audit log** — every transformation, recorded:")
    st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

    base_name = source_name.rsplit(".", 1)[0].replace(" ", "_")
    report_html = build_report(
        source_name=source_name,
        rows=len(df),
        score_before=score_before,
        dims_before=dims_before,
        score_after=score_after,
        dims_after=result["dims_after"],
        log=log,
        issues_before=len(profile["issues"]),
        issues_after=len(issues_after),
    )
    excel_buffer = BytesIO()
    clean.to_excel(excel_buffer, index=False)

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "⬇️ Clean CSV",
        clean.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"clean_{base_name}.csv",
        mime="text/csv",
    )
    d2.download_button(
        "⬇️ Clean Excel",
        excel_buffer.getvalue(),
        file_name=f"clean_{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d3.download_button(
        "📄 التقرير التنفيذي (عربي)",
        report_html.encode("utf-8"),
        file_name=f"quality_report_{base_name}.html",
        mime="text/html",
    )
    d4.download_button(
        "🧾 Audit log (JSON)",
        json.dumps(log, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"audit_log_{base_name}.json",
        mime="application/json",
    )
    st.caption("📄 The Arabic executive report is a self-contained HTML — open it and print to PDF for management.")

    if score_after < score_before:
        st.caption(
            "Note: exposing hidden nulls ('N/A', '-') can lower the completeness score — "
            "that's honesty, not regression. The validity dimension (Phase 2) reflects the true gain."
        )

st.divider()
st.caption("Roadmap: Phase 3 — per-op dry-run preview · Phase 4 — Arabic executive report + audit export.")
