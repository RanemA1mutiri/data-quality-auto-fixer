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
from src.planner import build_plan
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

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Tajawal:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', 'Tajawal', sans-serif; }

.stApp {
  background:
    radial-gradient(60rem 30rem at 12% -5%, rgba(168,85,247,.16), transparent 60%),
    radial-gradient(50rem 28rem at 108% 8%, rgba(6,182,212,.13), transparent 55%),
    #0b0b14;
}

/* Hero */
.dq-hero { padding: .4rem 0 1.1rem; }
.dq-hero h1 { font-size: 2.7rem; font-weight: 700; margin: 0 0 .4rem; letter-spacing: -.02em; }
.dq-grad {
  background: linear-gradient(92deg, #c084fc 0%, #818cf8 45%, #22d3ee 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.dq-tagline { color: #d7d7ea; font-size: 1.1rem; margin: 0 0 1rem; }
.dq-chips { display: flex; gap: .5rem; flex-wrap: wrap; }
.dq-chip {
  font-size: .82rem; color: #ddddf2; padding: .32rem .8rem; border-radius: 999px;
  border: 1px solid rgba(168,85,247,.4); background: rgba(168,85,247,.09);
}
.dq-steps { color: #9a9ac0; font-size: .95rem; margin-top: 1rem; }

/* Metric cards → glass */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 16px; padding: 1rem 1.15rem;
}
[data-testid="stMetricValue"] { font-weight: 700; }

/* Buttons */
.stButton > button { border-radius: 12px; }
.stButton > button[kind="primary"] {
  background: linear-gradient(92deg, #7c3aed, #06b6d4);
  border: 0; font-weight: 700;
  box-shadow: 0 4px 24px rgba(124,58,237,.35);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 32px rgba(124,58,237,.55);
}

/* Progress bars → gradient */
[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, #7c3aed, #22d3ee);
}

/* Expanders */
[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,.09); border-radius: 14px; }

/* Score gauges */
.dq-gauge-row { display: flex; gap: 1.8rem; align-items: center; flex-wrap: wrap; margin: .4rem 0 1rem; }
.dq-gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: .55rem; }
.dq-gauge {
  width: 152px; height: 152px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 44px rgba(124,58,237,.28);
}
.dq-gauge-inner {
  width: 116px; height: 116px; border-radius: 50%; background: #0e0e1a;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.dq-gauge-num { font-size: 2.35rem; font-weight: 700; line-height: 1.05; color: #ffffff; }
.dq-gauge-sub { color: #9a9ac0; font-size: .85rem; }
.dq-gauge-label { color: #d7d7ea; font-size: .95rem; }
.dq-arrow { font-size: 2.1rem; color: #22d3ee; }
</style>
"""

HERO_HTML = """
<div class="dq-hero">
  <h1><span class="dq-grad">🧹 Data Quality Auto-Fixer</span></h1>
  <p class="dq-tagline"><b>Turn messy data into a clean file in one minute — you approve every change.</b></p>
  <div class="dq-chips">
    <span class="dq-chip">🇸🇦 Arabic-first</span>
    <span class="dq-chip">🤖 Multi-agent · Evaluator–Optimizer</span>
    <span class="dq-chip">🧍 Human-in-the-loop</span>
    <span class="dq-chip">🔒 The LLM never touches your data</span>
  </div>
  <div class="dq-steps">📤 Upload → 🤖 Review the AI plan → ✅ Approve → ⬇️ Download</div>
</div>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)


def gauge_html(score: float, label: str) -> str:
    """Circular quality gauge — pure CSS conic-gradient, no external libs."""
    pct = max(0.0, min(100.0, score))
    color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
    return (
        f'<div class="dq-gauge-wrap">'
        f'<div class="dq-gauge" style="background: conic-gradient({color} {pct * 3.6}deg, #1d1d30 0deg);">'
        f'<div class="dq-gauge-inner"><div class="dq-gauge-num">{pct:.0f}</div>'
        f'<div class="dq-gauge-sub">/100</div></div></div>'
        f'<div class="dq-gauge-label">{label}</div></div>'
    )


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
    for key in ("plan", "rejected", "result", "loop_history", "previews", "preview_sig"):
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
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown(gauge_html(score_before, "Quality score"), unsafe_allow_html=True)
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
            st.session_state.pop("loop_history", None)
            st.session_state.pop("result", None)
        except Exception as e:
            st.error(str(e))

if run_auto:
    try:
        with st.status("🔁 Evaluator–optimizer loop running...", expanded=True) as status:
            best, history, rejected = run_loop(
                df, profile, threshold=float(threshold), on_event=status.write,
            )
            status.update(label="🔁 Loop finished", state="complete")
        st.session_state["plan"] = best["plan"] if best else []
        st.session_state["rejected"] = rejected
        st.session_state["loop_history"] = history
        st.session_state.pop("result", None)
    except Exception as e:
        st.error(str(e))

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

    st.markdown(
        '<div class="dq-gauge-row">'
        + gauge_html(score_before, "Before")
        + '<div class="dq-arrow">➜</div>'
        + gauge_html(score_after, "After")
        + "</div>",
        unsafe_allow_html=True,
    )
    b, c = st.columns(2)
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
