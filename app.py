"""Data Quality Auto-Fixer — Streamlit entry point.

Flow: upload (or one-click sample) → profile → AI cleaning plan → per-op
human approval → deterministic apply (pandas) → before/after score →
audit log → download.

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
from src.profiler import profile_dataframe
from src.quality import quality_score
from src.report import build_report

SAMPLE_PATH = Path(__file__).parent / "data" / "samples" / "messy_customers_ar.csv"

st.set_page_config(
    page_title="Data Quality Auto-Fixer",
    page_icon="assets/favicon.svg",
    layout="wide",
    menu_items={"About": "https://github.com/RanemA1mutiri/data-quality-auto-fixer"},
)


def icon(name: str, size: int = 18) -> str:
    """Inline monochrome stroke icon (Lucide-style). Use only inside
    st.markdown(..., unsafe_allow_html=True)."""
    paths = {
        "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
        "languages": '<path d="M4 6h9"/><path d="M8.5 4v2c0 3.5-2 6.5-5 8"/><path d="M6 10c1 2.5 3 4.5 6 5.5"/><path d="m13 20 3.5-8 3.5 8"/><path d="M14.2 17h4.6"/>',
        "bot": '<rect x="4" y="8" width="16" height="11" rx="2.5"/><path d="M12 8V4.5"/><circle cx="12" cy="3.2" r="1.2"/><path d="M9.5 13v1.5M14.5 13v1.5"/>',
        "user-check": '<circle cx="10" cy="8" r="3.3"/><path d="M4 20c0-3.3 2.7-6 6-6 1.1 0 2.2.3 3.1.8"/><path d="m15.5 17 2 2 3.5-3.5"/>',
        "lock": '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
        "upload": '<path d="M12 15V4"/><path d="m8 8 4-4 4 4"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
        "check": '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
        "download": '<path d="M12 4v11"/><path d="m8 11 4 4 4-4"/><path d="M4 19h16"/>',
        "arrow-right": '<path d="M4 12h15"/><path d="m13 6 6 6-6 6"/>',
    }
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-3px;margin-inline-end:.4rem">'
        f'{paths[name]}</svg>'
    )


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', 'IBM Plex Sans Arabic', sans-serif; }

.stApp { background: #FBFBFD; }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; max-width: 1180px; }

/* Hero */
.dq-hero { padding: .5rem 0 1.4rem; }
.dq-hero h1 { font-size: 1.9rem; font-weight: 600; margin: 0 0 .5rem; letter-spacing: -.01em; color: #1A1D24; }
.dq-hero h1 svg { color: #4F46E5; }
.dq-tagline { color: #5A6472; font-size: 1.05rem; margin: 0 0 1.1rem; }
.dq-chips { display: flex; gap: .5rem; flex-wrap: wrap; }
.dq-chip {
  display: inline-flex; align-items: center; font-size: .82rem; font-weight: 500;
  color: #475467; padding: .34rem .75rem; border-radius: 6px;
  border: 1px solid #E6E8EB; background: #F1F3F5;
}
.dq-chip svg { color: #5A6472; }
.dq-steps { display: flex; align-items: center; gap: .35rem; flex-wrap: wrap;
  color: #5A6472; font-size: .92rem; margin-top: 1.1rem; }
.dq-steps svg { color: #4F46E5; }
.dq-steps .sep { color: #98A2B3; margin: 0 .2rem; }

/* Metric cards */
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #E6E8EB;
  border-radius: 12px; padding: 1rem 1.15rem;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
}
[data-testid="stMetricValue"] { font-weight: 600; color: #1A1D24; }

/* Buttons */
.stButton > button { border-radius: 8px; font-weight: 500; }
.stButton > button[kind="primary"] {
  background: #4F46E5; border: 0; color: #fff; font-weight: 600;
  box-shadow: 0 1px 2px rgba(16,24,40,.08);
  transition: background .15s ease;
}
.stButton > button[kind="primary"]:hover { background: #4338CA; }

/* Apply button → green (the "commit the fix" action) */
.st-key-apply_btn button:not(:disabled) {
  background: #1A7F5A; color: #fff; border: 0; font-weight: 600;
  box-shadow: 0 1px 2px rgba(16,24,40,.08);
}
.st-key-apply_btn button:not(:disabled):hover { background: #15663F; }

/* Progress bars (dimension bars) */
[data-testid="stProgress"] > div > div > div { background: #4F46E5; }

/* Expanders */
[data-testid="stExpander"] { border: 1px solid #E6E8EB; border-radius: 12px; background: #FFFFFF; }

/* Score gauges */
.dq-gauge-row { display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; margin: .4rem 0 1rem; }
.dq-gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: .6rem; }
.dq-gauge {
  width: 96px; height: 96px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
}
.dq-gauge-inner {
  width: 74px; height: 74px; border-radius: 50%; background: #FFFFFF;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.dq-gauge-num { font-size: 1.55rem; font-weight: 700; line-height: 1.05; color: #1A1D24; }
.dq-gauge-sub { color: #98A2B3; font-size: .75rem; }
.dq-gauge-label { color: #5A6472; font-size: .9rem; font-weight: 500; }
.dq-arrow { color: #98A2B3; display: flex; align-items: center; }

/* Gauge as a card — matches the Rows / Issues metric cards exactly */
.dq-gauge-card {
  background: #FFFFFF; border: 1px solid #E6E8EB; border-radius: 12px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
  min-height: 150px; padding: 1rem 1.15rem;
  display: flex; flex-direction: column; align-items: flex-start; justify-content: center; gap: .6rem;
}
.dq-gauge-card .dq-gauge-label { order: -1; font-weight: 400; }
[data-testid="stMetric"] { min-height: 150px; display: flex; flex-direction: column; justify-content: center; }
</style>
"""

HERO_HTML = f"""
<div class="dq-hero">
  <h1>{icon("database", 26)}Data Quality Auto-Fixer</h1>
  <p class="dq-tagline">Turn messy data into a clean file in one minute — you approve every change.</p>
  <div class="dq-chips">
    <span class="dq-chip">{icon("languages", 15)}Arabic-first</span>
    <span class="dq-chip">{icon("bot", 15)}Multi-agent · Evaluator–Optimizer</span>
    <span class="dq-chip">{icon("user-check", 15)}Human-in-the-loop</span>
    <span class="dq-chip">{icon("lock", 15)}The LLM never touches your data</span>
  </div>
  <div class="dq-steps">
    {icon("upload", 16)}Upload <span class="sep">→</span>
    {icon("bot", 16)}Review the AI plan <span class="sep">→</span>
    {icon("check", 16)}Approve <span class="sep">→</span>
    {icon("download", 16)}Download
  </div>
</div>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)


def gauge_html(score: float, label: str, card: bool = False) -> str:
    """Circular quality gauge — pure CSS conic-gradient, no external libs.
    card=True wraps it in a bordered card matching the metric cards."""
    pct = max(0.0, min(100.0, score))
    color = "#0F8A45" if pct >= 90 else "#CC7309" if pct >= 65 else "#D13A3F"
    inner = (
        f'<div class="dq-gauge" style="background: conic-gradient({color} {pct * 3.6}deg, #EAECF0 0deg);">'
        f'<div class="dq-gauge-inner"><div class="dq-gauge-num">{pct:.0f}</div>'
        f'<div class="dq-gauge-sub">/100</div></div></div>'
        f'<div class="dq-gauge-label">{label}</div>'
    )
    wrap = "dq-gauge-card" if card else "dq-gauge-wrap"
    return f'<div class="{wrap}">{inner}</div>'


# --- Safe file loading -----------------------------------------------------

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate column names (name, name.1, name.2 ...) so every
    df[col] is a Series — duplicate headers otherwise crash all profiling."""
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df = df.copy()
    df.columns = new_cols
    return df


def load_dataframe(uploaded) -> pd.DataFrame | None:
    """Read CSV/Excel defensively: encodings, empty files, bad extensions,
    duplicate column names."""
    name = uploaded.name.lower()
    df = None
    try:
        if name.endswith(".csv"):
            for encoding in ("utf-8-sig", "utf-8", "cp1256"):
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                st.error("تعذّرت قراءة ترميز الملف — جرّبي حفظه بترميز UTF-8 من Excel (CSV UTF-8).")
                return None
        else:
            uploaded.seek(0)
            df = pd.read_excel(uploaded)
    except pd.errors.EmptyDataError:
        st.error("الملف فارغ — ما فيه بيانات تُقرأ.")
        return None
    except Exception as e:
        st.error(f"تعذّرت قراءة الملف: {e}")
        return None

    df = _dedupe_columns(df)
    if df.columns.duplicated().any():  # belt-and-suspenders
        st.error("الملف فيه أعمدة بأسماء مكررة يصعب التعامل معها — أعيدي تسميتها.")
        return None
    return df


# --- Input: upload OR one-click sample -------------------------------------

col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a messy CSV or Excel file", type=["csv", "xlsx"])
with col_sample:
    st.write("")
    st.write("")
    if st.button(":material/science: Try with sample data (messy Arabic customers)", type="primary"):
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
    st.info("Upload a file — or click the sample button to see the system in action instantly.")
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

@st.cache_data(show_spinner=False)
def cached_profile(_df, key):  # key drives the cache; _df is not re-hashed
    return profile_dataframe(_df)


@st.cache_data(show_spinner=False)
def cached_score(_df, key):
    return quality_score(_df)


profile = cached_profile(df, file_id)
score_before, dims_before = cached_score(df, file_id)

DIM_LABELS = {
    "completeness": "Completeness — non-empty cells",
    "validity": "Validity — values matching their column's target format",
    "uniqueness": "Uniqueness — non-duplicate rows",
    "consistency": "Consistency — text free of representation noise",
}


def render_dimensions(dims: dict) -> None:
    for key, value in dims.items():
        st.progress(min(max(value, 0.0), 1.0), text=f"{DIM_LABELS.get(key, key)}: {value:.0%}")


st.subheader("1 · Profile")
c1, c2, c3 = st.columns(3, vertical_alignment="center")
with c1:
    st.markdown(gauge_html(score_before, "Quality score", card=True), unsafe_allow_html=True)
c2.metric("Rows", len(df))
c3.metric("Issues detected", len(profile["issues"]))
with st.expander("Quality dimensions (computed, never generated)", expanded=True):
    render_dimensions(dims_before)

with st.expander(f"Preview — {source_name}", expanded=True):
    st.dataframe(df.head(15), use_container_width=True)

if profile["issues"]:
    with st.expander(f"Detected issues ({len(profile['issues'])}) — click to view", expanded=False):
        for issue in profile["issues"]:
            st.warning(issue)

# --- Plan (LLM proposes; deterministic pandas executes) --------------------

st.subheader("2 · Cleaning plan (AI-proposed, you approve)")
st.caption("Privacy: the AI sees only aggregate statistics and 5 sample rows — never your full dataset.")

threshold = st.slider(
    "Target quality score (for the auto-optimize loop)", 85, 100, 95,
    help="The evaluator–optimizer loop keeps improving the plan until the measured score passes this threshold (max 3 iterations, best plan always kept).",
)
run_auto = st.button(":material/autorenew: Auto-optimize (evaluator–optimizer loop)", type="primary")

if run_auto:
    try:
        with st.status("Evaluator–optimizer loop running...", expanded=True) as status:
            best, history, rejected = run_loop(
                df, profile, threshold=float(threshold), on_event=status.write,
            )
            status.update(label="Loop finished", state="complete")
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
            f"Score climbed {history[0]['score']:.0f} → {max(h['score'] for h in history):.0f} "
            f"across {len(history)} iterations — showing the winning plan below."
        )
    else:
        st.info(
            f"Converged in 1 pass — the planner's first proposal already met the target "
            f"(score {history[0]['score']:.0f}). The optimizer only re-plans when a weakness remains."
        )

plan = st.session_state.get("plan")
if plan is not None:
    for note in st.session_state.get("rejected", []):
        st.error(f"Validator: {note}")
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
    st.caption("Impact previews below assume the full plan runs in order — unchecking earlier ops may change later effects.")
    approved = []
    for i, item in enumerate(plan):
        preview = previews[i]
        label = (
            f"`{item['op']}` on **{item.get('column') or 'whole table'}** — {item.get('reason', '')} "
            f"· **{preview['affected']}** affected"
        )
        checked = st.checkbox(label, value=True, key=f"op_{file_id}_{i}")
        mapping = (item.get("params") or {}).get("mapping")
        if preview["examples"] or preview["note"] or mapping:
            with st.expander("Preview impact", expanded=False):
                if preview["examples"]:
                    st.caption("e.g. " + " · ".join(f"`{e['before']}` → `{e['after']}`" for e in preview["examples"]))
                elif preview["note"]:
                    st.caption(preview["note"])
                if mapping:
                    st.dataframe(
                        pd.DataFrame([{"from": k, "to": v} for k, v in mapping.items()]),
                        hide_index=True,
                    )
        if checked:
            approved.append(item)

    # --- Apply (deterministic pandas only) ---------------------------------
    st.subheader("3 · Apply & download")
    if st.button(f":material/check_circle: Apply {len(approved)} approved operations",
                 disabled=not approved, key="apply_btn"):
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
        st.toast(f"Done — quality improved by {score_after - score_before:+.1f} points")
        if score_after >= threshold:
            st.balloons()

result = st.session_state.get("result")
if result is not None:
    clean, log = result["clean"], result["log"]
    score_after, issues_after = result["score_after"], result["issues_after"]

    st.markdown(
        '<div class="dq-gauge-row">'
        + gauge_html(score_before, "Before")
        + f'<div class="dq-arrow">{icon("arrow-right", 24)}</div>'
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

    tab_after, tab_before = st.tabs(["After (changed cells highlighted)", "Before"])
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
                lambda hit: "background-color: #E8F5EF; color: #1A7F5A; font-weight: 600" if hit else ""
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
        ":material/table: Clean CSV",
        clean.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"clean_{base_name}.csv",
        mime="text/csv",
    )
    d2.download_button(
        ":material/grid_on: Clean Excel",
        excel_buffer.getvalue(),
        file_name=f"clean_{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d3.download_button(
        ":material/description: التقرير التنفيذي (عربي)",
        report_html.encode("utf-8"),
        file_name=f"quality_report_{base_name}.html",
        mime="text/html",
    )
    d4.download_button(
        ":material/data_object: Audit log (JSON)",
        json.dumps(log, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"audit_log_{base_name}.json",
        mime="application/json",
    )
    st.caption("The Arabic executive report is a self-contained HTML — open it and print to PDF for management.")

    if score_after < score_before:
        st.caption(
            "Note: exposing hidden nulls ('N/A', '-') can lower the completeness score — "
            "that's honesty, not regression. The validity dimension (Phase 2) reflects the true gain."
        )

st.divider()
st.caption("Built with: a multi-agent evaluator–optimizer loop · deterministic pandas execution · a closed, validated operation registry · human-in-the-loop approval.")
