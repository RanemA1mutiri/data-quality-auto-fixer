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


# --- Theme: follow Streamlit's NATIVE theme (⋮ menu → Settings → Theme) so
# the whole app — tables and chrome included — switches reliably with zero
# engine tampering. We only mirror the active theme into our custom surfaces. ---
LIGHT = {
    "base": "light", "bg": "#FBFBFD", "surface": "#FFFFFF", "surface2": "#F1F3F5",
    "border": "#E6E8EB", "text": "#1A1D24", "text2": "#5A6472", "muted": "#98A2B3",
    "chip_text": "#475467", "primary": "#4F46E5", "primary_hover": "#4338CA",
    "success": "#1A7F5A", "warning": "#B45309", "danger": "#B42318",
    "track": "#EAECF0", "gauge_inner": "#FBFBFD", "hl_bg": "#E8F5EF", "hl_text": "#1A7F5A",
}
DARK = {
    "base": "dark", "bg": "#0D1117", "surface": "#161B22", "surface2": "#21262D",
    "border": "#30363D", "text": "#E6EDF3", "text2": "#8B949E", "muted": "#7D8590",
    "chip_text": "#8B949E", "primary": "#6366F1", "primary_hover": "#818CF8",
    "success": "#3FB950", "warning": "#D29922", "danger": "#F85149",
    "track": "#21262D", "gauge_inner": "#0D1117", "hl_bg": "#0E2A1B", "hl_text": "#56D364",
}
def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


IS_DARK = _is_dark()
P = DARK if IS_DARK else LIGHT

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', 'IBM Plex Sans Arabic', sans-serif; }}

.stApp {{ background: {P['bg']}; }}

/* Hero */
.dq-hero {{ padding: .5rem 0 1.4rem; }}
.dq-hero h1 {{ font-size: 1.9rem; font-weight: 600; margin: 0 0 .5rem; letter-spacing: -.01em; color: {P['text']}; }}
.dq-hero h1 svg {{ color: {P['primary']}; }}
.dq-tagline {{ color: {P['text2']}; font-size: 1.05rem; margin: 0 0 1.1rem; }}
.dq-chips {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.dq-chip {{
  display: inline-flex; align-items: center; font-size: .82rem; font-weight: 500;
  color: {P['chip_text']}; padding: .34rem .75rem; border-radius: 6px;
  border: 1px solid {P['border']}; background: {P['surface2']};
}}
.dq-chip svg {{ color: {P['text2']}; }}
.dq-steps {{ display: flex; align-items: center; gap: .35rem; flex-wrap: wrap;
  color: {P['text2']}; font-size: .92rem; margin-top: 1.1rem; }}
.dq-steps svg {{ color: {P['primary']}; }}
.dq-steps .sep {{ color: {P['muted']}; margin: 0 .2rem; }}

/* Headings — full presence in both modes */
.stApp h1, .stApp h2, .stApp h3 {{ color: {P['text']}; }}

/* Metric cards */
[data-testid="stMetric"] {{
  background: {P['surface']};
  border: 1px solid {P['border']};
  border-radius: 14px; padding: 1.2rem 1.35rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.14), 0 2px 6px rgba(0,0,0,.10);
}}
[data-testid="stMetricValue"] {{ font-weight: 700; font-size: 2.1rem; color: {P['text']}; }}
[data-testid="stMetricLabel"] p {{ color: {P['text2']} !important; font-weight: 500; }}

/* Buttons */
.stButton > button {{ border-radius: 8px; font-weight: 500; }}
.stButton > button[kind="primary"] {{
  background: {P['primary']}; border: 0; color: #fff; font-weight: 600;
  box-shadow: 0 1px 2px rgba(0,0,0,.15);
}}
.stButton > button[kind="primary"]:hover {{ background: {P['primary_hover']}; }}

/* Progress bars (dimension bars) */
[data-testid="stProgress"] > div > div > div {{ background: {P['primary']}; }}

/* Expanders */
[data-testid="stExpander"] {{ border: 1px solid {P['border']}; border-radius: 12px; background: {P['surface']}; }}

/* Score gauges */
.dq-gauge-row {{ display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; margin: .4rem 0 1rem; }}
.dq-gauge-wrap {{ display: flex; flex-direction: column; align-items: center; gap: .6rem; }}
.dq-gauge {{
  width: 148px; height: 148px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
}}
.dq-gauge-inner {{
  width: 116px; height: 116px; border-radius: 50%; background: {P['bg']};
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}}
.dq-gauge-num {{ font-size: 2.25rem; font-weight: 600; line-height: 1.05; color: {P['text']}; }}
.dq-gauge-sub {{ color: {P['muted']}; font-size: .85rem; }}
.dq-gauge-label {{ color: {P['text2']}; font-size: .92rem; font-weight: 500; }}
.dq-arrow {{ color: {P['muted']}; display: flex; align-items: center; }}
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


def gauge_html(score: float, label: str) -> str:
    """Circular quality gauge — pure CSS conic-gradient, no external libs."""
    pct = max(0.0, min(100.0, score))
    color = P["success"] if pct >= 90 else P["warning"] if pct >= 65 else P["danger"]
    return (
        f'<div class="dq-gauge-wrap">'
        f'<div class="dq-gauge" style="background: conic-gradient({color} {pct * 3.6}deg, {P['track']} 0deg);">'
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

score_before, dims_before = quality_score(df)
profile = profile_dataframe(df)

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
    if st.button(f":material/check_circle: Apply {len(approved)} approved operations", disabled=not approved):
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
                lambda hit: f"background-color: {P['hl_bg']}; color: {P['hl_text']}; font-weight: 600" if hit else ""
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
st.caption("Roadmap: Phase 3 — per-op dry-run preview · Phase 4 — Arabic executive report + audit export.")
