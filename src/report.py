"""Arabic executive report — deterministic HTML (RTL), zero LLM.

Every number in the report is computed; every sentence is a code template.
The output is a self-contained HTML file the user can download, open,
print to PDF, or attach to an email — written for management, not engineers.
"""

from __future__ import annotations

import html
from datetime import datetime

from .ops import REGISTRY

DIM_AR = {
    "completeness": "الاكتمال (خلايا غير فارغة)",
    "validity": "الصلاحية (مطابقة الصيغة المستهدفة)",
    "uniqueness": "التفرّد (بلا صفوف مكررة)",
    "consistency": "الاتساق (نص بلا تشويش تمثيل)",
}

OP_AR = {
    "trim_whitespace": "إزالة المسافات الزائدة",
    "normalize_arabic_text": "توحيد الألف والهمزات وإزالة التشكيل",
    "unify_numerals": "توحيد الأرقام الهندية والعربية",
    "standardize_nulls": "كشف القيم الفارغة المخفية",
    "normalize_phone_sa": "توحيد أرقام الجوال السعودية (+966)",
    "parse_dates": "توحيد صيغ التواريخ (ISO)",
    "to_numeric": "تحويل النصوص الرقمية إلى أرقام",
    "drop_exact_duplicates": "حذف الصفوف المكررة",
    "map_values": "توحيد القيم المتغايرة",
}


def _score_color(score: float) -> str:
    return "#2f9e44" if score >= 80 else "#e8590c" if score >= 50 else "#c92a2a"


def build_report(
    source_name: str,
    rows: int,
    score_before: float,
    dims_before: dict,
    score_after: float,
    dims_after: dict,
    log: list[dict],
    issues_before: int,
    issues_after: int,
) -> str:
    """Render the executive report as a self-contained RTL HTML string."""
    total_affected = sum(entry["affected"] for entry in log)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    dims_rows = ""
    for key, label in DIM_AR.items():
        if key not in dims_before and key not in dims_after:
            continue
        b = dims_before.get(key)
        a = dims_after.get(key)
        b_txt = f"{b:.0%}" if b is not None else "—"
        a_txt = f"{a:.0%}" if a is not None else "—"
        delta = ""
        if b is not None and a is not None:
            d = (a - b) * 100
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "◀▶")
            color = "#2f9e44" if d > 0 else ("#c92a2a" if d < 0 else "#868e96")
            delta = f'<span style="color:{color}">{arrow} {abs(d):.0f} نقطة</span>'
        dims_rows += f"<tr><td>{label}</td><td>{b_txt}</td><td>{a_txt}</td><td>{delta}</td></tr>"

    log_rows = ""
    for entry in log:
        op_label = OP_AR.get(entry["op"], REGISTRY.get(entry["op"], {}).get("desc", entry["op"]))
        reason = html.escape(str(entry.get("reason", "")))
        log_rows += (
            f"<tr><td>{op_label}</td><td dir='ltr'>{html.escape(str(entry['column']))}</td>"
            f"<td>{entry['affected']}</td><td>{reason}</td></tr>"
        )

    summary = (
        f"ارتفعت درجة جودة البيانات من <b>{score_before:.0f}</b> إلى <b>{score_after:.0f}</b> من 100 "
        f"بعد تطبيق <b>{len(log)}</b> عملية تنظيف معتمدة بشريًا، شملت <b>{total_affected}</b> خلية/صفًا، "
        f"وانخفضت المشاكل المكتشفة من <b>{issues_before}</b> إلى <b>{issues_after}</b>."
    )

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>تقرير جودة البيانات — {html.escape(source_name)}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Arial, sans-serif; margin: 2rem auto; max-width: 860px;
         color: #212529; line-height: 1.7; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  .meta {{ color: #868e96; font-size: 0.9rem; }}
  .cards {{ display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }}
  .card {{ flex: 1; min-width: 180px; border: 1px solid #dee2e6; border-radius: 12px;
          padding: 1rem 1.25rem; text-align: center; }}
  .card .num {{ font-size: 2.2rem; font-weight: 700; }}
  .summary {{ background: #f1f8f4; border-right: 4px solid #2f9e44; border-radius: 8px;
             padding: 1rem 1.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.75rem; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: right; font-size: 0.95rem; }}
  th {{ background: #f8f9fa; }}
  footer {{ margin-top: 2.5rem; color: #868e96; font-size: 0.85rem; border-top: 1px solid #dee2e6;
           padding-top: 0.75rem; }}
  @media print {{ body {{ margin: 0.5rem; }} }}
</style>
</head>
<body>
<h1>🧹 تقرير جودة البيانات — ملخص تنفيذي</h1>
<div class="meta">الملف: <span dir="ltr">{html.escape(source_name)}</span> · عدد الصفوف: {rows} · تاريخ التقرير: {generated_at}</div>

<div class="cards">
  <div class="card"><div>الجودة قبل التنظيف</div>
    <div class="num" style="color:{_score_color(score_before)}">{score_before:.0f}</div><div>من 100</div></div>
  <div class="card"><div>الجودة بعد التنظيف</div>
    <div class="num" style="color:{_score_color(score_after)}">{score_after:.0f}</div><div>من 100</div></div>
  <div class="card"><div>خلايا/صفوف عولجت</div>
    <div class="num">{total_affected}</div><div>عبر {len(log)} عملية</div></div>
</div>

<div class="summary">{summary}</div>

<h2>أبعاد الجودة (محسوبة برمجيًا)</h2>
<table>
  <tr><th>البُعد</th><th>قبل</th><th>بعد</th><th>التغير</th></tr>
  {dims_rows}
</table>

<h2>سجل العمليات المعتمدة (Audit Log)</h2>
<table>
  <tr><th>العملية</th><th>العمود</th><th>المتأثر</th><th>السبب</th></tr>
  {log_rows}
</table>

<footer>
أُنشئ هذا التقرير آليًا بواسطة <b>Data Quality Auto-Fixer</b> — نظام وكلاء ذكاء اصطناعي بنمط evaluator–optimizer.
كل الأرقام أعلاه <b>محسوبة برمجيًا</b> (لا مولّدة)، وكل عملية طُبقت <b>بعد اعتماد بشري</b>، والملف الأصلي لم يُمس.
</footer>
</body>
</html>"""
