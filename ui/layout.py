"""ui.layout: shared layout helpers. Nothing in the UI may be cut off with an ellipsis; text wraps instead.

``inject_css`` makes Streamlit's built-in widgets (metrics, buttons, tabs, selectboxes, upload chips, captions) wrap long
text. ``render_table`` replaces ``st.dataframe`` (whose canvas cells truncate) with a plain HTML table whose cells wrap.
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Sequence

import streamlit as st

_CSS = """
<style>
/* Wrap, never truncate */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere !important;
}
[data-testid="stMetricValue"] { font-size: 1.45rem !important; line-height: 1.25 !important; }
.stButton button, .stDownloadButton button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  height: auto !important; min-height: 2.5rem; padding-top: .4rem; padding-bottom: .4rem;
}
.stButton button *, .stDownloadButton button *, button[kind] p {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere !important;
}
button[role="tab"] p, button[role="tab"] { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileData"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere !important; max-width: none !important;
}
[data-baseweb="select"] *, [data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"], [data-baseweb="menu"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere !important;
}
[data-baseweb="select"] > div { height: auto !important; min-height: 2.5rem; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"], label, [data-testid="stWidgetLabel"] * {
  overflow-wrap: anywhere; white-space: normal !important; text-overflow: clip !important;
}
[data-testid="stCode"] pre, .stCodeBlock pre { white-space: pre-wrap !important; overflow-wrap: anywhere; }

/* Streamlit shortens upload-chip names itself ("IMG-201...0008.jpg"), so the chips are hidden and the full names are listed below */
[data-testid="stFileChips"], [data-testid="stFileUploaderPagination"] { display: none !important; }

/* Wrapping table */
.of-table-wrap { overflow-x: auto; margin: .25rem 0 1rem; }
.of-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
.of-table th { text-align: left; font-weight: 600; padding: .45rem .6rem; border-bottom: 2px solid rgba(128,128,128,.45); white-space: normal; }
.of-table td { padding: .4rem .6rem; border-bottom: 1px solid rgba(128,128,128,.25); vertical-align: top; overflow-wrap: anywhere; white-space: normal; }
.of-bar { display: inline-block; width: 5.5rem; height: .55rem; border-radius: .3rem; background: rgba(128,128,128,.3); margin-right: .5rem; vertical-align: middle; }
.of-bar > span { display: block; height: 100%; border-radius: .3rem; background: #ff4b4b; }
</style>
"""


def inject_css() -> None:
    """Add the no-truncation styles to the page (call once per run, near the top of the app)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_table(rows: Sequence[Dict[str, Any]], percent_bars: Iterable[str] = ()) -> None:
    """Draw ``rows`` (a list of dicts with the same keys) as a table whose cells wrap. Columns in ``percent_bars`` hold
    0-100 numbers drawn as a bar plus the value."""
    if not rows:
        return
    bars = set(percent_bars)
    cols: List[str] = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            if c in bars and isinstance(v, (int, float)):
                pct = max(0.0, min(100.0, float(v)))
                cells.append(f'<td><span class="of-bar"><span style="width:{pct:.0f}%"></span></span>{pct:.0f}%</td>')
            else:
                cells.append(f"<td>{html.escape('' if v is None else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f'<div class="of-table-wrap"><table class="of-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
                unsafe_allow_html=True)
