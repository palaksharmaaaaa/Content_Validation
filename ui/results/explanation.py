"""ui.results.explanation: the plain-English explanation of a result."""
from __future__ import annotations

from typing import Any

import streamlit as st

from image_detector.explain import generate_newbie_explanation
from ui.text import neutralise_links


def image_explanation(item: dict[str, Any], profile_data: dict[str, Any], ai_result: dict[str, Any],
                      content_res: dict[str, Any], decision: dict[str, Any], filename: str) -> str | None:
    """The item's precomputed explanation, or one generated on demand for items that predate it."""
    text = item.get("newbie_explanation")
    if not text and profile_data and ai_result:
        text = generate_newbie_explanation(
            filename=filename, profile_data=profile_data, content_inventory=content_res,
            ai_result=ai_result, decision=decision,
        )
    return text


def render_explanation(text: str | None) -> None:
    """Draw the plain-English explanation text."""
    if text:
        st.markdown(neutralise_links(text))
    else:
        st.caption("No explanation available for this file.")
