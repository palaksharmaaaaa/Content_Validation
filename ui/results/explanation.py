"""ui.results.explanation: the plain-English explanation of a result."""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from image_detector.explain import generate_newbie_explanation


def image_explanation(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any],
                      content_res: Dict[str, Any], decision: Dict[str, Any], filename: str) -> Optional[str]:
    """The item's precomputed explanation, or one generated on demand for items that predate it."""
    text = item.get("newbie_explanation")
    if not text and profile_data and ai_result:
        text = generate_newbie_explanation(
            filename=filename, profile_data=profile_data, content_inventory=content_res,
            ai_result=ai_result, decision=decision,
        )
    return text


def render_explanation(text: Optional[str]) -> None:
    """Draw the plain-English explanation text."""
    if text:
        st.markdown(text)
    else:
        st.caption("No explanation available for this file.")
