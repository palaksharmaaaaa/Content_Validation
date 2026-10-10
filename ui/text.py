"""ui.text: make text that came from a file (EXIF, tags, file names, error messages) safe to show through Streamlit's markdown.

An uploaded file controls those strings. Left raw, a camera "Software" tag of ``![x](https://host/p.png)`` makes the reviewer's browser
fetch a remote image (a tracking beacon), ``[text](https://...)`` plants a link, and ``$...$`` or ``:emoji:`` rewrite what is shown.
"""
from __future__ import annotations

import re
from typing import Any

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$:&-])")
_LINK_BITS = re.compile(r"([\[\]!<>$&])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(value: Any) -> str:
    return _CONTROL.sub("", "" if value is None else str(value))


def md_escape(value: Any) -> str:
    """``value`` as plain text: every markdown, HTML and LaTeX metacharacter is backslash-escaped, control characters dropped."""
    return _MD_SPECIAL.sub(lambda m: "\\" + m.group(1), _clean(value))


def code_safe(value: Any) -> str:
    """``value`` for use inside an inline code span: a backtick would close the span, so it is replaced."""
    return _clean(value).replace("`", "'").replace("\n", " ")


def neutralise_links(value: Any) -> str:
    """Keep bold/italic/lists of generated markdown but disable links, images, raw HTML and maths in ``value``."""
    return _LINK_BITS.sub(lambda m: "\\" + m.group(1), _clean(value))
