import pandas as pd

from ui.batch_views import _row, csv_safe
from ui.text import code_safe, md_escape, neutralise_links


def test_markdown_image_and_link_are_escaped_exactly():
    assert md_escape("![x](http://a)") == "\\!\\[x\\]\\(http\\://a\\)"
    assert neutralise_links("**b** [l](u)") == "**b** \\[l\\](u)"          # bold kept, the link opener is disabled
    assert neutralise_links("<img src=x> $1") == "\\<img src=x\\> \\$1"


def test_escaped_text_never_contains_an_unescaped_link_opener():
    for raw in ("![x](https://evil.example/p.png)", "[click](https://evil.example)", "<img src=x>", "$$x$$", "a\\![b](c)"):
        for out in (md_escape(raw), neutralise_links(raw)):
            stripped = out.replace("\\\\", "")                        # drop escaped backslashes, then look at what is left
            assert not any(stripped[i] == "[" and (i == 0 or stripped[i - 1] != "\\") for i in range(len(stripped)))
            assert not any(stripped[i] == "<" and (i == 0 or stripped[i - 1] != "\\") for i in range(len(stripped)))


def test_code_span_cannot_be_closed_by_a_backtick():
    assert code_safe("a`b\nc\x00") == "a'b c"


def test_csv_formulas_are_prefixed():
    t = csv_safe(pd.DataFrame({"File": ["=HYPERLINK(\"http://x\")", "@a", "-1+1", "ok.png"], "n": [1, 2, 3, 4]}))
    assert list(t["File"]) == ["'=HYPERLINK(\"http://x\")", "'@a", "'-1+1", "ok.png"]


def test_blocked_file_is_not_reported_as_inconclusive_zero_percent():
    row = _row({"filename": "a.png", "success": True, "gate_blocked": True}, "image")
    assert "Blocked" in row["Verdict"] and row["AI likelihood"] is None
