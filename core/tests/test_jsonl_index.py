import json
import os

from core.forensics.jsonl_index import load_index


def _prep(row):
    int(row["phash"], 16)


def test_bad_lines_are_skipped_and_the_result_is_cached_until_the_file_changes(tmp_path):
    p = tmp_path / "i.jsonl"
    p.write_text(json.dumps({"phash": "ff"}) + "\nnot json\n" + json.dumps({"phash": "zz"}) + "\n\n" + json.dumps({"phash": "0a"}) + "\n")
    first = load_index(p, _prep)
    assert [r["phash"] for r in first] == ["ff", "0a"]
    assert load_index(p, _prep) is first                              # unchanged file: parsed once
    p.write_text(json.dumps({"phash": "01"}) + "\n")
    os.utime(p, ns=(1, 2_000_000_000))
    assert [r["phash"] for r in load_index(p, _prep)] == ["01"]


def test_missing_index_is_empty(tmp_path):
    assert load_index(tmp_path / "nope.jsonl", _prep) == []
