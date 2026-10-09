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


def test_hard_block_list_is_parsed_once_and_reread_when_it_changes(tmp_path):
    import hashlib

    from core.forensics import gates as G

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"known bad")
    good = tmp_path / "good.bin"
    good.write_bytes(b"fine")
    lst = tmp_path / "list.txt"
    lst.write_text("# comment\n" + hashlib.sha256(b"known bad").hexdigest() + "  label\nnot-a-hash\n")
    gate = G.HardBlockGate(lst)
    assert gate.check(bad).status == "HARD_BLOCK_ESCALATE" and gate.check(good).status == "CLEAR"
    first = G._load_blocklist(lst)
    assert G._load_blocklist(lst) is first
    lst.write_text("")
    os.utime(lst, ns=(1, 3_000_000_000))
    assert gate.check(bad).status == "CLEAR"
