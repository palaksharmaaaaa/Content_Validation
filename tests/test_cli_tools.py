"""The blind-test and vocabulary-embedding command-line tools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from services import blind_test


def test_verdict_collapses_the_taxonomy_to_three_answers():
    assert blind_test.verdict_of("FULLY_AI_GENERATED") == "AI"
    assert blind_test.verdict_of("AUTHENTIC_REAL_PHOTOGRAPH") == "REAL"
    assert blind_test.verdict_of(None) == "UNSURE" and blind_test.verdict_of("SOMETHING_NEW") == "UNSURE"


def test_images_are_listed_in_a_stable_order_and_only_images(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.png", "a.JPG", "sub/c.webp", "notes.txt", "clip.mp4"):
        (tmp_path / name).write_bytes(b"x")
    found = [p.relative_to(tmp_path).as_posix() for p in blind_test.iter_images([tmp_path])]
    assert found == ["a.JPG", "b.png", "sub/c.webp"]


def _item(state: str) -> dict:
    return {"success": True, "decision": {"taxonomy_state": state, "authenticity_probabilities": {"p_ai": 90.0, "p_real": 5.0}},
            "content_res": {"entities": {"humans": {"faces_count": 1, "persons_count": 1}}, "minors": {"review_required": False}}}


def test_run_is_blind_resumable_and_survives_a_bad_file(tmp_path, monkeypatch):
    import services.forensic_service as fs
    import ui.adapters as adapters

    monkeypatch.setattr(fs.ForensicService, "get_instance", classmethod(lambda cls, *a, **k: type("S", (), {"image_detector": None, "content_analyzer": None, "attribution_engine": None})()))

    def fake(path, name, *args, **kwargs):
        if name == "bad.png":
            raise ValueError("boom")
        return _item("FULLY_AI_GENERATED")

    monkeypatch.setattr(adapters, "process_single_image", fake)
    for name in ("a.png", "bad.png"):
        (tmp_path / name).write_bytes(b"x")
    out = tmp_path / "out" / "blind.jsonl"
    assert blind_test.run([tmp_path], out) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["verdict"] for r in rows} == {"AI", "ERROR"} and "boom" in next(r for r in rows if r["verdict"] == "ERROR")["error"]
    assert blind_test.run([tmp_path], out) == 0                       # resumable: nothing is analysed twice
    assert "labels" not in json.dumps(rows)                           # the record never carries ground truth


def test_summary_scores_against_the_labels_given_afterwards(tmp_path):
    out = tmp_path / "blind.jsonl"
    rows = [{"path": "/data/AI/1.png", "verdict": "AI"}, {"path": "/data/AI/2.png", "verdict": "REAL"}, {"path": "/data/AI/3.png", "verdict": "UNSURE"},
            {"path": "/data/Real/1.png", "verdict": "REAL"}, {"path": "/data/x.png", "verdict": "ERROR", "error": "e"}]
    out.write_text("\n".join(json.dumps(r) for r in rows))
    report = blind_test.summary(out, {"data/AI": "ai", "data/Real": "real"})
    ai = report["groups"]["ai"]
    assert (ai["n"], ai["correct"], ai["wrong"], ai["unsure"]) == (3, 1, 1, 1) and ai["accuracy_all"] == round(1 / 3, 4)
    assert ai["accuracy_when_decided"] == 0.5 and report["groups"]["real"]["accuracy_all"] == 1.0
    assert report["errors"] == 1 and "unlabelled" in report["groups"]


def test_the_validation_split_matches_the_face_models_split(tmp_path):
    """summary --split val must give exactly the files face_training.split puts in validation (near-duplicates together)."""
    from image_detector.face_training import split

    rng = np.random.default_rng(5)
    files = []
    for i in range(40):
        p = tmp_path / f"p{i}.png"
        Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(p)
        files.append(p)
    out = tmp_path / "blind.jsonl"
    out.write_text("\n".join(json.dumps({"path": str(p), "verdict": "AI"}) for p in files))
    _train, val = split(files)
    got = blind_test.summary(out, {str(tmp_path): "ai"}, split="val")
    assert got["groups"]["ai"]["n"] == len(val) and len(val) > 0


def test_build_vocab_embeddings_reports_a_missing_model_instead_of_crashing(monkeypatch, capsys):
    from core.perception import recognizer
    from services import build_vocab_embeddings

    monkeypatch.setattr(recognizer.ZeroShotRecognizer, "_ensure", lambda self: None)
    assert build_vocab_embeddings.main() == 1
    assert "fetch_models" in capsys.readouterr().err


def test_build_vocab_embeddings_writes_one_array_per_vocabulary(monkeypatch, tmp_path, capsys):
    import torch

    from core.perception import recognizer
    from core.perception.vocab import VOCABS
    from services import build_vocab_embeddings

    target = tmp_path / "vocab.npz"
    monkeypatch.setattr(recognizer, "EMBEDDINGS_FILE", target)
    monkeypatch.setattr(recognizer.ZeroShotRecognizer, "_ensure", lambda self: object())
    monkeypatch.setattr(recognizer.ZeroShotRecognizer, "_text_features", lambda self, vocab: torch.ones(len(VOCABS[vocab]), 4))
    assert build_vocab_embeddings.main() == 0
    with np.load(target) as data:
        assert set(VOCABS) <= set(data.files) and "key" in data.files
        assert str(data["key"]) == recognizer.vocabulary_key()
    assert Path(target).is_file() and "prompts" in capsys.readouterr().out
