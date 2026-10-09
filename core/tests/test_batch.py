import pytest

from core.batch import SequentialBatch, bucket_of


@pytest.mark.parametrize("status,bucket", [
    ("LIKELY_SYNTHETIC", "ai_generated"), ("LIKELY AI-GENERATED", "ai_generated"),
    ("PARTIALLY_SYNTHETIC_OR_EDITED", "composite"), ("LIKELY_AUTHENTIC", "real"), ("LIKELY REAL", "real"),
    ("BLANK_OR_DEGRADED", "no_content"), ("UNDETERMINED", "undecided"), ("UNDECIDED", "undecided"), ("something new", "undecided"),
])
def test_every_real_status_lands_in_its_bucket(status, bucket):
    assert bucket_of({"final_status": status}) == bucket


def test_errors_win():
    assert bucket_of({"final_status": "LIKELY_SYNTHETIC", "error": "x"}) == "errors"
    assert bucket_of({"content_valid": False}) == "errors"


class _Pipe:
    def analyze(self, path, sensitivity="balanced"):
        if "bad" in str(path):
            raise RuntimeError("boom")
        return {"final_status": "LIKELY_SYNTHETIC"}


def test_one_failing_file_does_not_stop_the_run(tmp_path):
    seen = []
    out = SequentialBatch(_Pipe()).process_files(["a.jpg", "bad.jpg", "c.jpg"], progress_callback=lambda i, n, r: seen.append((i, n)))
    assert out["summary"]["ai_generated"] == 2 and out["summary"]["errors"] == 1
    assert seen == [(1, 3), (2, 3), (3, 3)] and "boom" in out["results"][1]["error"]


def test_directory_scan_is_sorted_and_filtered(tmp_path):
    for n in ("b.jpg", "a.jpg", "c.txt"):
        (tmp_path / n).write_bytes(b"x")

    class B(SequentialBatch):
        extensions = {".jpg"}

    out = B(_Pipe()).process_directory(tmp_path)
    assert out["total_processed"] == 2
