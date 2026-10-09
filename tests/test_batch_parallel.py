"""Batch analysis runs several files at once without changing results, order, or the thread Streamlit is called from."""
import threading
import time

import numpy as np
import pytest
from PIL import Image

from ui import adapters


def _items(n):
    return [{"path": f"p{i}", "filename": f"f{i}"} for i in range(n)]


def test_worker_count_is_bounded_and_overridable(monkeypatch):
    monkeypatch.delenv("OMNI_BATCH_WORKERS", raising=False)
    monkeypatch.setattr(adapters.os, "cpu_count", lambda: 16)
    assert adapters.batch_workers("image", 50) == 4 and adapters.batch_workers("video", 50) == 2
    assert adapters.batch_workers("image", 1) == 1 and adapters.batch_workers("audio", 3) == 3
    monkeypatch.setattr(adapters.os, "cpu_count", lambda: 2)
    assert adapters.batch_workers("image", 50) == 1                      # a dual-core machine stays sequential
    monkeypatch.setenv("OMNI_BATCH_WORKERS", "3")
    assert adapters.batch_workers("image", 50) == 3 and adapters.batch_workers("image", 2) == 2
    monkeypatch.setenv("OMNI_BATCH_WORKERS", "1")
    assert adapters.batch_workers("image", 50) == 1
    monkeypatch.setenv("OMNI_BATCH_WORKERS", "junk")
    assert adapters.batch_workers("image", 50) == 1                      # falls back to the cpu rule (2 cores)


def test_results_keep_input_order_errors_stay_local_and_callbacks_use_the_calling_thread(monkeypatch):
    monkeypatch.setenv("OMNI_BATCH_WORKERS", "4")
    delays = {0: 0.30, 1: 0.01, 2: 0.15, 3: 0.01, 4: 0.20, 5: 0.01}

    def fake(img_path, filename, **_):
        time.sleep(delays[int(filename[1:])])
        if filename == "f3":
            raise RuntimeError("boom")
        return {"filename": filename, "success": True, "thread": threading.current_thread().name}

    monkeypatch.setattr(adapters, "process_single_image", fake)
    seen, callers = [], set()

    def progress(done, total, name):
        callers.add(threading.current_thread().name)
        seen.append((done, total))

    out = adapters.run_batch_pipeline(_items(6), "image", {"detector": 1, "content_analyzer": 1, "attribution_engine": 1}, progress_callback=progress)
    assert [r["filename"] for r in out] == [f"f{i}" for i in range(6)]
    assert out[3]["success"] is False and "boom" in out[3]["error"] and all(r["success"] for i, r in enumerate(out) if i != 3)
    assert {r["thread"] for r in out if r["success"]} != {threading.current_thread().name}      # really ran on workers
    assert callers == {threading.current_thread().name}
    assert [d for d, _ in seen] == [1, 2, 3, 4, 5, 6] and all(t == 6 for _, t in seen)


def test_unknown_modality_is_an_error_result_not_an_exception():
    out = adapters.run_batch_pipeline(_items(2), "hologram", {})
    assert [r["success"] for r in out] == [False, False]


def test_real_pipeline_gives_identical_results_parallel_and_sequential(tmp_path, monkeypatch):
    from services.forensic_service import ForensicService

    svc = ForensicService.get_instance()
    ip = svc.image_pipeline
    rng = np.random.default_rng(8)
    items = []
    for i in range(4):
        p = tmp_path / f"b{i}.jpg"
        Image.fromarray(np.clip(rng.normal(90 + 20 * i, 45, (200, 260, 3)), 0, 255).astype(np.uint8)).save(p, quality=90)
        items.append({"path": str(p), "filename": p.name})
    detectors = {"detector": ip.detector, "content_analyzer": ip.content_analyzer, "attribution_engine": ip.attribution_engine}

    def verdicts(workers):
        monkeypatch.setenv("OMNI_BATCH_WORKERS", str(workers))
        res = adapters.run_batch_pipeline(items, "image", detectors)
        assert all(r["success"] for r in res)
        return [(r["filename"], r["decision"]["final_status"], r["decision"]["authenticity_probabilities"]) for r in res]

    assert verdicts(1) == verdicts(3)
