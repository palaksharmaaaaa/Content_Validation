"""The model warm-up starts once, tolerates a failing loader, and can be switched off."""
import threading

from services import warmup


def setup_function(_):
    warmup._reset_for_tests()


def test_runs_every_loader_once_in_a_daemon_thread_even_if_one_fails():
    calls = []

    def ok_a():
        calls.append("a")

    def broken():
        calls.append("broken")
        raise RuntimeError("no weights")

    def ok_b():
        calls.append("b")

    thread = warmup.start_background_warmup([ok_a, broken, ok_b])
    assert thread is not None and thread.daemon
    thread.join(timeout=10)
    assert calls == ["a", "broken", "b"]


def test_second_start_does_nothing():
    first = warmup.start_background_warmup([lambda: None])
    first.join(timeout=10)
    assert warmup.start_background_warmup([lambda: (_ for _ in ()).throw(AssertionError("must not run"))]) is None


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNI_WARMUP", "0")
    assert warmup.enabled() is False and warmup.start_background_warmup([lambda: None]) is None
    monkeypatch.setenv("OMNI_WARMUP", "off")
    assert warmup.start_background_warmup([lambda: None]) is None
    monkeypatch.delenv("OMNI_WARMUP")
    assert warmup.enabled() is True


def test_an_analysis_during_warmup_waits_for_the_model_instead_of_loading_twice():
    """Loaders take the model's own lock; two threads asking for it at once load it once."""
    loads = []
    lock = threading.Lock()
    state = {"model": None}

    def ensure():
        with lock:
            if state["model"] is None:
                loads.append(1)
                state["model"] = object()
            return state["model"]

    threads = [threading.Thread(target=ensure) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(loads) == 1
