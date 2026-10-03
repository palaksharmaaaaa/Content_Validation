"""Video signal checks: interlacing/combing and temporal cadence (duplicates, pulldown)."""
import cv2
import numpy as np
import pytest

from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus
from video_detector.dimension_checks import _common as C
from video_detector.dimension_checks import signal
from video_detector.tests.video_fixtures import box, build_mp4, fiel, moving_square_frames, trak, write_clip


def _gray(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def _ctx(path, frames=None):
    extra = {"frames": frames} if frames is not None else {}
    return CheckContext(path=path, modality="video", extra=extra)


def _interlace(frames):
    out = []
    for a, b in zip(frames[:-1], frames[1:]):
        f = a.copy()
        f[1::2] = b[1::2]
        out.append(f)
    return out


# ---------------- interlacing ----------------
def test_progressive_content_not_combed(tmp_path):
    frames = _gray(moving_square_frames(30, step=6))
    f = signal.check_interlacing(_ctx(tmp_path / "x.mp4", frames))
    assert f.status == FindingStatus.PASS
    assert f.data["combing_ratio"] < 1.0
    assert f.evidence_class == EvidenceClass.PHYSICAL_SIGNAL and f.llr is None


def test_interlaced_motion_is_combed(tmp_path):
    frames = _gray(_interlace(moving_square_frames(30, step=6)))
    f = signal.check_interlacing(_ctx(tmp_path / "x.mp4", frames))
    assert f.status == FindingStatus.INFO
    assert f.data["combed"] is True and f.data["combing_ratio"] > 1.0


def test_container_fiel_flag_reported(tmp_path):
    p = tmp_path / "i.mp4"
    p.write_bytes(build_mp4([trak(entry_children=fiel(2, 1))]))
    frames = _gray(moving_square_frames(30, step=6))
    f = signal.check_interlacing(_ctx(p, frames))
    assert f.data["container_fiel"] == {"count": 2, "order": 1}
    assert f.status == FindingStatus.INFO  # flagged interlaced but no combing measured


def test_static_video_not_enough_motion(tmp_path):
    frames = [np.full((120, 160), 100, np.uint8) for _ in range(20)]
    assert signal.check_interlacing(_ctx(tmp_path / "x.mp4", frames)).status == FindingStatus.NOT_APPLICABLE


# ---------------- temporal cadence ----------------
def _with_repeats(frames, every):
    out = []
    for i, f in enumerate(frames):
        out.append(f)
        if (i + 1) % every == 0:
            out.append(f.copy())
    return out


def test_regular_pulldown_cadence(tmp_path):
    frames = _gray(_with_repeats(moving_square_frames(40, step=5), 4))  # one repeat per 5 output frames
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", frames))
    assert f.data["pattern"] == "PULLDOWN_3_2"
    assert f.status == FindingStatus.INFO


def test_regular_duplication_other_period(tmp_path):
    frames = _gray(_with_repeats(moving_square_frames(40, step=5), 7))
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", frames))
    assert f.data["pattern"] == "REGULAR_DUPLICATION" and f.data["period"] == 8


def test_isolated_duplicates(tmp_path):
    base = moving_square_frames(60, step=4)
    frames = base[:10] + [base[10]] + base[10:30] + [base[29]] + base[30:]
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", _gray(frames)))
    assert f.data["pattern"] == "ISOLATED_DUPLICATES"


def test_heavy_duplication_low_true_frame_rate(tmp_path):
    base = moving_square_frames(25, step=6)
    frames = [f for fr in base for f in (fr, fr.copy())]
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", _gray(frames)))
    assert f.data["pattern"] == "HEAVY_DUPLICATION"


def test_clean_motion_passes(tmp_path):
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", _gray(moving_square_frames(40, step=5))))
    assert f.status == FindingStatus.PASS and f.data["pattern"] == "NONE"


def test_hard_cut_counted(tmp_path):
    a = moving_square_frames(20, step=4)
    b = [np.full_like(a[0], 20) for _ in range(1)] + [np.roll(f, 20, axis=0)[::-1] for f in moving_square_frames(19, step=4)]
    f = signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", _gray(a + b)))
    assert f.data["hard_cuts"] >= 1


def test_too_few_frames(tmp_path):
    assert signal.check_temporal_cadence(_ctx(tmp_path / "x.mp4", _gray(moving_square_frames(10)))).status == FindingStatus.NOT_APPLICABLE


# ---------------- frame reading helper ----------------
def test_read_consecutive_gray(tmp_path):
    p = write_clip(tmp_path / "c.mp4", moving_square_frames(40), fps=25)
    frames = C.read_consecutive_gray(p, n=30, max_width=96)
    assert len(frames) == 30
    assert frames[0].ndim == 2 and frames[0].shape[1] <= 96


def test_checks_read_frames_themselves_when_missing(tmp_path):
    p = write_clip(tmp_path / "c.mp4", moving_square_frames(60, step=5), fps=25)
    f = signal.check_temporal_cadence(_ctx(p))
    assert f.status in (FindingStatus.PASS, FindingStatus.INFO)
