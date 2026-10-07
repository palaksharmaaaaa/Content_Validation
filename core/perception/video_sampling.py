"""core.perception.video_sampling: choose which frames of a video to examine for minors, cheaply and adaptively.

Examining a frame costs about a second (people, faces, age), so frames cannot all be examined; sampling a fixed 16 evenly
spaced frames sees a child on screen for under four seconds only by luck in a minute-long clip. This module spends the budget
where it matters:

  1. Probe the clip at up to ``PROBES`` evenly spaced points and keep a tiny grey thumbnail of each (about 30 ms a probe).
  2. A probe whose thumbnail differs sharply from the one before it is a content change (a cut, or someone entering the shot):
     the first frames after every change are always examined.
  3. On top of that a grid of at most one frame per ``GRID_SECONDS`` is examined, so a child who walks into a continuous shot
     is seen within about a second, not whenever a change happens to be large enough to notice.
  4. A frame that looks the same as the last examined one reuses its result (static footage costs almost nothing).

With a budget of ``max_frames`` the grid thins out on long videos; the result reports the longest unexamined stretch so the
gap is never hidden.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("core.perception.video_sampling")

PROBES = 240
THUMB = (32, 18)               # width, height of the thumbnail
GRID_SECONDS = 1.0
MIN_GAP_SECONDS = 0.4          # two chosen frames closer than this are the same moment
EVENT_FLOOR = 0.03             # mean absolute thumbnail change (0..1) below which nothing counts as a change
EVENT_MEDIAN_FACTOR = 5.0      # ... and a change must also exceed this many times the clip's typical frame-to-frame change
EVENT_SECOND_FRAME = 0.5       # a second look this long after each change
REUSE_DIFF = 0.012             # a frame this similar to the last examined one is not examined again


def thumbnail(frame_bgr: np.ndarray) -> np.ndarray:
    """A 32 x 18 greyscale version of a frame, scaled 0..1."""
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    """How different two thumbnails are (0..1): the largest mean absolute difference over a 6 x 8 grid of blocks. A whole new scene
    scores high, and so does a person appearing in one corner; the plain average over the frame would dilute that to nothing."""
    d = np.abs(a - b)
    h, w = d.shape
    blocks = d[: h - h % 6, : w - w % 8].reshape(6, (h - h % 6) // 6, 8, (w - w % 8) // 8).mean(axis=(1, 3))
    return float(blocks.max())


@dataclass
class Probes:
    """Thumbnails of a video taken at evenly spaced moments."""
    times: List[float] = field(default_factory=list)
    thumbs: List[np.ndarray] = field(default_factory=list)
    duration: float = 0.0
    fps: float = 0.0


def probe_video(path, n_probes: int = PROBES) -> Probes:
    """Open the video and take up to ``n_probes`` thumbnails. An unreadable video gives an empty result."""
    cap = cv2.VideoCapture(str(path))
    out = Probes()
    try:
        if not cap.isOpened():
            return out
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25.0 if not fps or np.isnan(fps) or fps <= 0 else float(fps)
        if total <= 0:
            return out
        out.fps, out.duration = fps, total / fps
        for i in sorted({int(round(v)) for v in np.linspace(0, total - 1, min(n_probes, total))}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.times.append(i / fps)
                out.thumbs.append(thumbnail(frame))
    finally:
        cap.release()
    return out


def find_changes(probes: Probes) -> List[int]:
    """Indices of the probes that start a new shot or bring a visible change (never index 0)."""
    if len(probes.thumbs) < 2:
        return []
    diffs = np.array([thumb_diff(probes.thumbs[i], probes.thumbs[i - 1]) for i in range(1, len(probes.thumbs))])
    threshold = max(EVENT_FLOOR, EVENT_MEDIAN_FACTOR * float(np.median(diffs)))
    return [i + 1 for i, d in enumerate(diffs) if d > threshold]


def choose_times(probes: Probes, max_frames: int = 60) -> Tuple[List[float], dict]:
    """Moments (seconds) to examine, sorted, at most ``max_frames``: the start and end, the frames just after each content change,
    and a grid of at most one a second (thinned evenly when the budget cannot afford it)."""
    if not probes.times:
        return [], {"changes": 0, "grid_seconds": None}
    changes = find_changes(probes)
    priority: List[float] = [probes.times[0], probes.times[-1]]
    for i in changes:
        priority += [probes.times[i], min(probes.times[-1], probes.times[i] + EVENT_SECOND_FRAME)]
    spacing = max(GRID_SECONDS, probes.duration / max(1, max_frames))
    grid = list(np.arange(spacing / 2, probes.duration, spacing))

    def dedupe(times: Sequence[float], kept: List[float]) -> List[float]:
        for t in times:
            if all(abs(t - k) >= MIN_GAP_SECONDS for k in kept):
                kept.append(float(t))
        return kept

    chosen: List[float] = []
    dedupe(priority[: max(2, max_frames * 6 // 10)], chosen)       # changes may take at most 60 % of the budget
    dedupe(grid, chosen)
    dedupe(priority[max(2, max_frames * 6 // 10):], chosen)
    if len(chosen) > max_frames:
        chosen = sorted(chosen)
        keep = {int(round(v)) for v in np.linspace(0, len(chosen) - 1, max_frames)}
        chosen = [t for i, t in enumerate(chosen) if i in keep]
    return sorted(chosen), {"changes": len(changes), "grid_seconds": round(spacing, 2)}


def read_frames(path, times: Sequence[float]) -> List[Tuple[float, np.ndarray]]:
    """The frames at the given moments, as (time, BGR frame); moments that cannot be read are skipped."""
    cap = cv2.VideoCapture(str(path))
    frames: List[Tuple[float, np.ndarray]] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((float(t), frame))
    finally:
        cap.release()
    return frames


def longest_gap(times: Sequence[float], duration: float) -> Optional[float]:
    """The longest stretch of the video (seconds) between examined moments, counting from the start and to the end."""
    if not times or duration <= 0:
        return None
    edges = [0.0] + sorted(times) + [duration]
    return round(max(b - a for a, b in zip(edges, edges[1:])), 2)
