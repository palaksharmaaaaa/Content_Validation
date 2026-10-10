"""Five probability bands (the two high-confidence ones are off until an operator calibrates)."""
import math

import pytest

from core import bands
from core.bands import Band, classify_band, classify_band_from_percent


@pytest.mark.parametrize(
    "p,expected",
    [
        (1.0, Band.HIGH_CONFIDENCE_SYNTHETIC),
        (0.995, Band.HIGH_CONFIDENCE_SYNTHETIC),
        (0.9949, Band.LEANING_SYNTHETIC),
        (0.61, Band.LEANING_SYNTHETIC),
        (0.6001, Band.LEANING_SYNTHETIC),
        (0.60, Band.INCONCLUSIVE),
        (0.50, Band.INCONCLUSIVE),
        (0.40, Band.INCONCLUSIVE),
        (0.3999, Band.LEANING_AUTHENTIC),
        (0.10, Band.LEANING_AUTHENTIC),
        (0.0051, Band.LEANING_AUTHENTIC),
        (0.005, Band.HIGH_CONFIDENCE_AUTHENTIC),
        (0.0, Band.HIGH_CONFIDENCE_AUTHENTIC),
    ],
)
def test_band_boundaries(p, expected, monkeypatch):
    monkeypatch.setattr(bands, "HIGH_CONFIDENCE_ENABLED", True)
    assert classify_band(p) == expected


def test_percent_helper(monkeypatch):
    monkeypatch.setattr(bands, "HIGH_CONFIDENCE_ENABLED", True)
    assert classify_band_from_percent(99.5) == Band.HIGH_CONFIDENCE_SYNTHETIC
    assert classify_band_from_percent(50.0) == Band.INCONCLUSIVE
    assert classify_band_from_percent(0.4) == Band.HIGH_CONFIDENCE_AUTHENTIC
    assert classify_band_from_percent(1.0) == Band.LEANING_AUTHENTIC  # 1% is 0.01, not 1.0


def test_out_of_range_is_clamped(monkeypatch):
    monkeypatch.setattr(bands, "HIGH_CONFIDENCE_ENABLED", True)
    assert classify_band(1.7) == Band.HIGH_CONFIDENCE_SYNTHETIC
    assert classify_band(-0.2) == Band.HIGH_CONFIDENCE_AUTHENTIC


def test_band_has_label():
    assert Band.INCONCLUSIVE.label == "Inconclusive / Indeterminate Evidence"


def test_without_calibration_the_strongest_band_is_leaning():
    assert bands.HIGH_CONFIDENCE_ENABLED is False
    assert classify_band(1.0) == Band.LEANING_SYNTHETIC and classify_band(0.0) == Band.LEANING_AUTHENTIC


def test_nan_is_inconclusive_not_certain():
    assert classify_band(math.nan) == Band.INCONCLUSIVE and classify_band(None) == Band.INCONCLUSIVE
