"""Tests for audi.augment."""

import numpy as np
import pytest

from audi.augment import (
    _fit_length,
    _rms,
    gain_jitter,
    highpass,
    lowpass,
    peak_limit,
)


class TestRMS:
    def test_unit_tone(self):
        """RMS of a unit-amplitude sine should be ~0.707."""
        t = np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)
        y = np.sin(t).astype(np.float32)
        r = _rms(y)
        assert 0.6 < r < 0.8, f"Expected ~0.707, got {r}"

    def test_silence(self):
        y = np.zeros(100, dtype=np.float32)
        r = _rms(y)
        # RMS formula uses sqrt(mean(y^2) + eps), so with eps=1e-8 we get ~1e-4
        assert r < 1e-3


class TestFitLength:
    def test_exact_length(self):
        y = np.ones(100, dtype=np.float32)
        result = _fit_length(y, 100)
        assert len(result) == 100

    def test_short_gets_tiled(self):
        y = np.array([1.0, 2.0], dtype=np.float32)
        result = _fit_length(y, 10)
        assert len(result) == 10

    def test_long_gets_cropped(self):
        y = np.arange(200, dtype=np.float32)
        result = _fit_length(y, 50)
        assert len(result) == 50

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _fit_length(np.array([], dtype=np.float32), 100)


class TestGainJitter:
    def test_zero_db_no_change(self):
        y = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        result = gain_jitter(y, 0.0)
        np.testing.assert_array_almost_equal(result, y)


class TestPeakLimit:
    def test_below_threshold_no_change(self):
        y = np.array([0.5, -0.3], dtype=np.float32)
        result = peak_limit(y, peak_target=0.98)
        np.testing.assert_array_equal(result, y)

    def test_above_threshold_clamped(self):
        y = np.array([1.5, -0.3], dtype=np.float32)
        result = peak_limit(y, peak_target=0.98)
        assert np.abs(result).max() <= 0.98


class TestHighpass:
    def test_below_cutoff_attenuated(self):
        """Low frequencies should be attenuated by a highpass filter."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        y = np.sin(2 * np.pi * 50 * t).astype(np.float32)  # 50 Hz tone
        filtered = highpass(y, cutoff_hz=125, sample_rate=sr)
        assert _rms(filtered) < _rms(y) * 0.5


class TestLowpass:
    def test_above_cutoff_attenuated(self):
        """High frequencies should be attenuated by a lowpass filter."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        y = np.sin(2 * np.pi * 5000 * t).astype(np.float32)  # 5 kHz tone
        filtered = lowpass(y, sr=sr, cutoff_range=(1000, 1000))
        assert _rms(filtered) < _rms(y) * 0.5
