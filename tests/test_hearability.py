"""Tests for audi.training.hearability — ERB band scaling."""

import numpy as np

from audi.training.hearability import scale_to_db


class TestScaleToDB:
    def test_equal_power_no_scaling(self):
        """When drone and bg have equal power, SNR=0 means no scaling."""
        drone = np.sin(np.linspace(0, 100 * np.pi, 16000, dtype=np.float32))
        bg = drone.copy()
        scaled = scale_to_db(drone, bg, 0.0)
        # Should be close to the original
        assert np.abs(scaled - drone).mean() < 0.1

    def test_snr_positive_increases_gain(self):
        """Positive SNR should increase drone amplitude."""
        drone = np.ones(16000, dtype=np.float32) * 0.1
        bg = np.ones(16000, dtype=np.float32) * 0.1
        scaled = scale_to_db(drone, bg, 10.0)
        # Drone should be amplified
        assert scaled.max() > drone.max()
