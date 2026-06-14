"""Tests for audi.training.validation — pure compute functions."""

import inspect

import numpy as np

from audi.hysteresis import apply_hysteresis
from audi.training.validation import (
    compute_calibration,
    compute_pr_curve,
    compute_roc_values,
    find_threshold_at_precision,
    split_by_bin,
    tpr_at_fpr,
)


def test_default_hysteresis_uses_documented_window_and_ratio():
    params = inspect.signature(apply_hysteresis).parameters

    assert params["window"].default == 8
    assert params["ratio"].default == 0.6
    assert params["margin"].default == 0.05


def test_hysteresis_ratio_uses_ceiling_not_floor():
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.56, 0.56, 0.56, 0.56, 0.56])

    detections = apply_hysteresis(scores, threshold=0.5)

    assert not detections[7]
    assert detections[8]


class TestComputeROCVALUES:
    def test_perfect_separation(self):
        """Perfect classifier should get AUC=1.0."""
        logits = np.array([-10.0, -10.0, 10.0, 10.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        _, _, _, auc = compute_roc_values(logits, labels)
        assert auc > 0.99

    def test_random_predictions(self):
        """Random predictions should get AUC near 0.5."""
        rng = np.random.RandomState(42)
        logits = rng.randn(1000).astype(np.float32)
        labels = (rng.rand(1000) > 0.5).astype(np.float32)
        _, _, _, auc = compute_roc_values(logits, labels)
        assert 0.4 < auc < 0.6

    def test_degenerate_all_positive(self):
        """All-positive input should return zeros."""
        logits = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 1.0, 1.0])
        fpr, tpr, _, auc = compute_roc_values(logits, labels)
        assert (fpr == 0).all()
        assert (tpr == 0).all()
        assert auc == 0.0

    def test_degenerate_all_negative(self):
        """All-negative input should return zeros."""
        logits = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.0, 0.0, 0.0])
        fpr, tpr, _, auc = compute_roc_values(logits, labels)
        assert (fpr == 0).all()
        assert auc == 0.0


class TestTPRAtFPR:
    def test_interpolation(self):
        fpr = np.array([1.0, 0.5, 0.0])
        tpr = np.array([1.0, 0.5, 0.0])
        assert tpr_at_fpr(fpr, tpr, 0.5) == 0.5

    def test_out_of_range(self):
        fpr = np.array([0.5, 0.0])
        tpr = np.array([0.5, 0.0])
        assert tpr_at_fpr(fpr, tpr, 1.0) == 0.0


class TestSplitByBin:
    def test_groups_correctly(self):
        logits = np.array([1.0, 2.0, 3.0, -1.0, -2.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        bins = ["a", "a", "", "", ""]
        result = split_by_bin(logits, labels, bins)
        assert "a" in result
        logits_a, labels_a = result["a"]
        # Both positives + all 3 negatives = 5 samples
        assert len(logits_a) == 5


class TestComputePRCurve:
    def test_perfect_separation(self):
        """Perfect classifier should have high AP. Needs enough samples for
        PR curve resolution."""
        rng = np.random.RandomState(42)
        n = 500
        logits = np.concatenate([
            rng.normal(-5, 1, n).astype(np.float32),   # negatives
            rng.normal(5, 1, n).astype(np.float32),    # positives
        ])
        labels = np.concatenate([
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        ])
        _, _, _, ap = compute_pr_curve(logits, labels)
        assert ap > 0.94


class TestComputeCalibration:
    def test_perfect_calibration(self):
        """Well-calibrated logits should have low ECE."""
        rng = np.random.RandomState(42)
        probs = rng.rand(1000).astype(np.float32)
        labels = (rng.rand(1000) < probs).astype(np.float32)
        logits = np.log(probs / (1 - probs)).astype(np.float32)
        _, _, _, ece = compute_calibration(logits, labels)
        assert ece < 0.3


class TestFindThresholdAtPrecision:
    def test_finds_target(self):
        precision = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 1.0])
        thresholds = np.array([0.0, 0.5, 1.0])
        th, tp, prec = find_threshold_at_precision(precision, tpr, thresholds, 0.5)
        assert th == 0.5
        assert tp == 0.5
        assert prec == 0.5
