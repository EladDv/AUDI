"""Pure compute functions for validation metrics.

No matplotlib imports — pure numpy computation.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Safe sigmoid: clip logits to prevent exp overflow in float64."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))


def compute_roc_values(
    logits: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(fpr, tpr, thresholds, auc)`` from raw logits and binary labels.

    Args:
        logits: Raw model logits.
        labels: Binary labels (0.0 or 1.0).
        num_thresholds: Number of probability thresholds to evaluate.

    Returns:
        Tuple of (fpr, tpr, thresholds, auc). Thresholds are sigmoid
        probabilities in [0, 1]. AUC via trapezoidal integration.
    """
    probs = _sigmoid(logits)

    if len(probs) == 0 or int(labels.sum()) == 0 or int((1 - labels).sum()) == 0:
        th = np.linspace(0.0, 1.0, num_thresholds)
        return np.zeros(num_thresholds), np.zeros(num_thresholds), th, 0.0

    th = np.linspace(0.0, 1.0, num_thresholds)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    fpr = np.empty(num_thresholds, dtype=np.float64)
    tpr = np.empty(num_thresholds, dtype=np.float64)

    pos_probs = probs[labels > 0.5]
    neg_probs = probs[labels < 0.5]

    for i, t in enumerate(th):
        tp = int((pos_probs >= t).sum())
        fp = int((neg_probs >= t).sum())
        tpr[i] = tp / max(n_pos, 1)
        fpr[i] = fp / max(n_neg, 1)

    sort_idx = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[sort_idx], fpr[sort_idx]))
    return fpr, tpr, th, auc


def compute_precision(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute precision = TP / (TP + FP) at each sigmoid probability threshold.

    Args:
        logits: Raw model logits.
        labels: Binary labels.
        thresholds: Probability thresholds in [0, 1].

    Returns:
        Precision array matching ``thresholds`` shape.
    """
    probs = _sigmoid(logits)
    pos = probs[labels > 0.5]
    neg = probs[labels < 0.5]
    prec = np.empty(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        tp = int((pos >= t).sum())
        fp = int((neg >= t).sum())
        prec[i] = tp / max(tp + fp, 1)
    return prec


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Return TPR at a given FPR via linear interpolation.

    Args:
        fpr: False positive rate array (decreasing).
        tpr: True positive rate array.
        target_fpr: Target FPR value.

    Returns:
        TPR at the target FPR, or 0.0 if out of range.
    """
    if len(fpr) == 0 or target_fpr < fpr.min() or target_fpr > fpr.max():
        return 0.0
    return float(np.interp(target_fpr, fpr[::-1], tpr[::-1]))


def find_threshold_at_precision(
    precision: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target: float,
) -> tuple[float, float, float]:
    """Return ``(threshold, tpr, actual_precision)`` nearest target precision.

    Args:
        precision: Precision values.
        tpr: True positive rate values.
        thresholds: Corresponding probability thresholds.
        target: Desired precision value.

    Returns:
        (threshold, tpr_at_threshold, actual_precision) or (nan, nan, nan) if empty.
    """
    if len(precision) == 0:
        return float("nan"), float("nan"), float("nan")
    best = int(np.argmin(np.abs(precision - target)))
    return float(thresholds[best]), float(tpr[best]), float(precision[best])


def compute_pr_curve(
    logits: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(precision, recall, thresholds, average_precision)``.

    Args:
        logits: Raw model logits.
        labels: Binary labels.
        num_thresholds: Number of thresholds.

    Returns:
        (precision, recall, thresholds, average_precision).
    """
    probs = _sigmoid(logits)
    if len(probs) == 0 or int(labels.sum()) == 0 or int((1 - labels).sum()) == 0:
        th = np.linspace(0.0, 1.0, num_thresholds)
        return np.zeros(num_thresholds), np.zeros(num_thresholds), th, 0.0

    th = np.linspace(0.0, 1.0, num_thresholds)
    precision = np.empty(num_thresholds, dtype=np.float64)
    recall = np.empty(num_thresholds, dtype=np.float64)
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())

    for i, t in enumerate(th):
        pred_pos = probs >= t
        tp = int((pred_pos & pos_mask).sum())
        fp = int((pred_pos & neg_mask).sum())
        precision[i] = tp / max(tp + fp, 1)
        recall[i] = tp / max(n_pos, 1)

    sort_idx = np.argsort(recall)
    ap = float(np.trapezoid(precision[sort_idx], recall[sort_idx]))
    return precision, recall, th, ap


def compute_det_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    clip: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (FPR, TPR) to probit-scale (DET) coordinates.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        clip: Clipping bound to avoid ±∞.

    Returns:
        (probit_fpr, probit_fnr).
    """
    from scipy.special import ndtri

    fnr = 1.0 - np.asarray(tpr, dtype=np.float64)
    fpr_c = np.clip(np.asarray(fpr, dtype=np.float64), clip, 1.0 - clip)
    fnr_c = np.clip(fnr, clip, 1.0 - clip)
    return ndtri(fpr_c).astype(np.float64), ndtri(fnr_c).astype(np.float64)


def compute_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(expected_prob, observed_freq, bin_counts, ece)``.

    Args:
        logits: Raw model logits.
        labels: Binary labels.
        n_bins: Number of calibration bins.

    Returns:
        (expected, observed, counts, ece). ECE = Expected Calibration Error.
    """
    probs = _sigmoid(logits)
    if len(probs) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    expected = np.zeros(n_bins, dtype=np.float64)
    observed = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            expected[i] = float(probs[mask].mean())
            observed[i] = float(labels[mask].mean())

    mask_nonzero = counts > 0
    if mask_nonzero.any():
        ece = float(
            np.average(
                np.abs(expected[mask_nonzero] - observed[mask_nonzero]),
                weights=counts[mask_nonzero],
            )
        )
    else:
        ece = 0.0

    return expected[mask_nonzero], observed[mask_nonzero], counts[mask_nonzero], ece


def split_by_bin(
    logits: np.ndarray,
    labels: np.ndarray,
    bin_names: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split validation results per SNR bin.

    Args:
        logits: Raw model logits.
        labels: Binary labels.
        bin_names: SNR bin name for each sample (empty = negative/unlabelled).

    Returns:
        Dict ``{bin_name: (logits, labels)}``. Each bin includes ALL negatives
        for valid FPR computation.
    """
    labels_a = np.asarray(labels)
    logits_a = np.asarray(logits)
    neg_mask = labels_a < 0.5
    neg_logits = logits_a[neg_mask]

    bins: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unique = sorted(set(n for n in bin_names if n))
    for bn in unique:
        pos_mask = (labels_a > 0.5) & (np.array(bin_names) == bn)
        pos_l = logits_a[pos_mask]
        if len(pos_l) == 0:
            continue
        comb_l = np.concatenate([pos_l, neg_logits])
        comb_ll = np.concatenate([np.ones_like(pos_l), np.zeros_like(neg_logits)])
        bins[bn] = (comb_l, comb_ll)
    return bins
