"""Schmitt-trigger hysteresis for detection smoothing."""

from __future__ import annotations

import math

import numpy as np


def apply_hysteresis(
    scores: np.ndarray,
    threshold: float,
    window: int = 8,
    ratio: float = 0.6,
    margin: float = 0.05,
) -> np.ndarray:
    """Schmitt-trigger hysteresis with moving-average confirmation.

    Uses asymmetric thresholds: turns ON only when >= ratio of the last
    ``window`` scores exceed ``threshold + margin``, and turns OFF only when
    >= ratio fall below ``threshold - margin``. State transitions require
    crossing the margin; staying in-state uses the relaxed side.

    Args:
        scores: 1D array of detection scores.
        threshold: Base decision threshold (sigma).
        window: Sliding window size for confirmation.
        ratio: Fraction of windows that must agree (default 0.6 -> 5 of 8).
        margin: Extra margin required to change state (default 0.05).

    Returns:
        Boolean array of same length as ``scores``.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=bool)

    lo = threshold - margin
    hi = threshold + margin

    state = False  # start OFF
    result = np.zeros(n, dtype=bool)
    for i in range(n):
        start = max(0, i - window + 1)
        recent = scores[start : i + 1]
        k = max(1, math.ceil(len(recent) * ratio))

        if state:
            # Currently ON — stay ON unless >= k recent scores drop below LO
            below = (recent < lo).sum()
            if below >= k:
                state = False
        else:
            # Currently OFF — turn ON if >= k recent scores exceed HI
            above = (recent > hi).sum()
            if above >= k:
                state = True
        result[i] = state

    return result
