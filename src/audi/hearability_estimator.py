"""Bayesian SNR-bin estimator for drone detection logits.

Fits per-bin Gaussian distributions on positive-sample logits, then given
a raw detection logit, returns P(bin | logit) via Bayes rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BinStats:
    name: str
    mean: float
    std: float
    prior: float  # P(bin) from training distribution


class HearabilityEstimator:
    """Given a detection logit, estimate the hearability bin."""

    def __init__(self, bins: list[BinStats]):
        self.bins = bins
        self._names = [b.name for b in bins]

    @classmethod
    def from_predictions(cls, pred_path: Path) -> HearabilityEstimator:
        """Fit Gaussians from a predictions_{tag}.pt file."""
        data = torch.load(pred_path, map_location="cpu", weights_only=False)
        logits = np.asarray(data["logits"]).flatten()
        labels = np.asarray(data["labels"]).flatten()
        bin_names = np.asarray(data["bin_names"])

        pos_mask = labels > 0.5
        pos_logits = logits[pos_mask]
        pos_bins = bin_names[pos_mask]

        unique_bins = sorted(set(bn for bn in pos_bins if bn))
        if not unique_bins:
            raise ValueError("No positive samples with bin labels found")

        total_pos = len(pos_logits)
        bins = []
        for bn in unique_bins:
            mask = pos_bins == bn
            vals = pos_logits[mask]
            bins.append(
                BinStats(
                    name=bn,
                    mean=float(vals.mean()),
                    std=float(vals.std()) if len(vals) > 1 else 1.0,
                    prior=len(vals) / total_pos,
                )
            )
        return cls(bins)

    def predict(self, logit: float) -> dict[str, float]:
        """Return P(bin | logit) for each bin (sums to 1)."""
        log_probs = {}
        for b in self.bins:
            z = (logit - b.mean) / max(b.std, 1e-8)
            log_likelihood = -0.5 * z * z - np.log(max(b.std, 1e-8))
            log_probs[b.name] = log_likelihood + np.log(max(b.prior, 1e-12))

        log_probs_arr = np.array(list(log_probs.values()))
        log_probs_arr -= log_probs_arr.max()  # stabilize
        probs = np.exp(log_probs_arr)
        probs /= probs.sum()

        return dict(zip(self._names, probs, strict=False))

    def classify(self, logit: float) -> tuple[str, float]:
        """Return (best_bin_name, confidence)."""
        probs = self.predict(logit)
        best = max(probs, key=probs.get)
        return best, probs[best]

    def save(self, path: Path) -> None:
        """Save as compact .npz for deployment."""
        np.savez_compressed(
            path,
            names=np.array(self._names),
            means=np.array([b.mean for b in self.bins], dtype=np.float32),
            stds=np.array([b.std for b in self.bins], dtype=np.float32),
            priors=np.array([b.prior for b in self.bins], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> HearabilityEstimator:
        """Load from .npz file."""
        data = np.load(path)
        bins = [
            BinStats(str(n), float(m), float(s), float(p))
            for n, m, s, p in zip(
                data["names"], data["means"], data["stds"], data["priors"], strict=False
            )
        ]
        return cls(bins)
