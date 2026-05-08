"""Precision thresholds and calibration curves for the eval dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

_PROJECT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_precision_thresholds() -> dict[str, dict[str, float]]:
    """Return {arch: {precision_level: sigmoid_threshold}} from Phase 3 CSV."""
    import csv
    import math
    csv_path = _PROJECT / "checkpoints_v2" / "phase3_full_results_20260507_070404.csv"
    if not csv_path.exists():
        return {}
    thresholds = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            arch = row.get("model_arch", "")
            pretrained = row.get("pretrained", "")
            tag = row.get("ablation_tag", "")
            if tag or pretrained != "True":
                continue
            if arch not in thresholds:
                p_levels = {}
                for pt in [80, 90, 95, 99]:
                    th = row.get(f"val/threshold_at_precision_{pt}")
                    tpr = row.get(f"val/tpr_at_precision_{pt}")
                    if th and tpr:
                        try:
                            p_levels[f"p{pt}"] = {
                                "sigmoid_threshold": 1.0 / (1.0 + math.exp(-float(th))),
                                "tpr": float(tpr),
                            }
                        except (ValueError, OverflowError):
                            pass
                if p_levels:
                    thresholds[arch] = p_levels
    return thresholds


def compute_precision_recall_curve(
    pred_file: str,
) -> dict:
    """Compute precision/recall vs threshold from predictions file.

    Returns dict with thresholds, precisions, recalls, sig_thresholds,
    and indices for key precision levels (P90, P95, P99).
    """
    pred_data = torch.load(pred_file, map_location="cpu", weights_only=False)
    val_logits = np.asarray(pred_data["logits"]).flatten()
    val_labels = np.asarray(pred_data["labels"]).flatten()

    thresholds = np.linspace(val_logits.min(), val_logits.max(), 200)
    precisions = []
    recalls = []
    for th in thresholds:
        preds = (val_logits > th).astype(int)
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    sig_thresholds = 1.0 / (1.0 + np.exp(-thresholds))
    p90_idx = np.argmin(np.abs(precisions - 0.90))
    p95_idx = np.argmin(np.abs(precisions - 0.95))
    p99_idx = np.argmin(np.abs(precisions - 0.99))

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "sig_thresholds": sig_thresholds,
        "p90_idx": p90_idx,
        "p95_idx": p95_idx,
        "p99_idx": p99_idx,
    }
