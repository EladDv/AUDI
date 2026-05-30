"""Precision thresholds and calibration curves for the eval dashboard."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import streamlit as st
import torch

_PROJECT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_precision_thresholds() -> dict[str, dict[str, dict]]:
    """Return {sweep/model: {P_level: {sigma, cov, bg}}} from attack eval CSV."""
    csv_path = _PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
    if not csv_path.exists():
        return {}
    thresholds: dict[str, dict[str, dict]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ref = f"{row.get('sweep','')}/{row['model']}"
            if ref not in thresholds:
                thresholds[ref] = {}
            thresholds[ref][row["precision"]] = {
                "sigma": float(row["sigma"]),
                "cov_pct": float(row["cov_pct"]),
                "first_pct": float(row["first_pct"]),
                "bg": int(row["bg"]),
                "bg_alerts": row.get("bg_alerts", "-") or "-",
            }
    return thresholds


def compute_precision_recall_curve(pred_file: str) -> dict:
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
    p90_idx = int(np.argmin(np.abs(precisions - 0.90)))
    p95_idx = int(np.argmin(np.abs(precisions - 0.95)))
    p99_idx = int(np.argmin(np.abs(precisions - 0.99)))

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "sig_thresholds": sig_thresholds,
        "p90_idx": p90_idx,
        "p95_idx": p95_idx,
        "p99_idx": p99_idx,
    }
