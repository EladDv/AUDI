import csv

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from scripts.cli.train_blue_red import _rates_at_min_blue_recall, _rates_at_min_recall
from sweeps.sweep import extract_metrics, write_csv


def test_rates_at_min_recall_picks_strictest_threshold_above_target():
    scores = np.array([0.95, 0.80, 0.60, 0.40, 0.90, 0.55])
    labels = np.array([1, 1, 1, 1, 0, 0])

    rates = _rates_at_min_recall(scores, labels, 0.75)

    assert rates["threshold"] == pytest.approx(0.60)
    assert rates["tpr"] == pytest.approx(0.75)
    assert rates["fpr"] == pytest.approx(0.50)


def test_rates_at_min_blue_recall_picks_lowest_threshold_above_target():
    scores = np.array([0.95, 0.80, 0.60, 0.40, 0.90, 0.55, 0.20, 0.10])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    rates = _rates_at_min_blue_recall(scores, labels, 0.75)

    assert rates["threshold"] == pytest.approx(0.60)
    assert 1.0 - rates["fpr"] == pytest.approx(0.75)
    assert rates["tpr"] == pytest.approx(0.75)


def test_extract_metrics_includes_red_classification_thresholds(tmp_path):
    log_dir = tmp_path / "run" / "lightning_logs" / "version_0"
    writer = SummaryWriter(str(log_dir))
    writer.add_scalar("val/tpr_at_precision_90", 0.81, 1)
    writer.add_scalar("val_red_threshold_at_fnr_10", 0.40, 1)
    writer.add_scalar("val_red_threshold_at_red_recall_90", 0.41, 1)
    writer.add_scalar("val_red_threshold_at_red_recall_90", 0.42, 2)
    writer.add_scalar("val_red_threshold_at_blue_recall_90", 0.66, 2)
    writer.close()

    metrics = extract_metrics(tmp_path / "run")

    assert metrics["tpr_at_precision_90"] == pytest.approx(0.81)
    assert metrics["val_red_threshold_at_fnr_10"] == pytest.approx(0.40)
    assert metrics["val_red_threshold_at_red_recall_90"] == pytest.approx(0.42)
    assert metrics["val_red_threshold_at_blue_recall_90"] == pytest.approx(0.66)


def test_write_csv_preserves_metric_columns_when_first_row_failed(tmp_path):
    csv_path = tmp_path / "results.csv"

    write_csv(
        csv_path,
        [
            {"name": "first", "status": "failed"},
            {
                "name": "second",
                "status": "ok",
                "val_red_threshold_at_red_recall_90": 0.42,
                "val_red_threshold_at_blue_recall_90": 0.66,
            },
        ],
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert "val_red_threshold_at_red_recall_90" in rows[0]
    assert "val_red_threshold_at_blue_recall_90" in rows[0]
    assert rows[1]["val_red_threshold_at_red_recall_90"] == "0.42"
    assert rows[1]["val_red_threshold_at_blue_recall_90"] == "0.66"
