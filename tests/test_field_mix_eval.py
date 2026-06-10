import numpy as np

from audi.config import SNRBin
from audi.evaluation.deployment import (
    deployment_score,
    detection_precision_curve_points,
    find_threshold_at_min_precision,
    mix_config_from_sweep_config,
)
from audi.evaluation.field_mix import FieldMixDataset


def _rows(value: float, count: int = 4, *, label: str = "x", label_id: int | None = None):
    rows = []
    for _ in range(count):
        row = {"audio": {"array": np.full(64, value, dtype=np.float32)}, "label": label}
        if label_id is not None:
            row["label_id"] = label_id
        rows.append(row)
    return rows


def test_field_mix_dataset_is_deterministic_and_balances_red_blue_hard_negatives():
    bg = _rows(0.2, count=5, label="no_drone")
    drones = _rows(0.5, count=3, label="blue", label_id=0) + _rows(
        0.8, count=3, label="red", label_id=1
    )
    hard = _rows(0.3, count=2, label="hard_fp")
    bins = [SNRBin("hard", -15.0, -10.0, 1.0)]

    ds1 = FieldMixDataset(
        background_ds=bg,
        drone_ds=drones,
        hard_negative_ds=hard,
        snr_bins=bins,
        target_length_samples=32,
        samples_per_color_bin=2,
        background_negatives=2,
        hard_negatives=2,
        seed=7,
    )
    ds2 = FieldMixDataset(
        background_ds=bg,
        drone_ds=drones,
        hard_negative_ds=hard,
        snr_bins=bins,
        target_length_samples=32,
        samples_per_color_bin=2,
        background_negatives=2,
        hard_negatives=2,
        seed=7,
    )

    assert ds1.metadata() == ds2.metadata()
    metadata = ds1.metadata()
    assert [r["color"] for r in metadata].count("blue") == 2
    assert [r["color"] for r in metadata].count("red") == 2
    assert [r["source"] for r in metadata].count("field_hard_negative") == 2


def test_mix_config_from_sweep_config_rebuilds_validation_inputs(tmp_path):
    cfg = tmp_path / "sweep_config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "noise_path: data/field_bg",
                "drone_path: data/drone",
                "flags: >-",
                "  --sample-rate 8000 --clip-seconds 5.12",
                "  --batch-size 24 --val-steps-per-epoch 120",
                "  --hard-noise data/hard --hard-noise-prob 0.07",
                "  --noise2 data/field_bg --noise2-prob 0.45",
                "  --snr-bin near:-8:-3:0.4 --snr-bin far:-25:-20:0.6",
            ]
        )
    )

    mix_cfg, bin_names = mix_config_from_sweep_config(cfg)

    assert mix_cfg.noise_path.as_posix() == "data/field_bg"
    assert mix_cfg.drone_path.as_posix() == "data/drone"
    assert mix_cfg.hard_noise_path.as_posix() == "data/hard"
    assert mix_cfg.hard_noise_prob == 0.07
    assert mix_cfg.noise2_path is None
    assert mix_cfg.dataset_length == 24 * 120
    assert mix_cfg.sample_rate == 8000
    assert mix_cfg.target_length_samples == 40960
    assert bin_names == ["near", "far"]


def test_deployment_score_is_weighted_auc_selector():
    score = deployment_score(classic_auc=0.80, field_mix_auc=0.50)

    assert score == 68.0


def test_find_threshold_at_min_precision_uses_best_tpr_above_target_precision():
    precision = np.array([0.5, 0.7, 0.85, 0.95, 1.0])
    tpr = np.array([1.0, 0.9, 0.8, 0.3, 0.0])
    thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    threshold, actual_precision, actual_tpr = find_threshold_at_min_precision(
        precision, tpr, thresholds, target=0.80
    )

    assert threshold == 0.5
    assert actual_precision == 0.85
    assert actual_tpr == 0.8


def test_detection_precision_curve_points_reports_tpr_and_fnr_by_precision_target():
    logits = np.array([4.0, 2.0, 0.0, -2.0])
    labels = np.array([1.0, 1.0, 0.0, 0.0])

    points = detection_precision_curve_points(
        logits, labels, precision_targets=[0.5, 0.9]
    )

    assert [p["target_precision"] for p in points] == [0.5, 0.9]
    assert set(points[0]) == {
        "target_precision",
        "threshold",
        "precision",
        "tpr",
        "fnr",
    }
