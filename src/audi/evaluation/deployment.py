"""Shared helpers for deployment-oriented validation."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import numpy as np

from audi.config import MixConfig, parse_snr_bins

DEFAULT_SNR_BINS = [
    "easy:-5:0:0.25",
    "medium:-10:-5:0.30",
    "hard:-15:-10:0.30",
    "extreme:-20:-25:0.15",
]


def find_threshold_at_min_precision(
    precision: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target: float,
) -> tuple[float, float, float]:
    """Return threshold, actual precision, and TPR for best point above precision."""

    if len(precision) == 0:
        return float("nan"), float("nan"), float("nan")
    precision = np.asarray(precision, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    eligible = np.where(precision >= float(target))[0]
    if len(eligible) == 0:
        idx = int(np.argmax(precision))
    else:
        best_tpr = np.max(tpr[eligible])
        best = eligible[tpr[eligible] == best_tpr]
        idx = int(best[np.argmax(precision[best])])
    return float(thresholds[idx]), float(precision[idx]), float(tpr[idx])


def detection_precision_curve_points(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    precision_targets: list[float],
) -> list[dict[str, float]]:
    """Summarise a detection operating curve as precision-targeted TPR/FNR points."""

    from audi.training.validation import compute_precision, compute_roc_values

    _fpr, tpr, thresholds, _auc = compute_roc_values(logits, labels)
    precision = compute_precision(logits, labels, thresholds)
    points = []
    for target in precision_targets:
        threshold, actual_precision, actual_tpr = find_threshold_at_min_precision(
            precision, tpr, thresholds, target
        )
        points.append(
            {
                "target_precision": float(target),
                "threshold": threshold,
                "precision": actual_precision,
                "tpr": actual_tpr,
                "fnr": 1.0 - actual_tpr,
            }
        )
    return points


def deployment_score(
    *,
    classic_auc: float,
    field_mix_auc: float,
) -> float:
    """AUC-weighted selector for choosing the best checkpoint in a run."""

    return float(100.0 * (0.6 * classic_auc + 0.4 * field_mix_auc))

def mix_config_from_run(
    run_dir: Path,
    *,
    fallback_noise_path: Path | None = None,
    fallback_drone_path: Path | None = None,
    clip_seconds: float | None = None,
) -> tuple[MixConfig, list[str]]:
    """Build validation ``MixConfig`` from a run's saved sweep config."""

    cfg_path = find_sweep_config(run_dir)
    if cfg_path is None:
        if fallback_noise_path is None or fallback_drone_path is None:
            raise FileNotFoundError(f"No sweep_config.yaml found under {run_dir}")
        return _mix_config_from_mapping(
            {},
            fallback_noise_path=fallback_noise_path,
            fallback_drone_path=fallback_drone_path,
            clip_seconds=clip_seconds,
        )
    return mix_config_from_sweep_config(
        cfg_path,
        fallback_noise_path=fallback_noise_path,
        fallback_drone_path=fallback_drone_path,
        clip_seconds=clip_seconds,
    )


def find_sweep_config(run_dir: Path) -> Path | None:
    """Return the saved per-run sweep config path, if present."""

    candidates = [
        run_dir / "checkpoints" / "sweep_config.yaml",
        run_dir / "lightning_logs" / "version_0" / "checkpoints" / "sweep_config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(run_dir.rglob("sweep_config.yaml"))
    return matches[0] if matches else None


def mix_config_from_sweep_config(
    config_path: Path,
    *,
    fallback_noise_path: Path | None = None,
    fallback_drone_path: Path | None = None,
    clip_seconds: float | None = None,
) -> tuple[MixConfig, list[str]]:
    """Build validation ``MixConfig`` from a saved ``sweep_config.yaml``."""

    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return _mix_config_from_mapping(
        data,
        fallback_noise_path=fallback_noise_path,
        fallback_drone_path=fallback_drone_path,
        clip_seconds=clip_seconds,
    )


def _mix_config_from_mapping(
    data: dict[str, Any],
    *,
    fallback_noise_path: Path | None,
    fallback_drone_path: Path | None,
    clip_seconds: float | None,
) -> tuple[MixConfig, list[str]]:
    flags = _parse_flags(str(data.get("flags", "")))
    noise_path = _path_value(data.get("noise_path")) or fallback_noise_path
    drone_path = _path_value(data.get("drone_path")) or fallback_drone_path
    if noise_path is None or drone_path is None:
        raise ValueError("noise_path and drone_path are required to rebuild validation")

    snr_bin_values = flags.get("snr_bin") or DEFAULT_SNR_BINS
    snr_bins = parse_snr_bins(list(snr_bin_values))
    batch_size = int(_first(flags, "batch_size", 16))
    val_steps = int(_first(flags, "val_steps_per_epoch", 200))
    seconds = float(_first(flags, "clip_seconds", clip_seconds or 2.56))
    sample_rate = int(_first(flags, "sample_rate", 16000))

    # Match train_detect.py validation semantics: hard-noise is included, but
    # multi-noise and augmentation are deliberately disabled for validation.
    mix_cfg = MixConfig(
        noise_path=noise_path,
        drone_path=drone_path,
        hard_noise_path=_path_value(_first(flags, "hard_noise", None)),
        hard_noise_prob=float(_first(flags, "hard_noise_prob", 0.0)),
        noise2_path=None,
        noise2_prob=0.0,
        snr_bins=snr_bins,
        target_length_samples=int(sample_rate * seconds),
        positive_probability=0.5,
        highpass_hz=float(_first(flags, "highpass_hz", 125.0)),
        sample_rate=sample_rate,
        dataset_length=batch_size * val_steps,
        aug=None,
    )
    return mix_cfg, [b.name for b in snr_bins]


def _parse_flags(flags: str) -> dict[str, list[str] | bool]:
    tokens = shlex.split(flags)
    parsed: dict[str, list[str] | bool] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
            parsed[key] = True
            i += 1
            continue
        parsed.setdefault(key, [])
        values = parsed[key]
        if isinstance(values, list):
            values.append(tokens[i + 1])
        i += 2
    return parsed


def _first(
    flags: dict[str, list[str] | bool], key: str, default: object
) -> object:
    value = flags.get(key)
    if isinstance(value, list) and value:
        return value[-1]
    if isinstance(value, bool):
        return value
    return default


def _path_value(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))
