#!/usr/bin/env python3
"""Run the small-size current finetunes, then generate and run V4 size sweep."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PRETRAIN_SWEEP = PROJECT / "checkpoints" / "efficientat_20260528_232959"
V3_MN10_SWEEP = PROJECT / "checkpoints" / "field_hard_negative_finetune_v3_coverage_20260530_110102"
SMALL_V3_CONFIG = PROJECT / "sweeps" / "configs" / "field_hard_negative_finetune_v3_small_sizes.yaml"
GENERATED_V4_CONFIG = PROJECT / "sweeps" / "configs" / "field_hard_negative_finetune_v4_sizes.generated.yaml"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, env=env, check=True)


def newest_sweep(prefix: str) -> Path:
    matches = sorted(
        PROJECT.glob(f"checkpoints/{prefix}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No sweep found for prefix {prefix!r}")
    return matches[0]


def only_ckpt(run_dir: Path) -> Path:
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    if len(ckpts) > 1:
        ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    return ckpts[-1]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_v4_config(small_v3_sweep: Path) -> None:
    mn04 = only_ckpt(small_v3_sweep / "01_mn04_cov_hard10_freeze4_lr12e6")
    mn05 = only_ckpt(small_v3_sweep / "02_mn05_cov_hard10_freeze4_lr12e6")
    dymn04 = only_ckpt(small_v3_sweep / "03_dymn04_cov_hard10_freeze4_lr12e6")
    mn10 = only_ckpt(V3_MN10_SWEEP / "01_mn10_cov_hard10_freeze4_lr12e6")
    dymn10 = only_ckpt(V3_MN10_SWEEP / "02_dymn10_cov_hard10_freeze4_lr12e6")

    config = {
        "name": "field_hard_negative_finetune_v4_sizes",
        "noise_path": "data/HF_dataset_v7_background",
        "drone_path": "data/HF_dataset_v2_drone",
        "description": (
            "V4 coverage/FP-TP sweep over MN 0.4/0.5/1.0 plus supported dynamic "
            "MN 0.4/1.0. Starts from current hard-negative finetuned checkpoints, "
            "backs off hard-negative/noise pressure, and does not use 551 data."
        ),
        "base_flags": (
            "--clip-seconds 5.12 --patience 0 --epochs 14 "
            "--batch-size 24 --steps-per-epoch 250 --val-steps-per-epoch 120 "
            "--loss bce --label-smoothing 0.05 --augment "
            "--lr 1.0e-5 --lr-schedule linear --warmup-epochs 1 --weight-decay 0.03 "
            "--freeze-backbone-epochs 3 --positive-probability 0.62 "
            "--hard-noise data/field_recordings_20260514/mined_hard_negatives/hf_dataset "
            "--hard-noise-prob 0.07 "
            "--noise2 data/HF_dataset_v7_background --noise2-prob 0.45 "
            "--noise2-multi-prob 0.40 --noise2-count 3 --noise2-max-attenuation -38 "
            "--doppler-prob 0.35 --pitch-prob 0.25 --stretch-prob 0.25 "
            "--reverb-prob 0.25 --eq-prob 0.35 --noise-inject-prob 0.25 "
            "--noise-inject-db -40 --time-mask-prob 0.20 --lowpass-prob 0.20 "
            "--atmospheric-prob 0.25"
        ),
        "configs": [
            {
                "name": "01_mn04_v4_hard07_freeze3_lr10e6",
                "flags": f"--arch mn04_as --finetune-from {mn04.relative_to(PROJECT)}",
                "note": "V4 MN width 0.4 from current hard-negative finetune.",
            },
            {
                "name": "02_mn05_v4_hard07_freeze3_lr10e6",
                "flags": f"--arch mn05_as --finetune-from {mn05.relative_to(PROJECT)}",
                "note": "V4 MN width 0.5 from current hard-negative finetune.",
            },
            {
                "name": "03_mn10_v4_hard07_freeze3_lr10e6",
                "flags": f"--arch mn10_as --finetune-from {mn10.relative_to(PROJECT)}",
                "note": "V4 MN width 1.0 from current hard-negative finetune.",
            },
            {
                "name": "04_dymn04_v4_hard07_freeze3_lr10e6",
                "flags": f"--arch dymn04_as --finetune-from {dymn04.relative_to(PROJECT)}",
                "note": "V4 supported dynamic MN width 0.4.",
            },
            {
                "name": "05_dymn10_v4_hard07_freeze3_lr10e6",
                "flags": f"--arch dymn10_as --finetune-from {dymn10.relative_to(PROJECT)}",
                "note": "V4 supported dynamic MN width 1.0.",
            },
        ],
    }
    with open(GENERATED_V4_CONFIG, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"\nGenerated {GENERATED_V4_CONFIG.relative_to(PROJECT)}", flush=True)


def main() -> int:
    print(f"Started V4 size pipeline at {datetime.now().isoformat(timespec='seconds')}")
    required_pretrains = [
        PRETRAIN_SWEEP / "11_dymn04_as" / "checkpoints" / "epoch=23-step=6000.ckpt",
        PRETRAIN_SWEEP / "14_mn04_as" / "checkpoints" / "epoch=24-step=6250.ckpt",
        PRETRAIN_SWEEP / "15_mn05_as" / "checkpoints" / "epoch=23-step=6000.ckpt",
        PRETRAIN_SWEEP / "16_mn10_as" / "checkpoints" / "epoch=22-step=5750.ckpt",
        V3_MN10_SWEEP / "01_mn10_cov_hard10_freeze4_lr12e6" / "checkpoints",
        V3_MN10_SWEEP / "02_dymn10_cov_hard10_freeze4_lr12e6" / "checkpoints",
    ]
    for path in required_pretrains:
        require(path)

    run(["uv", "run", "python", "sweeps/sweep.py", str(SMALL_V3_CONFIG.relative_to(PROJECT))])
    small_v3_sweep = newest_sweep("field_hard_negative_finetune_v3_small_sizes")
    write_v4_config(small_v3_sweep)
    run(["uv", "run", "python", "sweeps/sweep.py", str(GENERATED_V4_CONFIG.relative_to(PROJECT))])
    print(f"Finished V4 size pipeline at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
