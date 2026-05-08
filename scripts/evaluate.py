#!/usr/bin/env python3
"""Evaluation CLI for AUDI.

Usage:
    uv run python scripts/evaluate.py [--noise-path <path>] [--drone-path <path>] <subcommand> [args]
    uv run audi-eval [--noise-path <path>] [--drone-path <path>] <subcommand> [args]

Subcommands:
    postprocess, calibrate, fpr-thresholds, fpr-multi, operational, attack-runs, ensemble

Flags (attack-runs only):
    --all                 Force reprocessing of all checkpoints (skip nothing)
    --skip-postprocess    Skip auto-postprocessing of new checkpoints
    --skip-calibrate      Skip auto-calibration of new checkpoints
"""

from __future__ import annotations

import sys

from audi.cli_utils import parse_global_dataset_args
from scripts.cli.eval.attack_runs import run as run_attack_runs
from scripts.cli.eval.calibrate import run as run_calibrate
from scripts.cli.eval.ensemble import run as run_ensemble
from scripts.cli.eval.fpr import run as run_fpr_thresholds
from scripts.cli.eval.fpr_multi import run_multi as run_fpr_multi
from scripts.cli.eval.operational import run as run_operational
from scripts.cli.eval.postprocess import run as run_postprocess

_SUBCMDS = {
    "postprocess": run_postprocess,
    "calibrate": run_calibrate,
    "fpr-thresholds": run_fpr_thresholds,
    "fpr-multi": run_fpr_multi,
    "operational": run_operational,
    "attack-runs": run_attack_runs,
    "ensemble": run_ensemble,
}


def main() -> int:
    noise_path, drone_path, rest = parse_global_dataset_args(sys.argv[1:])
    if not rest or rest[0] not in _SUBCMDS:
        print(
            "Usage: uv run python scripts/evaluate.py "
            "[--noise-path <path>] [--drone-path <path>] <subcommand> [args]"
        )
        print(f"Subcommands: {', '.join(_SUBCMDS)}")
        return 1
    subcmd = rest[0]
    sys.argv = [sys.argv[0]] + rest[1:]
    _SUBCMDS[subcmd](noise_path, drone_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
