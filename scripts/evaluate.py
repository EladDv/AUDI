#!/usr/bin/env python3
"""Evaluation CLI for AUDI.

Usage:
    uv run python scripts/evaluate.py [global flags] <subcommand> [args]
    uv run audi-eval [global flags] <subcommand> [args]

Subcommands:
    attack-runs, field, postprocess, calibrate

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
from scripts.cli.eval.field import run as run_field
from scripts.cli.eval.postprocess import run as run_postprocess

_SUBCMDS = {
    "attack-runs": run_attack_runs,
    "field": run_field,
    "postprocess": run_postprocess,
    "calibrate": run_calibrate,
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
