"""Shared helpers for thin console-script dispatchers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path

CommandSpec = tuple[str, str]

NUM_WORKERS: int = 4

DATA_SUBCOMMANDS: dict[str, CommandSpec] = {
    "field-bg": ("scripts.cli.build.field_bg", "main"),
    "blue-red-recordings": ("scripts.cli.build.blue_red_recordings", "main"),
    "mine-field-hard-negatives": ("scripts.cli.build.mine_field_hard_negatives", "main"),
    "pyroom-dataset": ("scripts.cli.build.pyroom_dataset", "run"),
    "pyroom-mvdr-cache": ("scripts.cli.build.pyroom_dataset", "run_cache"),
    "precompute-waveforms": ("scripts.cli.build.precompute", "run_waveforms"),
    "precompute-features": ("scripts.cli.build.precompute", "run_features"),
}

EVAL_SUBCOMMANDS: dict[str, CommandSpec] = {
    "attack-runs": ("scripts.cli.eval.attack_runs", "run"),
    "field": ("scripts.cli.eval.field", "run"),
    "postprocess": ("scripts.cli.eval.postprocess", "run"),
    "calibrate": ("scripts.cli.eval.calibrate", "run"),
    "doa-compare": ("scripts.cli.eval.doa_compare", "run"),
    "pyroom-attack-sim": ("scripts.cli.eval.pyroom_attack_sim", "run"),
}


def print_usage(usage: str, subcommands: Mapping[str, CommandSpec]) -> None:
    print(usage)
    print(f"Subcommands: {', '.join(subcommands)}")


def load_subcommand(
    subcommands: Mapping[str, CommandSpec],
    name: str,
) -> Callable:
    module_name, attr = subcommands[name]
    return getattr(import_module(module_name), attr)


def parse_global_dataset_args(
    argv: list[str],
) -> tuple[Path | None, Path | None, list[str]]:
    """Extract --noise-path and --drone-path from argv."""
    noise_path = drone_path = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--noise-path" and i + 1 < len(argv):
            noise_path = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--drone-path" and i + 1 < len(argv):
            drone_path = Path(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return noise_path, drone_path, rest


def configure_torch_file_sharing() -> None:
    """Use /tmp-backed sharing for DataLoaders on machines with small /dev/shm."""
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")


def data_main() -> int:
    """Console entrypoint for dataset-building subcommands."""
    usage = "Usage: uv run audi-data <subcommand> [args]"
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_usage(usage, DATA_SUBCOMMANDS)
        return 0
    if sys.argv[1] not in DATA_SUBCOMMANDS:
        print_usage(usage, DATA_SUBCOMMANDS)
        return 1
    subcmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    load_subcommand(DATA_SUBCOMMANDS, subcmd)()
    return 0


def eval_main() -> int:
    """Console entrypoint for evaluation subcommands."""
    usage = (
        "Usage: uv run audi-eval "
        "[--noise-path <path>] [--drone-path <path>] <subcommand> [args]"
    )
    noise_path, drone_path, rest = parse_global_dataset_args(sys.argv[1:])
    if not rest or rest[0] in {"-h", "--help"}:
        print_usage(usage, EVAL_SUBCOMMANDS)
        return 0
    if rest[0] not in EVAL_SUBCOMMANDS:
        print_usage(usage, EVAL_SUBCOMMANDS)
        return 1
    subcmd = rest[0]
    sys.argv = [sys.argv[0]] + rest[1:]
    load_subcommand(EVAL_SUBCOMMANDS, subcmd)(noise_path, drone_path)
    return 0
