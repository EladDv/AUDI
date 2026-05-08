"""Shared argparse utilities for CLI scripts."""

from __future__ import annotations

from pathlib import Path

import torch

# Avoid PyTorch DataLoader shared-memory exhaustion with many workers.
# file_system uses /tmp instead of /dev/shm — trades a bit of speed for
# robustness on systems with limited shm or many concurrent DataLoaders.
torch.multiprocessing.set_sharing_strategy("file_system")

NUM_WORKERS: int = 4


def parse_global_dataset_args(
    argv: list[str],
) -> tuple[Path | None, Path | None, list[str]]:
    """Extract --noise-path and --drone-path from argv.

    Returns (noise_path, drone_path, remaining_args).
    Used by evaluate.py and scripts that need global dataset paths before
    subcommand dispatch.
    """
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


def require_paths(
    noise_path: Path | None,
    drone_path: Path | None,
) -> tuple[Path, Path]:
    """Validate that both dataset paths are provided. Raises SystemExit if not."""
    if noise_path is None or drone_path is None:
        print(
            "ERROR: --noise-path and --drone-path are required. Use one of:\n"
            "  --noise-path data/HF_dataset_v2_background \\\n"
            "  --drone-path data/HF_dataset_v2_drone"
        )
        raise SystemExit(1)
    return noise_path, drone_path
