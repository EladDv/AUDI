"""Shared helpers for eval subcommands — no global constants."""

from __future__ import annotations


def require_paths(
    noise_path: str | None,
    drone_path: str | None,
) -> tuple[str, str]:
    """Validate both paths are provided. Raises SystemExit if not."""
    if noise_path is None or drone_path is None:
        print(
            "ERROR: --noise-path and --drone-path are required. Use one of:\n"
            "  --noise-path data/HF_dataset_v2_background \\\n"
            "  --drone-path data/HF_dataset_v2_drone"
        )
        raise SystemExit(1)
    return noise_path, drone_path
