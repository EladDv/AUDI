#!/usr/bin/env python3
"""Inference CLI for AUDI.

Usage:
    uv run python scripts/inference.py <subcommand> [args]
    uv run audi-infer <subcommand> [args]

Subcommands: run, attackrun
"""

from __future__ import annotations

import sys

from scripts.cli.infer.attackrun import run as run_attackrun
from scripts.cli.infer.run import run as run_infer

_SUBCMDS = {
    "run": run_infer,
    "attackrun": run_attackrun,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _SUBCMDS:
        print("Usage: uv run python scripts/inference.py <subcommand> [args]")
        print(f"Subcommands: {', '.join(_SUBCMDS)}")
        return 1
    subcmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    _SUBCMDS[subcmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
