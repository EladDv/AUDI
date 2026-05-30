#!/usr/bin/env python3
"""Training CLI for AUDI drone detection.

Usage:
    uv run audi-train --arch cnn14 --epochs 30 --batch-size 32
    uv run audi-train detect --arch cnn14 --epochs 30 ...
    uv run audi-train classify --drone-path data/hf_dads_classify ...
"""

from __future__ import annotations

import sys

from scripts.cli.train_classify import run as run_classify
from scripts.cli.train_detect import run as run_detect

_SUBCMDS = {"detect": run_detect, "classify": run_classify}


def main() -> int:
    # Default to 'detect' for backward compat with sweeps calling with flat args
    if len(sys.argv) >= 2 and sys.argv[1] in _SUBCMDS:
        subcmd = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    else:
        subcmd = "detect"
    return _SUBCMDS[subcmd]()


if __name__ == "__main__":
    raise SystemExit(main())
