"""Data building CLI for AUDI.

Usage:
    uv run audi-data <subcommand> [args]

Subcommands: hearability-templates, urban-esc, filtered-hf, chunk-spectro,
             pretrain-drones, dads-classify, analyze-snr, mel-stats,
             audioset-fp, merge-audioset-fp, field-bg, blue-red-recordings,
             mine-field-hard-negatives
"""

from __future__ import annotations

import sys

from scripts.cli.build.analyze_snr import run as run_analyze_snr
from scripts.cli.build.audioset_fp import run as run_audioset_fp
from scripts.cli.build.blue_red_recordings import main as run_blue_red_recordings
from scripts.cli.build.chunk_spectro import run as run_chunk_spectro
from scripts.cli.build.dads import run as run_dads
from scripts.cli.build.field_bg import main as run_field_bg
from scripts.cli.build.filtered_hf import run as run_filtered_hf
from scripts.cli.build.hearability import run as run_hearability
from scripts.cli.build.mel_stats import run as run_mel_stats
from scripts.cli.build.merge_audioset_fp import main as run_merge_audioset_fp
from scripts.cli.build.mine_field_hard_negatives import (
    main as run_mine_field_hard_negatives,
)
from scripts.cli.build.pretrain import run as run_pretrain
from scripts.cli.build.urban_esc import run as run_urban_esc

_SUBCMDS = {
    "hearability-templates": run_hearability,
    "urban-esc": run_urban_esc,
    "audioset-fp": run_audioset_fp,
    "merge-audioset-fp": run_merge_audioset_fp,
    "field-bg": run_field_bg,
    "blue-red-recordings": run_blue_red_recordings,
    "mine-field-hard-negatives": run_mine_field_hard_negatives,
    "filtered-hf": run_filtered_hf,
    "chunk-spectro": run_chunk_spectro,
    "pretrain-drones": run_pretrain,
    "dads-classify": run_dads,
    "analyze-snr": run_analyze_snr,
    "mel-stats": run_mel_stats,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _SUBCMDS:
        print("Usage: uv run audi-data <subcommand> [args]")
        print(f"Subcommands: {', '.join(_SUBCMDS)}")
        return 1
    subcmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    _SUBCMDS[subcmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
