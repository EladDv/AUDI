#!/usr/bin/env python3
"""Extract clean background: merge all 5-min segments into one 24h audio,
zero out ±5 min around each alert, then split at consecutive zeros."""

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/home/elad/projects/AUDI")
DATA = ROOT / "data/field_recordings_20260514"
ALERTS_JSON = DATA / "wd003_scores.json"
SEGMENTS_DIR = DATA / "recordings"
OUT_DIR = DATA / "clean_background"

CONTAM_S = 300  # ±5 min
SR = 16000
SEGMENT_DUR = 300  # each segment is 300s


def parse_alert_timestamps() -> list[int]:
    scores = json.loads(ALERTS_JSON.read_text())
    return sorted({int(a["alert_dir"].split("_")[1]) for a in scores["alerts"]})


def parse_segments() -> list[tuple[int, Path]]:
    """(start_ts, path) sorted by timestamp."""
    segs = []
    for f in sorted(SEGMENTS_DIR.glob("seg_*.flac")):
        m = re.match(r"seg_(\d+)_\d+\.flac", f.name)
        if m:
            segs.append((int(m.group(1)), f))
    return segs


def main():
    alert_tss = parse_alert_timestamps()
    segments = parse_segments()
    print(f"Alerts: {len(alert_tss)}")
    print(f"Segments: {len(segments)}")

    t0 = segments[0][0]  # global start time
    t_end = segments[-1][0] + SEGMENT_DUR
    total_dur = t_end - t0
    print(f"Total span: {total_dur / 3600:.1f}h")
    # Mark each segment clean/contaminated
    seg_status = []
    for seg_ts, seg_path in segments:
        seg_end = seg_ts + SEGMENT_DUR
        contaminated = False
        for ats in alert_tss:
            if ats - CONTAM_S < seg_end and ats + CONTAM_S > seg_ts:
                contaminated = True
                break
        seg_status.append(not contaminated)

    # Find runs of consecutive clean segments
    runs = []
    i = 0
    while i < len(seg_status):
        if seg_status[i]:
            j = i
            while j < len(seg_status) and seg_status[j]:
                j += 1
            dur = (j - i) * SEGMENT_DUR
            if dur >= 10:
                runs.append((i, j))
            i = j
        else:
            i += 1

    print(f"\nClean runs ≥ 10s: {len(runs)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for run_idx, (rs, re) in enumerate(runs):
        chunks = []
        for i in range(rs, re):
            seg_ts, seg_path = segments[i]
            if seg_path.stat().st_size == 0:
                continue
            audio, sr = sf.read(str(seg_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            chunks.append(audio.astype(np.float32))

        concat = np.concatenate(chunks)
        dur = len(concat) / SR
        abs_start = segments[rs][0]
        out_path = OUT_DIR / f"bg_{run_idx:03d}_{dur:.0f}s_{abs_start}.wav"
        sf.write(str(out_path), concat, SR)
        print(f"  → {out_path.name}  ({dur/60:.1f}min)")
        del chunks, concat

    total_clean = sum(
        (re - rs) * SEGMENT_DUR for rs, re in runs
    )
    print(f"\nDone. {len(runs)} background runs, {total_clean/3600:.1f}h total in {OUT_DIR}")


if __name__ == "__main__":
    main()
