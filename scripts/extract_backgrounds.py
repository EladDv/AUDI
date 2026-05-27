#!/usr/bin/env python3
"""Extract background audio from field recording tags.

- nodrone files → copied whole to backgrounds/
- drone files with segments → split into non-drone regions with 10s buffer
  on each side of each expanded segment, keeping only splits > 10s
- drone files without segments → skipped (warned)
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/field_recordings_20260514"
LABELS_CSV = DATA / "labels.csv"
SEGMENTS_JSON = DATA / "segments.json"
ALERTS_DIR = DATA / "alerts"
OUT_DIR = DATA / "backgrounds"
BUFFER_S = 10.0
MIN_SPLIT_S = 10.0
DURATION_S = 120.0
SR = 16000


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    if LABELS_CSV.exists():
        reader = csv.reader(LABELS_CSV.read_text().strip().splitlines())
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                labels[row[0]] = row[1]
    return labels


def load_segments() -> dict[str, list[tuple[float, float]]]:
    if SEGMENTS_JSON.exists():
        raw = json.loads(SEGMENTS_JSON.read_text())
        return {k: sorted(tuple(s) for s in v) for k, v in raw.items()}
    return {}


def complement_regions(
    segments: list[tuple[float, float]],
    total_duration: float,
    buffer: float,
) -> list[tuple[float, float]]:
    """Return time regions OUTSIDE expanded [s-buffer, e+buffer] intervals."""
    # Expand and clamp
    excluded = sorted(
        (max(0.0, s - buffer), min(total_duration, e + buffer))
        for s, e in segments
    )
    # Merge overlapping excluded intervals
    merged: list[tuple[float, float]] = []
    for s, e in excluded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # Complement
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            regions.append((cursor, s))
        cursor = e
    if cursor < total_duration:
        regions.append((cursor, total_duration))
    return regions


def main():
    labels = load_labels()
    segments_data = load_segments()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodrone_total = 0
    drone_split_total = 0
    drone_no_segments: list[str] = []
    skipped_short = 0

    for alert_dir, label in sorted(labels.items()):
        wav_path = ALERTS_DIR / alert_dir / "full_120s.wav"
        if not wav_path.exists():
            print(f"  SKIP {alert_dir}: audio not found")
            continue

        if label == "nodrone":
            dst = OUT_DIR / f"{alert_dir}.wav"
            shutil.copy2(wav_path, dst)
            nodrone_total += 1
            print(f"  COPY {alert_dir}.wav (nodrone)")
            continue

        # drone
        segs = segments_data.get(alert_dir, [])
        if not segs:
            drone_no_segments.append(alert_dir)
            continue

        audio, _sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        regions = complement_regions(segs, DURATION_S, BUFFER_S)

        for i, (start_s, end_s) in enumerate(regions):
            dur = end_s - start_s
            if dur < MIN_SPLIT_S:
                skipped_short += 1
                print(f"  SKIP {alert_dir}[{i}]: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s < {MIN_SPLIT_S}s)")
                continue

            start_sample = int(start_s * SR)
            end_sample = int(end_s * SR)
            chunk = audio[start_sample:end_sample]

            dst = OUT_DIR / f"{alert_dir}_{i:02d}.wav"
            sf.write(str(dst), chunk, SR, subtype="PCM_16")
            drone_split_total += 1
            print(f"  SPLIT {dst.name}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)")

    print()
    print(f"Done: {nodrone_total} nodrone copies, {drone_split_total} drone splits")
    if drone_no_segments:
        print(f"Drone files without segments (skipped): {len(drone_no_segments)}")
        for d in sorted(drone_no_segments):
            print(f"  {d}")
    if skipped_short:
        print(f"Skipped {skipped_short} splits shorter than {MIN_SPLIT_S}s")


if __name__ == "__main__":
    main()
