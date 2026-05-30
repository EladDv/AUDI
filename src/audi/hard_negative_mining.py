"""Utilities for mining field-recording false positives as hard negatives."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AlertRun:
    """A contiguous run of high model scores."""

    start_s: float
    end_s: float
    max_score: float
    mean_score: float
    n_windows: int


@dataclass(frozen=True)
class FieldRecording:
    """A raw field recording and its epoch-aligned start time if known."""

    path: Path
    start_epoch: float | None


def extract_alert_runs(
    scores: np.ndarray,
    times_s: np.ndarray,
    *,
    threshold: float,
    min_windows: int,
) -> list[AlertRun]:
    """Return contiguous above-threshold runs with at least ``min_windows``."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    times_s = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if len(scores) != len(times_s):
        raise ValueError("scores and times_s must have the same length")

    runs: list[AlertRun] = []
    start: int | None = None
    mask = scores >= float(threshold)
    for i, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            end = i
            if end - start >= min_windows:
                run_scores = scores[start:end]
                runs.append(
                    AlertRun(
                        start_s=round(float(times_s[start]), 6),
                        end_s=round(float(times_s[end - 1]), 6),
                        max_score=round(float(run_scores.max()), 6),
                        mean_score=round(float(run_scores.mean()), 6),
                        n_windows=int(end - start),
                    )
                )
            start = None
    return runs


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping ``(start, end)`` intervals."""
    cleaned = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def is_allowed(
    start_epoch: float,
    end_epoch: float,
    exclusions: Iterable[tuple[float, float]],
) -> bool:
    """Return False when ``[start_epoch, end_epoch]`` overlaps an exclusion."""
    for excl_start, excl_end in exclusions:
        if start_epoch < excl_end and end_epoch > excl_start:
            return False
    return True


def clip_with_padding(
    audio: np.ndarray,
    *,
    start_sample: int,
    length_samples: int,
) -> np.ndarray:
    """Slice ``length_samples`` from ``audio``, right-padding with zeros."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    start_sample = max(0, int(start_sample))
    end_sample = start_sample + int(length_samples)
    clip = audio[start_sample:end_sample]
    if len(clip) < length_samples:
        clip = np.pad(clip, (0, length_samples - len(clip)))
    return clip.astype(np.float32, copy=False)


def discover_field_recordings(field_dir: Path) -> list[FieldRecording]:
    """Find raw field recordings under ``field_dir/recordings`` only."""
    recording_dir = Path(field_dir) / "recordings"
    paths = sorted(
        p
        for p in recording_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
    )
    recordings: list[FieldRecording] = []
    for path in paths:
        try:
            sf.info(str(path))
        except sf.LibsndfileError:
            continue
        match = re.search(r"seg_(\d+(?:\.\d+)?)_", path.name)
        start_epoch = float(match.group(1)) if match else None
        recordings.append(FieldRecording(path=path, start_epoch=start_epoch))
    return recordings


def _load_labels(labels_csv: Path) -> dict[str, str]:
    if not labels_csv.exists():
        return {}
    with labels_csv.open() as f:
        return {r["alert_dir"]: r["label"] for r in csv.DictReader(f)}


def _load_segments(segments_json: Path) -> dict[str, list[tuple[float, float]]]:
    if not segments_json.exists():
        return {}
    raw = json.loads(segments_json.read_text())
    return {k: [(float(s), float(e)) for s, e in v] for k, v in raw.items()}


def build_alert_exclusions(
    field_dir: Path,
    *,
    buffer_s: float = 10.0,
    full_alert_duration_s: float = 120.0,
    exclude_unlabeled_alerts: bool = True,
) -> list[tuple[float, float]]:
    """Build absolute-time intervals that should not become hard negatives.

    ``nodrone`` alert files are intentionally not excluded; they are known
    false positives. Drone-labeled alerts are excluded by annotated drone
    segments when available, otherwise by the full pre/post alert window.
    Unlabeled alert windows are excluded by default to avoid training on
    unknown positives.
    """
    field_dir = Path(field_dir)
    labels = _load_labels(field_dir / "labels.csv")
    segments = _load_segments(field_dir / "segments.json")
    exclusions: list[tuple[float, float]] = []

    for metadata_path in sorted((field_dir / "alerts").glob("*/metadata.json")):
        alert_dir = metadata_path.parent.name
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        timestamp = float(metadata["timestamp"])
        full_start = timestamp - full_alert_duration_s / 2.0
        label = labels.get(alert_dir)

        if alert_dir in segments:
            for start_s, end_s in segments[alert_dir]:
                exclusions.append(
                    (
                        full_start + start_s - buffer_s,
                        full_start + end_s + buffer_s,
                    )
                )
        elif label == "drone" or (label is None and exclude_unlabeled_alerts):
            exclusions.append((full_start, full_start + full_alert_duration_s))

    return merge_intervals(exclusions)


def write_manifest(rows: list[dict], manifest_path: Path) -> None:
    """Write mined-clip metadata to CSV."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
