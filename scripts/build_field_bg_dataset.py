#!/usr/bin/env python3
"""Build HF_dataset_v7_background from field recording backgrounds.

Streaming approach: processes one file at a time, writes parquet shards
incrementally, then concatenates into a DatasetDict. Avoids loading all
audio into memory at once.

Source: data/field_recordings_20260514/backgrounds/
Output: data/HF_dataset_v7_background (HF DatasetDict with train/validation)
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, concatenate_datasets


def extract_windows(
    audio: np.ndarray, sr: int, clip_s: float, stride_s: float
) -> list[np.ndarray]:
    """Extract sliding windows from audio."""
    win_samples = int(sr * clip_s)
    stride_samples = int(sr * stride_s)
    windows = []
    for start in range(0, len(audio) - win_samples + 1, stride_samples):
        windows.append(audio[start : start + win_samples].copy())
    return windows


def build_shard(records: list[dict], shard_dir: Path, shard_idx: int) -> None:
    """Save a shard of records as a parquet file."""
    shard_path = shard_dir / f"shard_{shard_idx:04d}"
    ds = Dataset.from_list(records)
    ds.save_to_disk(str(shard_path))


def main():
    parser = argparse.ArgumentParser(description="Build field background dataset")
    parser.add_argument(
        "--bg-dir",
        default="data/field_recordings_20260514/backgrounds",
        help="Directory containing background WAV files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/HF_dataset_v7_background",
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--clip-seconds", type=float, default=30.0, help="Clip duration in seconds"
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=2.0,
        help="Stride between windows in seconds (smaller = more oversampling)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Validation split fraction"
    )
    parser.add_argument(
        "--shard-size", type=int, default=200, help="Number of clips per parquet shard"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tmp-dir",
        default="data/HF_dataset_v7_background_tmp",
        help="Temporary directory for shards",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_dir = Path(args.bg_dir)
    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)

    wav_files = sorted(bg_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} background WAV files")

    # Clean temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    train_shards = tmp_dir / "train"
    val_shards = tmp_dir / "val"
    train_shards.mkdir(parents=True)
    val_shards.mkdir(parents=True)

    # First pass: count total windows to determine train/val split point
    # (we only need counts, not the actual audio)
    print("Counting windows (dry run)...")
    file_windows = {}  # path -> window count
    total_dur = 0.0
    for wf in wav_files:
        info = sf.info(str(wf))
        total_dur += info.duration
        n_windows = max(0, int((info.duration - args.clip_seconds) / args.stride_seconds) + 1)
        if n_windows > 0:
            file_windows[str(wf)] = n_windows

    total_windows = sum(file_windows.values())
    n_val = max(1, int(total_windows * args.val_split))
    n_train = total_windows - n_val

    print(f"Total source audio: {total_dur / 60:.1f} min")
    print(f"Total windows: {total_windows} ({args.clip_seconds}s each, stride={args.stride_seconds}s)")
    print(f"Train: {n_train} ({n_train * args.clip_seconds / 3600:.1f} hours)")
    print(f"Val:   {n_val} ({n_val * args.clip_seconds / 3600:.1f} hours)")

    # Shuffle windows at the granularity of windows (not files)
    # Build a flat list of (file, window_index) pairs, shuffle, then assign to train/val
    all_indices = []
    for wf_path, count in file_windows.items():
        for wi in range(count):
            all_indices.append((wf_path, wi))
    random.shuffle(all_indices)

    val_indices = set(all_indices[:n_val])  # set for O(1) lookup
    # Rest are train

    # Second pass: stream through files, build shards
    train_records = []
    val_records = []
    train_shard_idx = 0
    val_shard_idx = 0
    processed = 0

    for wf in wav_files:
        wf_str = str(wf)
        n_windows = file_windows.get(wf_str, 0)
        if n_windows == 0:
            continue

        audio, sr = sf.read(wf_str)
        assert sr == 16000, f"Expected 16kHz, got {sr} for {wf.name}"
        windows = extract_windows(audio, sr, args.clip_seconds, args.stride_seconds)

        for wi, w in enumerate(windows):
            key = (wf_str, wi)
            record = {
                "audio": {
                    "array": w.astype(np.float32),
                    "sampling_rate": 16000,
                },
                "label": 0,
            }

            if key in val_indices:
                val_records.append(record)
                if len(val_records) >= args.shard_size:
                    build_shard(val_records, val_shards, val_shard_idx)
                    val_shard_idx += 1
                    val_records = []
            else:
                train_records.append(record)
                if len(train_records) >= args.shard_size:
                    build_shard(train_records, train_shards, train_shard_idx)
                    train_shard_idx += 1
                    train_records = []

            processed += 1
            if processed % 5000 == 0:
                print(f"  Processed {processed}/{total_windows} windows...")

        # Free per-file memory
        del audio, windows
        gc.collect()

    # Flush remaining records
    if train_records:
        build_shard(train_records, train_shards, train_shard_idx)
        train_shard_idx += 1
    if val_records:
        build_shard(val_records, val_shards, val_shard_idx)
        val_shard_idx += 1

    print(f"Wrote {train_shard_idx} train shards, {val_shard_idx} val shards")

    # Concatenate shards into final datasets
    print("Concatenating train shards...")
    train_datasets = []
    for i in range(train_shard_idx):
        train_datasets.append(
            Dataset.load_from_disk(str(train_shards / f"shard_{i:04d}"))
        )
    train_ds = concatenate_datasets(train_datasets) if train_datasets else Dataset.from_list([])

    print("Concatenating val shards...")
    val_datasets = []
    for i in range(val_shard_idx):
        val_datasets.append(
            Dataset.load_from_disk(str(val_shards / f"shard_{i:04d}"))
        )
    val_ds = concatenate_datasets(val_datasets) if val_datasets else Dataset.from_list([])

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    print(f"Saving to {output_dir}...")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    ds_dict.save_to_disk(str(output_dir))

    # Cleanup
    print("Cleaning up temp shards...")
    shutil.rmtree(tmp_dir)

    print(f"Done! {len(ds_dict['train'])} train + {len(ds_dict['validation'])} val clips")
    print(
        f"Total: {len(ds_dict['train']) + len(ds_dict['validation'])} clips "
        f"({(len(ds_dict['train']) + len(ds_dict['validation'])) * args.clip_seconds / 3600:.1f} hours)"
    )


if __name__ == "__main__":
    main()
