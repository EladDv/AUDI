"""audioset-fp — Download AudioSet false-positive categories for drone detection.

Downloads only specific AudioSet categories that are most confusable with
drones (helicopters, motorcycles, engines, wind, lawn mowers, etc.).

Single-pass per shard: downloads one parquet shard, scans labels for matches,
extracts audio for matches inline, processes to 16kHz chunks. Saves each shard's
results immediately to disk — restart picks up where it left off.

Usage:
    uv run audi-data audioset-fp --max-per-category 500
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets
from tqdm import tqdm

# ── target AudioSet categories mapped to readable names ──────────────────
AUDIOSET_FP_CATEGORIES: dict[str, str] = {
    # Tier 1 — directly confusable with drone signatures
    "/m/09ct_": "helicopter",
    "/m/04_sv": "motorcycle",
    "/m/014yck": "aircraft_engine",
    "/m/02x984l": "mechanical_fan",
    "/m/07pjwq1": "buzz",
    "/m/01h3n": "bee_wasp",
    "/m/09f96": "mosquito",
    # Tier 2 — vehicles (specific types)
    "/m/0k4j": "car",
    "/m/07r04": "truck",
    "/m/01bjv": "bus",
    "/m/0ltv": "race_car",
    "/m/07q2z82": "accelerating_revving",
    "/m/07pb8fc": "idling",
    "/m/0btp2": "traffic_noise",
    "/m/0912c9": "vehicle_horn",
    # Tier 3 — trains
    "/m/07jdr": "train",
    "/m/06d_3": "rail_transport",
    # Tier 4 — aircraft (not drones, but similar domain)
    "/m/0cmf2": "airplane",
    "/m/04229": "jet_engine",
    "/m/02l6bg": "propeller",
    # Tier 5 — wind
    "/m/03m9d0z": "wind",
    # Tier 6 — motorized equipment (sustained tonal noise)
    "/m/01yg9g": "lawn_mower",
    "/m/01j4z9": "chainsaw",
    "/m/03p19w": "jackhammer",
    "/m/0d31p": "vacuum_cleaner",
    # Tier 7 — continuous tones
    "/m/07phhsh": "rumble",
    "/m/07rcgpl": "hum",
    # Tier 8 — sirens
    "/m/03kmc9": "siren",
    "/m/03j1ly": "emergency_vehicle",
}

_SR: int = 16000


def _category_for(labels: list[str]) -> str | None:
    """Return the first matching readable category name, or None."""
    for label in labels:
        if label in AUDIOSET_FP_CATEGORIES:
            return AUDIOSET_FP_CATEGORIES[label]
    return None


def _shard_dir(output_path: Path) -> Path:
    return output_path / "shards"


def _shard_path(output_path: Path, shard_idx: int) -> Path:
    return _shard_dir(output_path) / f"{shard_idx:03d}.parquet"


def _empty_path(output_path: Path, shard_idx: int) -> Path:
    return _shard_dir(output_path) / f"{shard_idx:03d}.empty"


def _completed_shards(output_path: Path) -> set[int]:
    """Return set of shard indices that already have saved output."""
    sd = _shard_dir(output_path)
    if not sd.exists():
        return set()
    completed = set()
    for fp in sd.glob("*.parquet"):
        try:
            completed.add(int(fp.stem))
        except ValueError:
            continue
    for fp in sd.glob("*.empty"):
        try:
            completed.add(int(fp.stem))
        except ValueError:
            continue
    return completed


def _load_progress(output_path: Path) -> dict[str, int]:
    """Load per-category counts from existing shard files."""
    sd = _shard_dir(output_path)
    counts: dict[str, int] = {}
    if not sd.exists():
        return counts
    for fp in sorted(sd.glob("*.parquet")):
        try:
            df = pl.read_parquet(fp)
            for cat in df["category"].unique().to_list():
                counts[cat] = (
                    counts.get(cat, 0)
                    + df.filter(pl.col("category") == cat).height
                )
        except Exception:
            continue
    return counts


def run() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Download AudioSet false-positive categories for drone detection"
    )
    ap.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/hf_background_audioset_fp"),
    )
    ap.add_argument("--max-per-category", type=int, default=500)
    ap.add_argument("--chunk-sec", type=float, default=10.0)
    ap.add_argument(
        "--split",
        type=str,
        default="unbalanced",
        choices=["unbalanced", "balanced", "full"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if (args.output_path / "dataset").exists() and not args.overwrite:
        print(
            f"Output dataset {args.output_path / 'dataset'} already exists. "
            "Use --overwrite to replace."
        )
        return

    if args.overwrite and args.output_path.exists():
        import shutil
        shutil.rmtree(args.output_path)
        print(f"Removed {args.output_path} for fresh start.")

    random.seed(args.seed)
    categories: list[str] = sorted(AUDIOSET_FP_CATEGORIES.values())
    name_to_id = {v: k for k, v in AUDIOSET_FP_CATEGORIES.items()}

    print(f"Target categories ({len(categories)}):")
    for cat in categories:
        print(f"  {cat}  ({name_to_id[cat]})")

    # ── Shard storage ────────────────────────────────────────────────────
    shard_dir = _shard_dir(args.output_path)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from existing shards ──────────────────────────────────────
    completed = _completed_shards(args.output_path)
    per_cat_counts = _load_progress(args.output_path)
    limits: dict[str, int] = {cat: args.max_per_category for cat in categories}

    if completed:
        total_existing = sum(per_cat_counts.values())
        remaining = sum(
            1 for c in categories if per_cat_counts.get(c, 0) < limits[c]
        )
        print(
            f"\nResuming: {len(completed)} shards completed, "
            f"{total_existing} clips collected"
        )
        print(
            f"║  {remaining} categories still below target ({args.max_per_category})"
            + " " * 22
            + "║"
        )
        print(f"╟{'─' * 68}╢")
        # Show all categories sorted by count
        for cat in sorted(
            categories, key=lambda c: per_cat_counts.get(c, 0), reverse=True
        ):
            n = per_cat_counts.get(cat, 0)
            pct = (
                n * 100 // args.max_per_category if args.max_per_category else 0
            )
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            done = " ✓" if n >= limits[cat] else f" ({limits[cat] - n} needed)"
            print(f"║  {cat:<24} {n:>4}/{limits[cat]:<4} {bar} {done:<12}║")
        print(f"╚{'═' * 68}╝")
        # Check if we're already done — but still need to merge
        if all(per_cat_counts.get(c, 0) >= limits[c] for c in categories):
            print("\nAll categories at quota — skipping shard scan, merging existing shards.")

    # ── Scan shards ──────────────────────────────────────────────────────
    split_prefix = {
        "unbalanced": "unbal_train",
        "balanced": "bal_train",
        "full": "full_train",
    }[args.split]
    base_url = "hf://datasets/agkphysics/AudioSet/data"
    chunk_n = int(_SR * args.chunk_sec)

    print(f"\n── Scanning {args.split} split, one shard at a time ──")
    shard_idx = 0
    MAX_SHARDS = 1000
    shards_processed = 0
    pbar = tqdm(desc="Scanning shards", unit=" shard")

    while shard_idx < MAX_SHARDS:
        # Check if all categories are full
        if all(per_cat_counts.get(c, 0) >= limits[c] for c in categories):
            break

        # Skip already-completed shards
        if shard_idx in completed:
            shard_idx += 1
            pbar.update(1)
            continue

        shard_url = f"{base_url}/{split_prefix}/{shard_idx:03d}.parquet"

        try:
            df = pl.scan_parquet(shard_url).collect()
        except Exception:
            shard_idx += 1
            pbar.update(1)
            continue

        # Find matching rows in this shard
        shard_matches: list[tuple[int, str]] = []
        labels_col = df["labels"].to_list()
        for i, labels in enumerate(labels_col):
            if all(per_cat_counts.get(c, 0) >= limits[c] for c in categories):
                break
            cat = _category_for(labels)
            if cat is None:
                continue
            if per_cat_counts.get(cat, 0) < limits[cat]:
                shard_matches.append((i, cat))
                per_cat_counts[cat] = per_cat_counts.get(cat, 0) + 1

        # Extract audio and save immediately to shard file
        if shard_matches:
            audio_col = df["audio"].to_list()
            records: list[dict] = []
            for local_idx, cat in shard_matches:
                audio_struct = audio_col[local_idx]
                flac_bytes = audio_struct["bytes"]
                arr, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)

                if sr != _SR:
                    from scipy.signal import resample

                    arr = resample(arr, int(len(arr) * _SR / sr)).astype(
                        np.float32
                    )

                for start in range(0, len(arr), chunk_n):
                    chunk = arr[start : start + chunk_n]
                    if len(chunk) < chunk_n // 2:
                        continue
                    if len(chunk) < chunk_n:
                        chunk = np.pad(chunk, (0, chunk_n - len(chunk)))
                    records.append(
                        {
                            "audio_array": chunk.tobytes(),
                            "category": cat,
                            "source": "audioset",
                        }
                    )

            if records:
                # Save as parquet with raw audio bytes (compact)
                shard_df = pl.DataFrame(records)
                sp = _shard_path(args.output_path, shard_idx)
                shard_df.write_parquet(sp)
                completed.add(shard_idx)
                shards_processed += 1
            else:
                # Mark empty shard so we don't re-download on resume
                _empty_path(args.output_path, shard_idx).touch()
                completed.add(shard_idx)

        shard_idx += 1
        pbar.update(1)

    pbar.close()
    print(
        f"\nProcessed {shards_processed} new shards (total completed: {len(completed)})"
    )

    # ── Report ───────────────────────────────────────────────────────────
    total = sum(per_cat_counts.values())
    print(f"\nCollected {total} clips across {len(categories)} categories:")
    for cat in categories:
        n = per_cat_counts.get(cat, 0)
        bar = "█" * min(n * 20 // max(args.max_per_category, 1), 20)
        print(f"  {cat:<25} {n:>4}/{args.max_per_category}  {bar}")

    shortfall = [
        c
        for c in categories
        if per_cat_counts.get(c, 0) < args.max_per_category
    ]
    if shortfall:
        print(
            f"\n⚠ {len(shortfall)} categories below target: {', '.join(shortfall)}"
        )

    # ── Merge all shard files into final DatasetDict ─────────────────────
    print("\n── Merging shards into final dataset ──")
    features = Features(
        {
            "audio": Audio(sampling_rate=_SR),
            "category": Value("string"),
            "source_dataset": Value("string"),
        }
    )

    # Collect all records in batches to avoid OOM
    all_records: list[dict] = []
    batch_size = 500
    datasets: list[Dataset] = []

    for sp in sorted(shard_dir.glob("*.parquet")):
        df = pl.read_parquet(sp)
        for row in df.iter_rows(named=True):
            arr = np.frombuffer(row["audio_array"], dtype=np.float32)
            all_records.append(
                {
                    "audio": {"array": arr, "sampling_rate": _SR},
                    "category": row["category"],
                    "source_dataset": "audioset_fp",
                }
            )
            if len(all_records) >= batch_size:
                datasets.append(Dataset.from_list(all_records, features=features))
                all_records = []

    if all_records:
        datasets.append(Dataset.from_list(all_records, features=features))

    # Concatenate all batches
    full_ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    n_total = len(full_ds)
    n_val = max(1, int(n_total * 0.10))
    n_test = max(1, int(n_total * 0.10))
    n_train = n_total - n_val - n_test

    # Shuffle and split
    full_ds = full_ds.shuffle(seed=args.seed)
    train_ds = full_ds.select(range(n_train))
    val_ds = full_ds.select(range(n_train, n_train + n_val))
    test_ds = full_ds.select(range(n_train + n_val, n_total))

    dd = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # Save final dataset
    final_path = args.output_path / "dataset"
    final_path.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(final_path))
    print(f"Final dataset saved to {final_path}")

    # Save progress manifest
    manifest = {
        "categories": {cat: per_cat_counts.get(cat, 0) for cat in categories},
        "shards_completed": sorted(completed),
        "total_clips": n_total,
    }
    with open(args.output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {args.output_path / 'manifest.json'}")
    print("Done.")
