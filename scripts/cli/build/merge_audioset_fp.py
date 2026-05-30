#!/usr/bin/env python3
"""Merge audioset_fp (hf_background_audioset_fp) into the sharded_expanded dataset.

Produces: data/hf_dataset_sharded_expanded_merged/
         = sharded_expanded + audioset_fp (category-mapped, feature-harmonized)

Usage:
    uv run audi-data merge-audioset-fp [--overwrite]
"""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIOSET_FP_DIR = PROJECT_ROOT / "data" / "hf_background_audioset_fp" / "dataset"
SHARDED_EXPANDED_DIR = PROJECT_ROOT / "data" / "hf_dataset_sharded_expanded"
OUTPUT_DIR = PROJECT_ROOT / "data" / "hf_dataset_sharded_expanded_merged"

# Map audioset_fp fine-grained categories -> sharded_expanded broad categories
CATEGORY_MAP: dict[str, str] = {
    # --- cars ---
    "car": "cars",
    "truck": "cars",
    "bus": "cars",
    "motorcycle": "cars",
    "race_car": "cars",
    "accelerating_revving": "cars",
    "accelerating, revving, vroom": "cars",
    "idling": "cars",
    "traffic_noise": "cars",
    "traffic noise, roadway noise": "cars",
    "rail_transport": "cars",
    "train": "cars",
    "vehicle_horn": "cars",
    "vehicle horn, car horn, honking": "cars",
    # --- mechanical ---
    "chainsaw": "mechanical",
    "lawn_mower": "mechanical",
    "vacuum_cleaner": "mechanical",
    "mechanical_fan": "mechanical",
    "hair_dryer": "mechanical",
    "jackhammer": "mechanical",
    "hum": "mechanical",
    "whir": "mechanical",
    "buzz": "mechanical",
    "rumble": "mechanical",
    "jet_engine": "mechanical",
    "aircraft_engine": "mechanical",
    "aircraft": "mechanical",
    "airplane": "mechanical",
    "helicopter": "mechanical",
    "propeller": "mechanical",
    "propeller, airscrew": "mechanical",
    # --- urban ---
    "emergency_vehicle": "urban",
    "siren": "urban",
    # --- wind ---
    "wind": "wind",
    # --- environment (nature/animals) ---
    "bee_wasp": "environment",
    "bee, wasp, etc.": "environment",
    "mosquito": "environment",
}


def _map_category(category: str) -> str:
    """Map audioset_fp category to broad category, with fallback."""
    mapped = CATEGORY_MAP.get(category)
    if mapped is not None:
        return mapped
    # Fuzzy fallback for categories we haven't explicitly mapped
    print(f"  [WARN] No mapping for category '{category}', defaulting to 'mechanical'")
    return "mechanical"


def merge_split(fp_ds: Dataset, exp_ds: Dataset) -> Dataset:
    """Merge a single split from audioset_fp into sharded_expanded."""
    # 1. Remap categories
    new_categories = [_map_category(c) for c in fp_ds["category"]]

    # 2. Build harmonized audioset_fp table — audioset_fp already has source_dataset
    fp_mapped = fp_ds.remove_columns(["category"]).add_column(
        "category", new_categories
    )

    # 3. Ensure sharded_expanded has source_dataset (it does, but be safe)
    if "source_dataset" not in exp_ds.features:
        exp_ds = exp_ds.add_column(
            "source_dataset",
            ["expanded_noise_v1"] * len(exp_ds),
        )

    # 4. Concatenate
    return concatenate_datasets([exp_ds, fp_mapped])


def main():
    parser = argparse.ArgumentParser(description="Merge audioset_fp into sharded_expanded")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    if OUTPUT_DIR.exists() and not args.overwrite:
        print(f"Output already exists: {OUTPUT_DIR}")
        print("Use --overwrite to replace.")
        return

    print(f"Loading audioset_fp from {AUDIOSET_FP_DIR}")
    fp = load_from_disk(str(AUDIOSET_FP_DIR))

    print(f"Loading sharded_expanded from {SHARDED_EXPANDED_DIR}")
    expanded = load_from_disk(str(SHARDED_EXPANDED_DIR))

    merged = DatasetDict()
    for split in ["train", "validation", "test"]:
        fp_n = len(fp[split])
        exp_n = len(expanded[split])
        print(
            f"\n--- {split}: audioset_fp={fp_n}, sharded_expanded={exp_n} "
            f"→ merged={fp_n + exp_n} ---"
        )

        # Show category distribution changes
        from collections import Counter

        fp_cats = Counter(fp[split]["category"])
        mapped_cats = Counter(_map_category(c) for c in fp[split]["category"])
        exp_cats = Counter(expanded[split]["category"])

        print(f"  audioset_fp top-5: {fp_cats.most_common(5)}")
        print(f"  after mapping:   {mapped_cats}")
        print(f"  sharded_expanded:{dict(exp_cats)}")

        merged[split] = merge_split(fp[split], expanded[split])

        merged_cats = Counter(merged[split]["category"])
        print(f"  merged:          {dict(merged_cats)}")

    print(f"\nSaving to {OUTPUT_DIR} ...")
    # Save each split as parquet — skip save_to_disk (Arrow 2GB limit)
    for split in ["train", "validation", "test"]:
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        merged[split].to_parquet(str(split_dir / "data.parquet"))
        print(f"  saved {split} ({len(merged[split])} rows)")
    print("Done.")

    # Quick verification
    print("\n=== Verification ===")
    for split in ["train", "validation", "test"]:
        ds = merged[split]
        print(f"{split}: {len(ds)} rows, features={list(ds.features.keys())}")


if __name__ == "__main__":
    main()
