"""cmd_filtered_hf data-building subcommand."""

from __future__ import annotations

import sys
from pathlib import Path


def run() -> None:
    import argparse
    from collections.abc import Iterable

    import numpy as np
    from datasets import Audio, Dataset, DatasetDict, Features, Value

    _AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")

    def _iter_audio_files(dir_path: Path) -> Iterable[Path]:
        for p in sorted(dir_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
                yield p

    def _split_dataset(ds, *, val_ratio, test_ratio, seed):
        if len(ds) == 0:
            return None
        if len(ds) == 1:
            return DatasetDict(train=ds)
        test_size = int(round(float(test_ratio) * len(ds)))
        val_size = int(round(float(val_ratio) * len(ds)))
        if test_ratio > 0.0:
            test_size = max(1, test_size)
        if val_ratio > 0.0 and len(ds) - test_size > 1:
            val_size = max(1, val_size)
        max_holdout = max(0, len(ds) - 1)
        if test_size + val_size > max_holdout:
            overflow = test_size + val_size - max_holdout
            reduce_val = min(val_size, overflow)
            val_size -= reduce_val
            overflow -= reduce_val
            test_size = max(0, test_size - overflow)
        if test_size <= 0:
            trainval = ds
            test = None
        else:
            tmp = ds.train_test_split(
                test_size=test_size, seed=int(seed), shuffle=True
            )
            trainval = tmp["train"]
            test = tmp["test"]
        if val_size <= 0:
            return (
                DatasetDict(train=trainval, test=test)
                if test
                else DatasetDict(train=trainval)
            )
        tmp2 = trainval.train_test_split(
            test_size=val_size, seed=int(seed + 1), shuffle=True
        )
        return (
            DatasetDict(train=tmp2["train"], validation=tmp2["test"], test=test)
            if test
            else DatasetDict(train=tmp2["train"], validation=tmp2["test"])
        )

    ap = argparse.ArgumentParser(
        description="Build filtered HF datasets from audio directories."
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with noise subfolders",
    )
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", type=str, default="noise")
    ap.add_argument(
        "--split-background",
        action="store_false",
        help="Split BG subfolder into _background, rest into _drone",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=30.0,
        help="Chunk audio into fixed-length segments (seconds). Default: 30",
    )
    args = ap.parse_args()

    from scipy.signal import resample as sp_resample

    print(f"Scanning {args.input_dir}...")
    audio_files = list(_iter_audio_files(args.input_dir))
    print(f"Found {len(audio_files)} audio files")

    def _load_audio(fp: Path) -> np.ndarray | None:
        import soundfile as sf

        try:
            audio, sr = sf.read(str(fp))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if sr != args.target_sr:
                new_len = int(len(audio) * args.target_sr / sr)
                audio = sp_resample(audio, new_len).astype(np.float32)
            return audio
        except Exception as e:
            print(f"  SKIP {fp.name}: {e}")
            return None

    def _chunk_audio(audio: np.ndarray) -> list[np.ndarray]:
        chunk_samples = int(args.chunk_sec * args.target_sr)
        total = len(audio)
        if total <= chunk_samples:
            return [audio]
        chunks = []
        pos = 0
        while pos + chunk_samples < total:
            chunks.append(audio[pos : pos + chunk_samples].copy())
            pos += chunk_samples
        # Last chunk: overlap backward to get a full segment
        chunks.append(audio[total - chunk_samples : total].copy())
        return chunks

    _CHUNK_SIZE = 1000  # records per in-memory shard (~2 GB peak)

    def _build_dataset(files, label: str, output_path: Path):
        import shutil

        features = Features(
            {
                "audio": Audio(sampling_rate=args.target_sr),
                "label": Value("string"),
            }
        )
        tmp_dir = output_path.parent / f"_tmp_{output_path.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        chunk_dirs = []
        records = []
        chunk_idx = 0
        total = 0
        for fp in files:
            audio = _load_audio(fp)
            if audio is None:
                continue
            for c in _chunk_audio(audio):
                records.append(
                    {
                        "audio": {
                            "array": c,
                            "sampling_rate": args.target_sr,
                        },
                        "label": label,
                    }
                )
                if len(records) >= _CHUNK_SIZE:
                    chunk = Dataset.from_list(records, features=features)
                    cd = tmp_dir / f"chunk_{chunk_idx:06d}"
                    chunk.save_to_disk(str(cd))
                    chunk_dirs.append(cd)
                    total += len(records)
                    print(f"  {label}: chunk {chunk_idx} ({len(records)} records) → {cd}")
                    records = []
                    chunk_idx += 1

        # Flush remaining
        if records:
            chunk = Dataset.from_list(records, features=features)
            cd = tmp_dir / f"chunk_{chunk_idx:06d}"
            chunk.save_to_disk(str(cd))
            chunk_dirs.append(cd)
            total += len(records)
            print(f"  {label}: chunk {chunk_idx} ({len(records)} records) → {cd}")
            records = []

        if not chunk_dirs:
            print(f"  WARNING: No records — skipping {output_path}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        print(f"  {label}: {total} records across {len(chunk_dirs)} chunks")

        # Combine chunks
        from datasets import concatenate_datasets as _concat

        print(f"  Combining {len(chunk_dirs)} chunks ...")
        ds = _concat([Dataset.load_from_disk(str(cd)) for cd in chunk_dirs])
        for cd in chunk_dirs:
            shutil.rmtree(cd, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        dd = _split_dataset(
            ds,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        if dd is None:
            print(f"  ERROR: Empty dataset for {output_path}")
            return
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(output_path))
        for split_name, split_ds in dd.items():
            print(f"    {split_name}: {len(split_ds)} samples")
        print(f"  Saved: {output_path}")

    if args.split_background:
        bg_files = [f for f in audio_files if f.parent.name == "BG"]
        drone_files = [f for f in audio_files if f.parent.name != "BG"]
        _build_dataset(
            bg_files,
            "no_drone",
            args.output_path.with_name(args.output_path.name + "_background"),
        )
        _build_dataset(
            drone_files,
            "drone",
            args.output_path.with_name(args.output_path.name + "_drone"),
        )
    else:
        # Incremental chunked build — same approach as _build_dataset
        import shutil

        fs = Features(
            {
                "audio": Audio(sampling_rate=args.target_sr),
                "label": Value("string"),
            }
        )
        tmp_dir = args.output_path.parent / f"_tmp_{args.output_path.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        chunk_dirs = []
        records = []
        chunk_idx = 0
        total = 0
        for fp in audio_files:
            audio = _load_audio(fp)
            if audio is None:
                continue
            relative = fp.relative_to(args.input_dir)
            category = (
                str(relative.parent)
                if relative.parent != Path(".")
                else args.label
            )
            for c in _chunk_audio(audio):
                records.append(
                    {
                        "audio": {
                            "array": c,
                            "sampling_rate": args.target_sr,
                        },
                        "label": category,
                    }
                )
                if len(records) >= _CHUNK_SIZE:
                    chunk = Dataset.from_list(records, features=fs)
                    cd = tmp_dir / f"chunk_{chunk_idx:06d}"
                    chunk.save_to_disk(str(cd))
                    chunk_dirs.append(cd)
                    total += len(records)
                    print(f"  chunk {chunk_idx} ({len(records)} records) → {cd}")
                    records = []
                    chunk_idx += 1

        if records:
            chunk = Dataset.from_list(records, features=fs)
            cd = tmp_dir / f"chunk_{chunk_idx:06d}"
            chunk.save_to_disk(str(cd))
            chunk_dirs.append(cd)
            total += len(records)
            print(f"  chunk {chunk_idx} ({len(records)} records) → {cd}")
            records = []

        if not chunk_dirs:
            print("ERROR: Empty dataset")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            sys.exit(1)

        print(f"Loaded {total} records across {len(chunk_dirs)} chunks")
        print(f"Combining {len(chunk_dirs)} chunks ...")
        from datasets import concatenate_datasets as _concat

        ds = _concat([Dataset.load_from_disk(str(cd)) for cd in chunk_dirs])
        for cd in chunk_dirs:
            shutil.rmtree(cd, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        dd = _split_dataset(
            ds,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        if dd is None:
            print("ERROR: Empty dataset")
            sys.exit(1)
        if args.output_path.exists():
            shutil.rmtree(args.output_path)
        args.output_path.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(args.output_path))
        for split_name, split_ds in dd.items():
            print(f"  {split_name}: {len(split_ds)} samples")
        print(f"\nSaved: {args.output_path}")

    # ====================================================================
    # chunk-spectro — Chunk audio into segments and draw spectrograms
    # ====================================================================
