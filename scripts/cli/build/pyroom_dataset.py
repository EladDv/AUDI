#!/usr/bin/env python3
"""Build an HF DatasetDict of pyroomacoustics-simulated detector waveforms."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_from_disk,
)

from audi.config import MixConfig, parse_snr_bins
from audi.training.dataset import waveform_config_hash, waveform_config_payload
from audi.training.pyroom_dataset import (
    PyRoomDataset,
    PyRoomSimulationConfig,
    probe_audio_file,
    pyroom_config_hash,
    pyroom_config_payload,
    split_wavpack_files,
    write_mvdr_cache_file,
)

DEFAULT_NOISE_PATH = Path("data/20260603_uma16channel_lebanon_false_hunt")
DEFAULT_DRONE_PATH = Path("data/HF_dataset_v2_drone")
DEFAULT_BG_NOISE_PATH = Path("data/HF_dataset_v2_background")
DEFAULT_MVDR_CACHE_PATH = Path("data/pyroom_mvdr_cache")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build an HF DatasetDict of mono detector waveforms from real "
            "16-channel false-hunt background and pyroomacoustics free-space "
            "drone simulation."
        )
    )
    ap.add_argument("--noise-path", type=Path, default=DEFAULT_NOISE_PATH)
    ap.add_argument("--drone-path", type=Path, default=DEFAULT_DRONE_PATH)
    ap.add_argument(
        "--bg-noise-path",
        type=Path,
        default=DEFAULT_BG_NOISE_PATH,
        help="Mono HF background dataset to spatialize as extra random BG sources.",
    )
    ap.add_argument(
        "--snr-bin",
        action="append",
        default=[
            "easy:-5:0:0.25",
            "medium:-10:-5:0.30",
            "hard:-15:-10:0.30",
            "extreme:-20:-20:0.15",
        ],
    )
    ap.add_argument("--clip-seconds", type=float, default=1.28)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--highpass-hz", type=float, default=125.0)
    ap.add_argument("--positive-probability", type=float, default=0.5)
    ap.add_argument("--split", choices=["train", "validation", "test"], required=True)
    ap.add_argument("--num-examples", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel worker processes used to generate this split.",
    )
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--min-azimuth-deg", type=float, default=-75.0)
    ap.add_argument("--max-azimuth-deg", type=float, default=75.0)
    ap.add_argument("--min-elevation-deg", type=float, default=5.0)
    ap.add_argument("--max-elevation-deg", type=float, default=85.0)
    ap.add_argument("--min-distance-m", type=float, default=20.0)
    ap.add_argument("--max-distance-m", type=float, default=450.0)
    ap.add_argument(
        "--drone-reference-distance-m",
        type=float,
        default=10.0,
        help="Approximate distance where drone source recordings were captured.",
    )
    ap.add_argument("--steering-error-deg", type=float, default=0.0)
    ap.add_argument("--stft-n-fft", type=int, default=512)
    ap.add_argument("--hop-length", type=int, default=160)
    ap.add_argument("--diagonal-loading", type=float, default=1e-2)
    ap.add_argument(
        "--mvdr-cache-dir",
        type=Path,
        default=None,
        help="Optional per-background MVDR inverse-covariance cache directory.",
    )
    ap.add_argument(
        "--mvdr-cache-seconds",
        type=float,
        default=30.0,
        help="Seconds per background file used when creating missing MVDR cache entries.",
    )
    ap.add_argument(
        "--beamformer",
        choices=["mvdr", "mean", "random-channel"],
        default="mvdr",
        help="How to collapse 16 channels into detector mono audio.",
    )
    ap.add_argument("--sensor-noise-db", type=float, default=-45.0)
    ap.add_argument("--no-sensor-noise", action="store_true")
    ap.add_argument("--temperature-c", type=float, default=20.0)
    ap.add_argument("--humidity-percent", type=float, default=50.0)
    ap.add_argument("--no-air-absorption", action="store_true")
    ap.add_argument("--no-deglitch-audio", action="store_true")
    ap.add_argument("--deglitch-threshold", type=float, default=0.001)
    ap.add_argument("--deglitch-loudness-ratio", type=float, default=8.0)
    ap.add_argument("--deglitch-diff-ratio", type=float, default=12.0)
    ap.add_argument("--deglitch-window-samples", type=int, default=64)
    ap.add_argument("--bg-noise-probability", type=float, default=0.25)
    ap.add_argument("--bg-noise-multi-probability", type=float, default=0.5)
    ap.add_argument("--bg-noise-count", type=int, default=3)
    ap.add_argument("--bg-noise-max-attenuation-db", type=float, default=-40.0)
    ap.add_argument(
        "--random-beam-probability",
        type=float,
        default=0.0,
        help="Probability that a positive sample uses an independent beam direction.",
    )
    ap.add_argument(
        "--soft-target-by-beam-alignment",
        action="store_true",
        help="Scale positive labels by max(cosine(drone, beam), floor).",
    )
    ap.add_argument(
        "--target-alignment-floor",
        type=float,
        default=0.5,
        help="Minimum positive target value when soft beam-alignment labels are enabled.",
    )
    return ap


def build_cache_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Precompute per-background MVDR inverse-covariance cache files."
    )
    ap.add_argument("--noise-path", type=Path, default=DEFAULT_NOISE_PATH)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_MVDR_CACHE_PATH)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--highpass-hz", type=float, default=125.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--split", choices=["all", "train", "validation"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stft-n-fft", type=int, default=512)
    ap.add_argument("--hop-length", type=int, default=160)
    ap.add_argument("--diagonal-loading", type=float, default=1e-2)
    ap.add_argument("--mvdr-cache-seconds", type=float, default=30.0)
    ap.add_argument("--no-deglitch-audio", action="store_true")
    ap.add_argument("--deglitch-threshold", type=float, default=0.001)
    ap.add_argument("--deglitch-loudness-ratio", type=float, default=8.0)
    ap.add_argument("--deglitch-diff-ratio", type=float, default=12.0)
    ap.add_argument("--deglitch-window-samples", type=int, default=64)
    return ap


def make_mix_cfg(args: argparse.Namespace) -> MixConfig:
    return MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        noise2_path=args.bg_noise_path,
        noise2_prob=args.bg_noise_probability,
        noise2_multi_noise_prob=args.bg_noise_multi_probability,
        noise2_count=args.bg_noise_count,
        noise2_max_attenuation_db=args.bg_noise_max_attenuation_db,
        snr_bins=parse_snr_bins(args.snr_bin),
        target_length_samples=int(args.sample_rate * args.clip_seconds),
        positive_probability=args.positive_probability,
        highpass_hz=args.highpass_hz,
        sample_rate=args.sample_rate,
        aug=None,
    )


def make_pyroom_cfg(args: argparse.Namespace) -> PyRoomSimulationConfig:
    return PyRoomSimulationConfig(
        min_azimuth_deg=args.min_azimuth_deg,
        max_azimuth_deg=args.max_azimuth_deg,
        min_elevation_deg=args.min_elevation_deg,
        max_elevation_deg=args.max_elevation_deg,
        min_distance_m=args.min_distance_m,
        max_distance_m=args.max_distance_m,
        drone_reference_distance_m=args.drone_reference_distance_m,
        steering_error_deg=args.steering_error_deg,
        stft_n_fft=args.stft_n_fft,
        hop_length=args.hop_length,
        diagonal_loading=args.diagonal_loading,
        mvdr_cache_dir=str(args.mvdr_cache_dir) if args.mvdr_cache_dir else None,
        mvdr_cache_seconds=args.mvdr_cache_seconds,
        sensor_noise_db=None if args.no_sensor_noise else args.sensor_noise_db,
        temperature_c=args.temperature_c,
        humidity_percent=args.humidity_percent,
        air_absorption=not args.no_air_absorption,
        random_beam_probability=args.random_beam_probability,
        soft_target_by_beam_alignment=args.soft_target_by_beam_alignment,
        target_alignment_floor=args.target_alignment_floor,
        beamformer=args.beamformer,
        deglitch_audio=not args.no_deglitch_audio,
        deglitch_threshold=args.deglitch_threshold,
        deglitch_loudness_ratio=args.deglitch_loudness_ratio,
        deglitch_diff_ratio=args.deglitch_diff_ratio,
        deglitch_window_samples=args.deglitch_window_samples,
        spatial_bg_probability=args.bg_noise_probability,
        spatial_bg_multi_probability=args.bg_noise_multi_probability,
        spatial_bg_count=args.bg_noise_count,
        spatial_bg_max_attenuation_db=args.bg_noise_max_attenuation_db,
    )


def make_cache_pyroom_cfg(args: argparse.Namespace) -> PyRoomSimulationConfig:
    return PyRoomSimulationConfig(
        stft_n_fft=args.stft_n_fft,
        hop_length=args.hop_length,
        diagonal_loading=args.diagonal_loading,
        beamformer="mvdr",
        deglitch_audio=not args.no_deglitch_audio,
        deglitch_threshold=args.deglitch_threshold,
        deglitch_loudness_ratio=args.deglitch_loudness_ratio,
        deglitch_diff_ratio=args.deglitch_diff_ratio,
        deglitch_window_samples=args.deglitch_window_samples,
        mvdr_cache_dir=str(args.cache_dir),
        mvdr_cache_seconds=args.mvdr_cache_seconds,
    )


def _record_from_item(item: tuple[torch.Tensor, ...], sample_rate: int) -> dict:
    if len(item) == 12:
        snr_idx = 5
        meta_start = 6
    elif len(item) == 10:
        snr_idx = 3
        meta_start = 4
    else:
        raise ValueError(f"Unsupported PyRoomDataset item length: {len(item)}")
    return {
        "audio": {
            "array": item[0].detach().cpu().numpy().astype(np.float32),
            "sampling_rate": sample_rate,
        },
        "label": float(item[1].item()),
        "bin_idx": int(item[2].item()),
        "snr_db": float(item[snr_idx].item()),
        "drone_azimuth_deg": float(item[meta_start].item()),
        "drone_elevation_deg": float(item[meta_start + 1].item()),
        "drone_distance_m": float(item[meta_start + 2].item()),
        "beam_azimuth_deg": float(item[meta_start + 3].item()),
        "beam_elevation_deg": float(item[meta_start + 4].item()),
        "beam_target_scale": float(item[meta_start + 5].item()),
    }


def _features(sample_rate: int) -> Features:
    return Features(
        {
            "audio": Audio(sampling_rate=sample_rate),
            "label": Value("float32"),
            "bin_idx": Value("int64"),
            "snr_db": Value("float32"),
            "drone_azimuth_deg": Value("float32"),
            "drone_elevation_deg": Value("float32"),
            "drone_distance_m": Value("float32"),
            "beam_azimuth_deg": Value("float32"),
            "beam_elevation_deg": Value("float32"),
            "beam_target_scale": Value("float32"),
        }
    )


def _save_split_dataset(
    output_dir: Path,
    *,
    split: str,
    dataset: Dataset,
    manifest: dict,
) -> None:
    existing: dict[str, Dataset] = {}
    manifest_path = output_dir / "pyroom_manifest.json"
    existing_manifest = {}
    if (output_dir / "dataset_dict.json").exists():
        loaded = load_from_disk(str(output_dir))
        existing = {name: loaded[name] for name in loaded.keys()}
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text())
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} exists but is not an HF DatasetDict. Remove it or use a new path."
        )

    existing[split] = dataset
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(existing).save_to_disk(str(tmp_dir))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(tmp_dir), str(output_dir))
    if existing_manifest.get("format") != "audi_pyroom_hf_dataset_manifest_v1":
        existing_manifest = {
            "format": "audi_pyroom_hf_dataset_manifest_v1",
            "splits": {},
        }
    existing_manifest["splits"][split] = manifest
    manifest_path.write_text(json.dumps(existing_manifest, indent=2, sort_keys=True))


def _worker_counts(num_examples: int, num_workers: int) -> list[int]:
    workers = min(max(1, num_workers), num_examples)
    base = num_examples // workers
    extra = num_examples % workers
    return [base + (1 if idx < extra else 0) for idx in range(workers)]


def _shard_root(args: argparse.Namespace) -> Path:
    return args.output_dir.with_name(f".{args.output_dir.name}.{args.split}.shards")


def _make_pyroom_dataset(args: argparse.Namespace, *, length: int) -> PyRoomDataset:
    mix_cfg = make_mix_cfg(args)
    pyroom_cfg = make_pyroom_cfg(args)
    return PyRoomDataset(
        noise_dir=args.noise_path,
        drone_path=args.drone_path,
        split=args.split,
        snr_bins=mix_cfg.snr_bins,
        target_length_samples=mix_cfg.target_length_samples,
        positive_probability=mix_cfg.positive_probability,
        highpass_hz=mix_cfg.highpass_hz,
        sample_rate=mix_cfg.sample_rate,
        cfg=pyroom_cfg,
        length=length,
        return_components=False,
        validation_fraction=args.validation_fraction,
        file_split_seed=args.seed,
        spatial_bg_path=args.bg_noise_path if args.bg_noise_probability > 0.0 else None,
    )


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_dataset_shard(
    args: argparse.Namespace,
    *,
    shard_idx: int,
    num_rows: int,
    shard_dir: Path,
) -> str:
    seed = args.seed + 1009 * (shard_idx + 1)
    _seed_worker(seed)
    mix_cfg = make_mix_cfg(args)
    ds = _make_pyroom_dataset(args, length=num_rows)

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            range(num_rows),
            desc=f"pyroom {args.beamformer} {args.split} w{shard_idx}",
            position=shard_idx,
            unit="row",
        )
    except Exception:
        iterator = range(num_rows)

    def record_iter():
        for idx in iterator:
            yield _record_from_item(ds[idx], mix_cfg.sample_rate)

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        record_iter,
        features=_features(mix_cfg.sample_rate),
        keep_in_memory=False,
        cache_dir=str(shard_dir.with_suffix(".cache")),
    )
    dataset.save_to_disk(str(shard_dir))
    return str(shard_dir)


def _build_serial_dataset(args: argparse.Namespace) -> Dataset:
    mix_cfg = make_mix_cfg(args)
    ds = _make_pyroom_dataset(args, length=args.num_examples)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            range(args.num_examples),
            desc=f"pyroom {args.beamformer} {args.split}",
            unit="row",
        )
    except Exception:
        iterator = range(args.num_examples)

    def record_iter():
        for idx in iterator:
            yield _record_from_item(ds[idx], mix_cfg.sample_rate)

    return Dataset.from_generator(
        record_iter,
        features=_features(mix_cfg.sample_rate),
        keep_in_memory=False,
    )


def _build_parallel_dataset(args: argparse.Namespace) -> Dataset:
    counts = _worker_counts(args.num_examples, args.num_workers)
    shard_root = _shard_root(args)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str | None] = [None] * len(counts)
    try:
        with ProcessPoolExecutor(max_workers=len(counts)) as executor:
            futures = {
                executor.submit(
                    _write_dataset_shard,
                    args,
                    shard_idx=idx,
                    num_rows=count,
                    shard_dir=shard_root / f"shard_{idx:03d}",
                ): idx
                for idx, count in enumerate(counts)
            }
            try:
                from tqdm.auto import tqdm

                future_iter = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"pyroom {args.beamformer} {args.split} shards",
                    unit="shard",
                )
            except Exception:
                future_iter = as_completed(futures)

            for future in future_iter:
                idx = futures[future]
                shard_paths[idx] = future.result()

        loaded = [load_from_disk(path) for path in shard_paths if path is not None]
        return concatenate_datasets(loaded)
    finally:
        for cache_dir in shard_root.parent.glob(f"{shard_root.name}*.cache"):
            if cache_dir.exists():
                shutil.rmtree(cache_dir)


def run() -> int:
    args = build_parser().parse_args()
    if args.num_examples <= 0:
        raise SystemExit("--num-examples must be positive")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")

    _seed_worker(args.seed)

    mix_cfg = make_mix_cfg(args)
    pyroom_cfg = make_pyroom_cfg(args)
    manifest = {
        "format": "audi_pyroom_hf_dataset_v1",
        "split": args.split,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "num_workers": min(args.num_workers, args.num_examples),
        "waveform_config_hash": waveform_config_hash(mix_cfg),
        "waveform_config": waveform_config_payload(mix_cfg),
        "pyroom_config_hash": pyroom_config_hash(pyroom_cfg),
        "pyroom_config": pyroom_config_payload(pyroom_cfg),
    }
    if args.num_workers == 1:
        dataset = _build_serial_dataset(args)
    else:
        dataset = _build_parallel_dataset(args)
    _save_split_dataset(
        args.output_dir,
        split=args.split,
        dataset=dataset,
        manifest=manifest,
    )
    if args.num_workers > 1:
        shard_root = _shard_root(args)
        if shard_root.exists():
            shutil.rmtree(shard_root)

    print(f"wrote HF dataset split {args.split!r} to {args.output_dir} n={len(dataset)}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_cache() -> int:
    args = build_cache_parser().parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided")

    if args.split == "all":
        files = sorted(args.noise_path.glob("*.wv"))
        if not files:
            raise SystemExit(f"No .wv files found in {args.noise_path}")
    else:
        files = split_wavpack_files(
            args.noise_path,
            split=args.split,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    if args.limit is not None:
        files = files[: args.limit]

    cfg = make_cache_pyroom_cfg(args)
    manifest = {
        "format": "audi_pyroom_mvdr_cache_manifest_v1",
        "noise_path": str(args.noise_path),
        "cache_dir": str(args.cache_dir),
        "split": args.split,
        "num_files": len(files),
        "seed": args.seed,
        "highpass_hz": args.highpass_hz,
        "sample_rate": args.sample_rate,
        "pyroom_config_hash": pyroom_config_hash(cfg),
        "pyroom_config": pyroom_config_payload(cfg),
        "files": [],
    }

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(files, desc="pyroom mvdr cache", unit="file")
    except Exception:
        iterator = files

    for path in iterator:
        info = probe_audio_file(path)
        cache_path = write_mvdr_cache_file(
            info,
            cache_dir=args.cache_dir,
            cfg=cfg,
            highpass_hz=args.highpass_hz,
            sample_rate=args.sample_rate,
            cache_seconds=args.mvdr_cache_seconds,
        )
        manifest["files"].append(
            {
                "source": str(path),
                "cache": str(cache_path),
            }
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "pyroom_mvdr_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote MVDR cache files to {args.cache_dir} n={len(files)}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
