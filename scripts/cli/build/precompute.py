#!/usr/bin/env python3
"""Precompute detection datasets for CPU-light training.

Use ``precompute-waveforms`` to freeze only the waveform-side pipeline: HF audio
decode, resampling, random drone/background mixing, noise2 layering, and
waveform augmentations.

Use ``precompute-features`` to turn waveform shards into normalized mel frontend
feature shards. Feature shards are tied to both waveform and frontend HPs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from audi.config import AugmentationConfig, MelConfig, MixConfig, parse_snr_bins
from audi.training.dataset import (
    PrecomputedDetectionDataset,
    frontend_config_hash,
    frontend_config_payload,
    make_dataset,
    waveform_config_hash,
    waveform_config_payload,
)
from audi.training.detector import PCEN


def build_waveform_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Precompute mixed/augmented detection waveform shards."
    )
    ap.add_argument("--noise-path", type=Path, required=True)
    ap.add_argument("--drone-path", type=Path, required=True)
    ap.add_argument("--hard-noise", type=Path, default=None)
    ap.add_argument("--hard-noise-prob", type=float, default=0.0)
    ap.add_argument("--noise2", type=Path, default=None)
    ap.add_argument("--noise2-prob", type=float, default=0.25)
    ap.add_argument("--noise2-multi-prob", type=float, default=0.5)
    ap.add_argument("--noise2-count", type=int, default=3)
    ap.add_argument("--noise2-max-attenuation", type=float, default=-40.0)
    ap.add_argument(
        "--snr-bin",
        action="append",
        default=[
            "easy:-5:0:0.25",
            "medium:-10:-5:0.30",
            "hard:-15:-10:0.30",
            "extreme:-20:-25:0.15",
        ],
    )
    ap.add_argument("--clip-seconds", type=float, default=1.28)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--highpass-hz", type=float, default=125.0)
    ap.add_argument("--positive-probability", type=float, default=0.5)
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--num-examples", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=7)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)

    # Waveform augmentations. Defaults match train_detect.py.
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--doppler-prob", type=float, default=0.2)
    ap.add_argument("--pitch-prob", type=float, default=0.25)
    ap.add_argument("--stretch-prob", type=float, default=0.25)
    ap.add_argument("--reverb-prob", type=float, default=0.25)
    ap.add_argument("--eq-prob", type=float, default=0.25)
    ap.add_argument("--noise-inject-prob", type=float, default=0.25)
    ap.add_argument("--noise-inject-db", type=float, default=-40.0)
    ap.add_argument("--time-mask-prob", type=float, default=0.25)
    ap.add_argument("--lowpass-prob", type=float, default=0.25)
    ap.add_argument("--atmospheric-prob", type=float, default=0.25)
    return ap


def build_feature_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Precompute normalized frontend feature tensors from waveform shards."
    )
    ap.add_argument("--waveform-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--win-length", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=160)
    ap.add_argument("--use-pcen", action="store_true")
    ap.add_argument("--pcen-s", type=float, default=0.025)
    ap.add_argument("--pcen-alpha", type=float, default=0.98)
    ap.add_argument("--pcen-delta", type=float, default=2.0)
    ap.add_argument("--pcen-r", type=float, default=0.5)
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Feature extraction device. auto uses CUDA when available, otherwise CPU.",
    )
    return ap


def make_waveform_mix_cfg(args: argparse.Namespace) -> MixConfig:
    aug_cfg = None
    if args.augment:
        aug_cfg = AugmentationConfig(
            enable=True,
            doppler_prob=args.doppler_prob,
            pitch_prob=args.pitch_prob,
            stretch_prob=args.stretch_prob,
            reverb_prob=args.reverb_prob,
            eq_prob=args.eq_prob,
            noise_inject_prob=args.noise_inject_prob,
            noise_inject_db=args.noise_inject_db,
            time_mask_prob=args.time_mask_prob,
            lowpass_prob=args.lowpass_prob,
            atmospheric_prob=args.atmospheric_prob,
        )
    return MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        hard_noise_path=args.hard_noise,
        hard_noise_prob=args.hard_noise_prob,
        noise2_path=args.noise2,
        noise2_prob=args.noise2_prob,
        noise2_multi_noise_prob=args.noise2_multi_prob,
        noise2_count=args.noise2_count,
        noise2_max_attenuation_db=args.noise2_max_attenuation,
        snr_bins=parse_snr_bins(args.snr_bin),
        target_length_samples=int(args.sample_rate * args.clip_seconds),
        positive_probability=args.positive_probability,
        highpass_hz=args.highpass_hz,
        sample_rate=args.sample_rate,
        aug=aug_cfg,
    )


def make_feature_mel_cfg(args: argparse.Namespace) -> MelConfig:
    cfg = MelConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
    )
    if args.use_pcen:
        cfg = replace(
            cfg,
            use_pcen=True,
            pcen_s=args.pcen_s,
            pcen_alpha=args.pcen_alpha,
            pcen_delta=args.pcen_delta,
            pcen_r=args.pcen_r,
        )
    return cfg


def run_waveforms() -> int:
    args = build_waveform_parser().parse_args()
    if args.shard_size <= 0 or args.num_examples <= 0:
        raise SystemExit("--shard-size and --num-examples must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = make_waveform_mix_cfg(args)
    return_components = args.split == "validation"
    ds = make_dataset(
        cfg=cfg,
        split=args.split,
        return_bin=not return_components,
        return_components=return_components,
    )
    ds.length = args.num_examples

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.pt"):
        stale.unlink()

    manifest = {
        "format": "audi_precomputed_waveforms_v1",
        "split": args.split,
        "num_examples": args.num_examples,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "waveform_config_hash": waveform_config_hash(cfg),
        "waveform_config": waveform_config_payload(cfg),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    loader = DataLoader(
        ds,
        batch_size=args.shard_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    total_shards = math.ceil(args.num_examples / args.shard_size)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            loader,
            total=total_shards,
            desc=f"precompute {args.split}",
            unit="shard",
            dynamic_ncols=True,
        )
    except Exception:
        iterator = loader

    for shard_idx, batch in enumerate(iterator):
        cols = tuple(batch)
        out = {
            "mix": cols[0].float(),
            "label": cols[1].float(),
            "bin_idx": cols[2].long(),
        }
        if return_components:
            out["drone"] = cols[3].float()
            out["noise"] = cols[4].float()
            out["snr_db"] = cols[5].float()
        path = args.output_dir / f"shard-{shard_idx:05d}.pt"
        torch.save(out, path)
        print(f"wrote {path} n={int(out['mix'].shape[0])}", flush=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_features() -> int:
    args = build_feature_parser().parse_args()
    mel_cfg = make_feature_mel_cfg(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    wave_manifest_path = args.waveform_path / "manifest.json"
    if not wave_manifest_path.exists():
        raise SystemExit(f"Missing waveform manifest: {wave_manifest_path}")
    wave_manifest = json.loads(wave_manifest_path.read_text())
    if wave_manifest.get("split") != args.split:
        raise SystemExit(
            f"Waveform split mismatch: expected {args.split}, "
            f"manifest has {wave_manifest.get('split')}"
        )

    ds = PrecomputedDetectionDataset(args.waveform_path, return_bin=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mel_transform = T.MelSpectrogram(
        sample_rate=mel_cfg.sample_rate,
        n_fft=mel_cfg.n_fft,
        win_length=mel_cfg.win_length,
        hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels,
    ).to(device)
    to_db = T.AmplitudeToDB().to(device)
    pcen = (
        PCEN(
            s=mel_cfg.pcen_s,
            alpha=mel_cfg.pcen_alpha,
            delta=mel_cfg.pcen_delta,
            r=mel_cfg.pcen_r,
            eps=mel_cfg.pcen_eps,
        ).to(device)
        if mel_cfg.use_pcen
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.pt"):
        stale.unlink()

    manifest = {
        "format": "audi_precomputed_features_v1",
        "split": args.split,
        "source_waveform_path": str(args.waveform_path),
        "waveform_config_hash": wave_manifest["waveform_config_hash"],
        "waveform_config": wave_manifest["waveform_config"],
        "frontend_config_hash": frontend_config_hash(mel_cfg),
        "frontend_config": frontend_config_payload(mel_cfg),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            loader,
            total=math.ceil(len(ds) / args.batch_size),
            desc=f"features {args.split}",
            unit="batch",
            dynamic_ncols=True,
        )
    except Exception:
        iterator = loader

    with torch.no_grad():
        for shard_idx, batch in enumerate(iterator):
            wav, label, bin_idx = batch
            wav = wav.to(device, non_blocking=device.type == "cuda")
            mel = mel_transform(wav)
            if pcen is not None:
                mel = pcen(mel)
            else:
                mel = to_db(mel)
                if mel_cfg.mean_db is not None and mel_cfg.std_db is not None:
                    mel = (mel - mel_cfg.mean_db) / mel_cfg.std_db
            spec = mel.unsqueeze(1).expand(-1, 3, -1, -1).contiguous().cpu()
            out = {
                "spec": torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0),
                "label": label.float().cpu(),
                "bin_idx": bin_idx.long().cpu(),
            }
            path = args.output_dir / f"shard-{shard_idx:05d}.pt"
            torch.save(out, path)
            print(f"wrote {path} n={int(out['spec'].shape[0])}", flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_waveforms())
