"""cmd_hearability_templates data-building subcommand."""
from __future__ import annotations

import sys
from pathlib import Path


def run() -> None:
    import argparse
    import json
    import math
    from dataclasses import asdict, dataclass
    from typing import Any
    
    import numpy as np
    from datasets import DatasetDict
    
    EPS = 1e-12
    
    @dataclass(frozen=True)
    class HearabilityConfig:
        sample_rate: int = 16000
        n_fft: int = 2048
        win_length: int = 2048
        hop_length: int = 512
        erb_bands: int = 48
        min_freq_hz: float = 80.0
        max_freq_hz: float = 7600.0
        template_percentile: float = 65.0
        active_weight_threshold: float = 0.01
        masking_offset_db: float = 0.0
        target_margin_db: float = 0.0
    
    def _to_mono_float32(audio: Any) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            arr = arr.mean(axis=axis)
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    
    def _decode_audio_row(
        row: dict[str, Any], *, key: str = "audio"
    ) -> tuple[np.ndarray, int | None]:
        value = row[key]
        if isinstance(value, dict):
            arr = np.asarray(
                value.get("array", value.get("path", value.get("data", value))),
                dtype=np.float32,
            )
            sr = value.get("sampling_rate", None)
        elif isinstance(value, str):
            raise NotImplementedError(
                "Path-based audio not supported — use HF datasets with dict audio"
            )
        else:
            arr = np.asarray(value, dtype=np.float32)
            sr = None
        arr = _to_mono_float32(arr)
        return arr, sr
    
    ap = argparse.ArgumentParser(
        description="Build hearability templates from background noise."
    )
    ap.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="HF dataset path on disk",
    )
    ap.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/hearability_templates"),
    )
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--win-length", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--erb-bands", type=int, default=48)
    ap.add_argument("--min-freq", type=float, default=80.0)
    ap.add_argument("--max-freq", type=float, default=7600.0)
    ap.add_argument("--template-percentile", type=float, default=65.0)
    ap.add_argument("--active-weight-threshold", type=float, default=0.01)
    ap.add_argument("--masking-offset-db", type=float, default=0.0)
    ap.add_argument("--target-margin-db", type=float, default=0.0)
    ap.add_argument(
        "--max-samples", type=int, default=5000, help="Limit for debugging"
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    cfg = HearabilityConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.n_fft if args.win_length is None else args.win_length,
        hop_length=args.hop_length,
        erb_bands=args.erb_bands,
        min_freq_hz=args.min_freq,
        max_freq_hz=args.max_freq,
        template_percentile=args.template_percentile,
        active_weight_threshold=args.active_weight_threshold,
        masking_offset_db=args.masking_offset_db,
        target_margin_db=args.target_margin_db,
    )
    if cfg.win_length <= 0 or cfg.win_length > cfg.n_fft:
        raise SystemExit(
            f"--win-length must be in [1, --n-fft], got {cfg.win_length}"
        )
    
    def erb_point(f_hz: float) -> float:
        return 21.4 * math.log10(1.0 + 0.00437 * f_hz)
    
    def erb_inv(n_erb: float) -> float:
        return (10.0 ** (n_erb / 21.4) - 1.0) / 0.00437
    
    def build_erb_filters(
        sr: int, n_fft: int, n_bands: int, f_min: float, f_max: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges_erb = np.linspace(
            erb_point(f_min), erb_point(f_max), n_bands + 2, dtype=np.float64
        )
        edges_hz = np.array([erb_inv(e) for e in edges_erb], dtype=np.float64)
        n_bins = n_fft // 2 + 1
        freqs = np.linspace(0.0, sr / 2.0, n_bins)
        filters = np.zeros((n_bands, n_bins), dtype=np.float64)
        for i in range(n_bands):
            lo, mid, hi = edges_hz[i], edges_hz[i + 1], edges_hz[i + 2]
            slope_up = (freqs - lo) / (mid - lo + EPS)
            slope_down = (hi - freqs) / (hi - mid + EPS)
            filters[i] = np.maximum(0.0, np.minimum(slope_up, slope_down))
        return (
            filters.astype(np.float32),
            edges_hz.astype(np.float32),
            freqs.astype(np.float32),
        )
    
    def power_spectrum(
        y: np.ndarray, n_fft: int, win_length: int, hop_length: int, sr: int
    ) -> np.ndarray:
        window = np.hanning(win_length).astype(np.float64)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            right = n_fft - win_length - left
            window = np.pad(window, (left, right), mode="constant")
        n_samples = len(y)
        frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                y,
                (0, n_fft - n_samples % hop_length)
                if n_samples % hop_length
                else (0, 0),
            ),
            n_fft,
        )[::hop_length]
        if frames.ndim == 1:
            frames = frames[np.newaxis, :]
        spec = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
        return spec.astype(np.float32)
    
    def erb_spectrum(power_spec: np.ndarray, filters: np.ndarray) -> np.ndarray:
        return np.dot(power_spec, filters.T).astype(np.float32)
    
    def amplitude_to_db(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(x, EPS))
    
    
    def compute_template(
        erb_spec_db: np.ndarray, filters: np.ndarray, percentile: float = 65.0
    ) -> np.ndarray:
        level = np.percentile(erb_spec_db, percentile, axis=0)
        return level.astype(np.float32)
    
    def compute_noise_masking_threshold(
        levels_db: np.ndarray,
        filters: np.ndarray,
        edges_hz: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        return levels_db.copy()
    
    print(f"Config: {asdict(cfg)}")
    print(f"\nLoading dataset from {args.dataset_path}...")
    ds_dict = DatasetDict.load_from_disk(str(args.dataset_path))
    print(f"Splits: {list(ds_dict.keys())}")
    ds = ds_dict.get(args.split, list(ds_dict.values())[0])
    print(f"Split '{args.split}': {len(ds)} samples")
    
    filters, edges_hz, freqs = build_erb_filters(
        cfg.sample_rate,
        cfg.n_fft,
        cfg.erb_bands,
        cfg.min_freq_hz,
        cfg.max_freq_hz,
    )
    print(f"Filters: {filters.shape} (bands x FFT bins)")
    print(f"Frequency range: {edges_hz[0]:.1f} – {edges_hz[-1]:.1f} Hz\n")
    
    total_duration = 0.0
    frame_powers = []
    
    n_samples = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
    print(f"Processing {n_samples} samples...")
    for idx in range(n_samples):
        row = ds[idx]
        audio, sr = _decode_audio_row(row)
        if sr and sr != cfg.sample_rate:
            from scipy.signal import resample
    
            audio = resample(
                audio, int(len(audio) * cfg.sample_rate / sr)
            ).astype(np.float32)
        dur = len(audio) / cfg.sample_rate
        if dur < args.min_duration:
            continue
        total_duration += dur
        power_spec = power_spectrum(
            audio, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.sample_rate
        )
        erb_spec = erb_spectrum(power_spec, filters)
        erb_db = amplitude_to_db(erb_spec)
        frame_powers.append(erb_db)
        if (idx + 1) % 500 == 0:
            print(
                f"  {idx + 1}/{n_samples} ({total_duration / 60:.1f} min audio)"
            )
    
    if not frame_powers:
        print("ERROR: No valid audio frames extracted")
        sys.exit(1)
    
    all_frames = np.concatenate(frame_powers, axis=0)
    print(
        f"\nTotal frames: {all_frames.shape[0]}  ({total_duration / 60:.1f} min)"
    )
    
    baseline_template = compute_template(
        all_frames, filters, cfg.template_percentile
    )
    masked_template = compute_noise_masking_threshold(
        baseline_template, filters, edges_hz, cfg.sample_rate
    )
    
    args.output_path.mkdir(parents=True, exist_ok=True)
    center_freqs = (edges_hz[:-2] + edges_hz[1:-1]) / 2.0
    band_edges = np.column_stack((edges_hz[:-2], edges_hz[1:-1]))
    np.savez_compressed(
        args.output_path / "hearability_template.npz",
        band_edges=band_edges.astype(np.float32),
        center_freqs=center_freqs.astype(np.float32),
        baseline_template=baseline_template.astype(np.float32),
        masked_template=masked_template.astype(np.float32),
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        erb_bands=cfg.erb_bands,
        total_frames=all_frames.shape[0],
        total_duration_seconds=total_duration,
    )
    print(f"\nSaved: {args.output_path / 'hearability_template.npz'}")
    
    meta = dict(
        config=asdict(cfg),
        total_frames=int(all_frames.shape[0]),
        total_duration_seconds=round(total_duration, 1),
        band_edges=band_edges.tolist(),
        center_freqs=center_freqs.tolist(),
        baseline_template=baseline_template.tolist(),
        masked_template=masked_template.tolist(),
    )
    with open(args.output_path / "hearability_template.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {args.output_path / 'hearability_template.json'}")
    
    print(f"\n{'=' * 55}")
    print("  Template Summary")
    print(f"{'=' * 55}")
    print(f"  {all_frames.shape[0]} frames ({total_duration / 60:.1f} min)")
    print(f"  Bands: {cfg.erb_bands}")
    print(f"  Percentile: {cfg.template_percentile:.0f}%")
    print(
        f"\n  {'Band':>5} {'Center(Hz)':>10} {'Level(dB)':>10}  {'Masked(dB)':>10}"
    )
    print(f"  {'─' * 40}")
    for i in range(0, cfg.erb_bands, max(1, cfg.erb_bands // 8)):
        print(
            f"  {i:>5} {center_freqs[i]:>10.0f} {baseline_template[i]:>10.2f}"
            f"  {masked_template[i]:>10.2f}"
        )
    
    
    # ====================================================================
    # urban-esc — Build ESC/UrbanSound chunked dataset
    # ====================================================================
    
    
