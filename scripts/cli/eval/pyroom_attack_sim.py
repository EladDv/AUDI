"""Moving-source PyRoom/MVDR attack simulation artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk
from scipy.signal import istft, spectrogram, stft
from tqdm import tqdm

from audi.augment import highpass, peak_limit
from audi.checkpoint import load_model_from_checkpoint
from audi.training.dataset import _rms_normalize
from audi.training.pyroom_dataset import (
    PyRoomSimulationConfig,
    deglitch_multichannel,
    direction_unit_vector,
    estimate_mvdr_inverse_covariance,
    load_hf_audio_array,
    load_multichannel_segment,
    mvdr_weights_from_inverse_covariance,
    planar_array_positions,
    probe_audio_file,
    split_wavpack_files,
)

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_NOISE_DIR = PROJECT / "data/20260603_uma16channel_lebanon_false_hunt"
DEFAULT_DRONE_PATH = PROJECT / "data/HF_dataset_v2_drone"
DEFAULT_OUT_DIR = PROJECT / "artifacts/pyroom_attack_sim_20260603_test_mvdr"
APP_SRC = PROJECT / "audi-app/src"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _ffmpeg_decode_full(
    path: Path,
    *,
    target_sample_rate: int,
    expected_channels: int,
) -> np.ndarray:
    info = probe_audio_file(path)
    length_samples = int(math.ceil(info.duration_seconds * target_sample_rate))
    return load_multichannel_segment(
        info,
        start_seconds=0.0,
        length_samples=length_samples,
        target_sample_rate=target_sample_rate,
        expected_channels=expected_channels,
    )


def _load_drone_clip(dataset_path: Path, *, split: str, idx: int, sample_rate: int) -> np.ndarray:
    dd = load_from_disk(str(dataset_path))
    actual_split = split if split in dd else "validation"
    ds = dd[actual_split]
    row = ds[int(idx) % len(ds)]
    drone = load_hf_audio_array(row["audio"], target_sample_rate=sample_rate)
    if drone.size == 0:
        raise ValueError(f"empty drone row {idx} in {dataset_path}")
    drone = highpass(drone, cutoff_hz=125.0, sample_rate=sample_rate)
    return _rms_normalize(drone)


def _tile_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= length:
        return audio[:length]
    repeats = int(math.ceil(length / max(audio.size, 1)))
    return np.tile(audio, repeats)[:length].astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _moving_drone_channels(
    drone: np.ndarray,
    *,
    positions_m: np.ndarray,
    sample_rate: int,
    length_samples: int,
    azimuth_deg: float,
    elevation_deg: float,
    start_distance_m: float,
    speed_mps: float,
    min_distance_m: float,
    reference_distance_m: float,
    reference_amplitude: float,
    speed_of_sound_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a direct-path moving source with time-varying delay and distance loss."""
    source = _tile_to_length(drone, length_samples + int(sample_rate * 8))
    source = source * np.float32(reference_amplitude)
    t = np.arange(length_samples, dtype=np.float64) / float(sample_rate)
    distance = np.maximum(
        float(min_distance_m),
        float(start_distance_m) - float(speed_mps) * t,
    )
    direction = direction_unit_vector(azimuth_deg, elevation_deg)
    channels = np.empty((positions_m.shape[0], length_samples), dtype=np.float32)
    source_time_base = np.arange(source.size, dtype=np.float64) / float(sample_rate)
    for mic_idx, mic_pos in enumerate(positions_m):
        mic_delay = -(float(np.dot(mic_pos, direction)) / float(speed_of_sound_mps))
        sample_time = t - (distance / float(speed_of_sound_mps)) - mic_delay
        shifted = np.interp(sample_time, source_time_base, source, left=0.0, right=0.0)
        gain = float(reference_distance_m) / np.maximum(distance, 1.0)
        channels[mic_idx] = (shifted * gain).astype(np.float32)
    return channels, distance.astype(np.float32)


def _audio_normalization_scale(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 1e-8:
        return 1.0
    scale = 1.0 / rms
    peak = float(np.max(np.abs(audio)))
    if peak * scale > 0.98:
        scale = 0.98 / max(peak, 1e-8)
    return float(scale)


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    return 20.0 * math.log10(max(_rms(signal), 1e-12) / max(_rms(noise), 1e-12))


def _window_snr_db(
    signal: np.ndarray,
    noise: np.ndarray,
    *,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> float:
    start = max(0, int(round(start_seconds * sample_rate)))
    stop = min(signal.shape[-1], start + int(round(duration_seconds * sample_rate)))
    if stop <= start:
        return float("nan")
    return _snr_db(signal[:, start:stop], noise[:, start:stop])


def _windowed_scores(
    model: torch.nn.Module,
    audio: np.ndarray,
    *,
    sample_rate: int,
    clip_samples: int,
    stride_samples: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, max(0, audio.size - clip_samples + 1), stride_samples, dtype=np.int64)
    if starts.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    device = next(model.parameters()).device
    scores: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    with torch.no_grad():
        for start in starts:
            window = np.asarray(audio[start : start + clip_samples], dtype=np.float32)
            window = window * np.float32(_audio_normalization_scale(window))
            batch.append(window)
            if len(batch) >= batch_size:
                wav = torch.as_tensor(np.stack(batch), dtype=torch.float32, device=device)
                logits = model(wav).detach().float().cpu().numpy().reshape(-1)
                scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
                batch.clear()
        if batch:
            wav = torch.as_tensor(np.stack(batch), dtype=torch.float32, device=device)
            logits = model(wav).detach().float().cpu().numpy().reshape(-1)
            scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    times = starts.astype(np.float32) / float(sample_rate)
    return times, np.concatenate(scores).astype(np.float32)


def _discover_default_models() -> list[ModelSpec]:
    groups = {
        "finetune": [
            PROJECT / "checkpoints/pyroom_mvdr_mn10_finetune_more_*/checkpoints/*.ckpt",
            PROJECT / "checkpoints/pyroom_mvdr_mn10_finetune_20260701_*/checkpoints/*.ckpt",
        ],
        "scratch": [
            PROJECT / "checkpoints/pyroom_mvdr_mn10_scratch_more_*/checkpoints/*.ckpt",
            PROJECT / "checkpoints/pyroom_mvdr_mn10_scratch_20260701_*/checkpoints/*.ckpt",
        ],
    }
    models: list[ModelSpec] = []
    for name, patterns in groups.items():
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(Path(p) for p in glob.glob(str(pattern)))
            if candidates:
                break
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        models.append(ModelSpec(name=name, checkpoint=latest))
    if not models:
        raise SystemExit("No PyRoom MN10 checkpoints found; pass --checkpoint explicitly.")
    return models


def _parse_models(values: list[str] | None) -> list[ModelSpec]:
    if not values:
        return _discover_default_models()
    specs: list[ModelSpec] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = (
                Path(path).parents[1].name
                if Path(path).parent.name == "checkpoints"
                else Path(path).stem
            )
        specs.append(ModelSpec(name=_safe_name(name), checkpoint=Path(path)))
    return specs


def _beamform_all(
    mix_channels: np.ndarray,
    noise_channels: np.ndarray,
    *,
    cfg: PyRoomSimulationConfig,
    beams: list[tuple[int, float, float]],
    sample_rate: int,
    positions_m: np.ndarray,
) -> dict[int, np.ndarray]:
    noise_channels = deglitch_multichannel(
        noise_channels,
        threshold=cfg.deglitch_threshold,
        loudness_ratio=cfg.deglitch_loudness_ratio,
        diff_ratio=cfg.deglitch_diff_ratio,
        window_samples=cfg.deglitch_window_samples,
    )
    mix_channels = deglitch_multichannel(
        mix_channels,
        threshold=cfg.deglitch_threshold,
        loudness_ratio=cfg.deglitch_loudness_ratio,
        diff_ratio=cfg.deglitch_diff_ratio,
        window_samples=cfg.deglitch_window_samples,
    )
    outputs: dict[int, np.ndarray] = {}
    freqs, inverse_covariance = estimate_mvdr_inverse_covariance(
        noise_channels,
        sample_rate=sample_rate,
        n_fft=cfg.stft_n_fft,
        hop_length=cfg.hop_length,
        diagonal_loading=cfg.diagonal_loading,
    )
    mix_freqs, _, mix_spec = stft(
        np.asarray(mix_channels, dtype=np.float32),
        fs=sample_rate,
        nperseg=cfg.stft_n_fft,
        noverlap=cfg.stft_n_fft - cfg.hop_length,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    if len(mix_freqs) != len(freqs):
        raise RuntimeError("MVDR covariance and mix STFT frequency bins differ")
    for beam_idx, az, el in tqdm(beams, desc="MVDR beams", leave=False):
        weights = mvdr_weights_from_inverse_covariance(
            freqs,
            inverse_covariance,
            positions_m=positions_m,
            look_direction=direction_unit_vector(az, el),
            speed_of_sound_mps=cfg.speed_of_sound_mps,
        )
        beam_spec = np.einsum("fm,mft->ft", weights.conj(), mix_spec, optimize=True)
        _, beam = istft(
            beam_spec,
            fs=sample_rate,
            nperseg=cfg.stft_n_fft,
            noverlap=cfg.stft_n_fft - cfg.hop_length,
            input_onesided=True,
        )
        beam = np.nan_to_num(beam[: mix_channels.shape[1]], nan=0.0, posinf=0.0, neginf=0.0)
        if beam.shape[0] < mix_channels.shape[1]:
            beam = np.pad(beam, (0, mix_channels.shape[1] - beam.shape[0]))
        outputs[beam_idx] = beam.astype(np.float32, copy=False)
    return outputs


def _build_beams(
    count: int,
    elevation_count: int,
    min_el: float,
    max_el: float,
) -> list[tuple[int, float, float]]:
    elevations = (
        [(min_el + max_el) / 2.0]
        if elevation_count <= 1
        else np.linspace(min_el, max_el, elevation_count).tolist()
    )
    az_count = max(1, math.ceil(count / max(1, elevation_count)))
    beams: list[tuple[int, float, float]] = []
    for el in elevations:
        for az_idx in range(az_count):
            if len(beams) >= count:
                return beams
            # App convention: 0, 10, 20... degrees around the horizontal plane.
            beams.append((len(beams), 360.0 * az_idx / az_count, float(el)))
    return beams


def _plot_artifact(
    *,
    output_path: Path,
    mix_mean: np.ndarray,
    sample_rate: int,
    distance: np.ndarray,
    score_rows: list[dict],
    models: list[ModelSpec],
    beams: list[tuple[int, float, float]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_rows = 2 + len(models)
    fig, axes = plt.subplots(
        fig_rows,
        1,
        figsize=(16, 3.0 * fig_rows),
        constrained_layout=True,
    )
    if fig_rows == 1:
        axes = [axes]
    ax0 = axes[0]
    t = np.arange(distance.size) / float(sample_rate)
    ax0.plot(t, distance, color="black", linewidth=1.0)
    ax0.set_title(title)
    ax0.set_ylabel("distance m")
    ax0.grid(True, alpha=0.25)

    freqs, bins, spec = spectrogram(
        mix_mean,
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
        mode="magnitude",
    )
    ax1 = axes[1]
    db = 20.0 * np.log10(np.maximum(spec, 1e-8))
    im = ax1.pcolormesh(bins, freqs, db, shading="auto", cmap="magma")
    ax1.set_ylim(0, min(6000, sample_rate // 2))
    ax1.set_ylabel("Hz")
    ax1.set_title("Mean-channel mix spectrogram")
    fig.colorbar(im, ax=ax1, label="dB")

    beam_indices = [idx for idx, _, _ in beams]
    for model_idx, model in enumerate(models):
        ax = axes[2 + model_idx]
        rows = [r for r in score_rows if r["model"] == model.name]
        times = sorted({float(r["time_s"]) for r in rows})
        matrix = np.full((len(beam_indices), len(times)), np.nan, dtype=np.float32)
        time_to_col = {time_value: i for i, time_value in enumerate(times)}
        beam_to_row = {beam_idx: i for i, beam_idx in enumerate(beam_indices)}
        for row in rows:
            matrix[beam_to_row[int(row["beam_index"])], time_to_col[float(row["time_s"])]] = float(
                row["score"]
            )
        mesh = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[
                min(times, default=0.0),
                max(times, default=0.0),
                -0.5,
                len(beam_indices) - 0.5,
            ],
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax.set_ylabel("beam")
        ax.set_xlabel("time s")
        ax.set_title(f"{model.name}: score per MVDR beam")
        fig.colorbar(mesh, ax=ax, label="score")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_scores_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "model",
        "beam_index",
        "beam_azimuth_deg",
        "beam_elevation_deg",
        "time_s",
        "distance_m",
        "score",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(noise_path: str | None = None, drone_path: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-path",
        type=Path,
        default=Path(noise_path) if noise_path else DEFAULT_NOISE_DIR,
    )
    parser.add_argument(
        "--drone-path",
        type=Path,
        default=Path(drone_path) if drone_path else DEFAULT_DRONE_PATH,
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="name=/path/to.ckpt or /path/to.ckpt. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-duration-seconds", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--beam-count", type=int, default=36)
    parser.add_argument("--elevation-count", type=int, default=3)
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--max-elevation-deg", type=float, default=70.0)
    parser.add_argument("--drone-azimuth-deg", type=float, default=0.0)
    parser.add_argument("--drone-elevation-deg", type=float, default=10.0)
    parser.add_argument("--start-distance-m", type=float, default=1000.0)
    parser.add_argument("--approach-speed-mps", type=float, default=10.0)
    parser.add_argument("--min-distance-m", type=float, default=10.0)
    parser.add_argument("--reference-distance-m", type=float, default=10.0)
    parser.add_argument(
        "--reference-snr-db",
        type=float,
        default=5.0,
        help=(
            "Drone SNR at --reference-distance-m before distance loss, measured against "
            "the field background RMS. 5 dB makes 1000 m roughly -35 dB with a 10 m reference."
        ),
    )
    parser.add_argument("--stride-seconds", type=float, default=0.32)
    parser.add_argument("--noise-covariance-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _parse_models(args.checkpoint)
    beams = _build_beams(
        args.beam_count,
        args.elevation_count,
        args.min_elevation_deg,
        args.max_elevation_deg,
    )
    cfg = PyRoomSimulationConfig(
        drone_reference_distance_m=args.reference_distance_m,
        stft_n_fft=512,
        hop_length=160,
        diagonal_loading=1e-2,
        deglitch_audio=True,
    )
    positions = planar_array_positions()
    files = split_wavpack_files(args.noise_path, split=args.split, seed=42)
    if args.limit is not None:
        files = files[: args.limit]

    loaded_models: list[tuple[ModelSpec, torch.nn.Module, int]] = []
    for spec in models:
        model = load_model_from_checkpoint(spec.checkpoint, device="cpu", quiet=True)
        model.eval().to(args.device)
        clip_samples = int(round(float(model._clip_seconds) * int(model._mel_cfg.sample_rate)))
        loaded_models.append((spec, model, clip_samples))

    manifest = {
        "format": "audi_pyroom_attack_sim_v1",
        "noise_path": str(args.noise_path),
        "drone_path": str(args.drone_path),
        "split": args.split,
        "files": [str(path) for path in files],
        "models": [{"name": spec.name, "checkpoint": str(spec.checkpoint)} for spec in models],
        "beam_count": len(beams),
        "drone_azimuth_deg": args.drone_azimuth_deg,
        "drone_elevation_deg": args.drone_elevation_deg,
        "start_distance_m": args.start_distance_m,
        "approach_speed_mps": args.approach_speed_mps,
        "min_distance_m": args.min_distance_m,
        "reference_distance_m": args.reference_distance_m,
        "reference_snr_db": args.reference_snr_db,
        "stride_seconds": args.stride_seconds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_rows: list[dict] = []
    for file_idx, path in enumerate(tqdm(files, desc="attack files")):
        file_dir = output_dir / path.stem
        done_path = file_dir / "done.json"
        if args.skip_existing and done_path.exists():
            continue
        file_dir.mkdir(parents=True, exist_ok=True)

        bg = _ffmpeg_decode_full(path, target_sample_rate=args.sample_rate, expected_channels=16)
        if args.max_duration_seconds is not None:
            bg = bg[:, : int(round(args.max_duration_seconds * args.sample_rate))]
        length_samples = bg.shape[1]
        cov_len = min(length_samples, int(round(args.noise_covariance_seconds * args.sample_rate)))
        background_reference_rms = _rms(bg[:, :cov_len])
        drone_reference_amplitude = background_reference_rms * (
            10.0 ** (float(args.reference_snr_db) / 20.0)
        )
        drone = _load_drone_clip(
            args.drone_path,
            split=args.split,
            idx=file_idx,
            sample_rate=args.sample_rate,
        )
        drone_channels, distance = _moving_drone_channels(
            drone,
            positions_m=positions,
            sample_rate=args.sample_rate,
            length_samples=length_samples,
            azimuth_deg=args.drone_azimuth_deg,
            elevation_deg=args.drone_elevation_deg,
            start_distance_m=args.start_distance_m,
            speed_mps=args.approach_speed_mps,
            min_distance_m=args.min_distance_m,
            reference_distance_m=args.reference_distance_m,
            reference_amplitude=drone_reference_amplitude,
            speed_of_sound_mps=cfg.speed_of_sound_mps,
        )
        mix = (bg + drone_channels).astype(np.float32, copy=False)
        mix_mean = np.mean(mix, axis=0).astype(np.float32)
        sf.write(file_dir / f"{path.stem}_mix_mean.wav", peak_limit(mix_mean), args.sample_rate)

        beam_audio = _beamform_all(
            mix,
            bg[:, :cov_len],
            cfg=cfg,
            beams=beams,
            sample_rate=args.sample_rate,
            positions_m=positions,
        )

        score_rows: list[dict] = []
        best: dict[str, tuple[float, int]] = {}
        stride_samples = max(1, int(round(args.stride_seconds * args.sample_rate)))
        for spec, model, clip_samples in loaded_models:
            best_score = -1.0
            best_beam_idx = -1
            for beam_idx, az, el in tqdm(beams, desc=f"score {spec.name}", leave=False):
                times, scores = _windowed_scores(
                    model,
                    beam_audio[beam_idx],
                    sample_rate=args.sample_rate,
                    clip_samples=clip_samples,
                    stride_samples=stride_samples,
                    batch_size=args.batch_size,
                )
                if scores.size and float(np.max(scores)) > best_score:
                    best_score = float(np.max(scores))
                    best_beam_idx = int(beam_idx)
                for time_s, score in zip(times, scores, strict=True):
                    dist_idx = min(distance.size - 1, int(round(float(time_s) * args.sample_rate)))
                    score_rows.append(
                        {
                            "source_file": path.name,
                            "model": spec.name,
                            "beam_index": beam_idx,
                            "beam_azimuth_deg": round(float(az), 4),
                            "beam_elevation_deg": round(float(el), 4),
                            "time_s": round(float(time_s), 4),
                            "distance_m": round(float(distance[dist_idx]), 3),
                            "score": round(float(score), 6),
                        }
                    )
            best[spec.name] = (best_score, best_beam_idx)
            if best_beam_idx >= 0:
                sf.write(
                    file_dir / f"{path.stem}_{spec.name}_best_beam{best_beam_idx:02d}.wav",
                    peak_limit(beam_audio[best_beam_idx]),
                    args.sample_rate,
                )
                summary_rows.append(
                    {
                        "source_file": path.name,
                        "model": spec.name,
                        "best_score": round(best_score, 6),
                        "best_beam_index": best_beam_idx,
                    }
                )

        _write_scores_csv(file_dir / "scores.csv", score_rows)
        _plot_artifact(
            output_path=file_dir / f"{path.stem}_attack_scores.png",
            mix_mean=mix_mean,
            sample_rate=args.sample_rate,
            distance=distance,
            score_rows=score_rows,
            models=models,
            beams=beams,
            title=f"{path.name}: 1km approach at {args.approach_speed_mps:g} m/s",
        )
        done_path.write_text(
            json.dumps(
                {
                    "source_file": str(path),
                    "samples": int(length_samples),
                    "seconds": length_samples / args.sample_rate,
                    "background_reference_rms": background_reference_rms,
                    "drone_reference_amplitude": drone_reference_amplitude,
                    "reference_snr_db": args.reference_snr_db,
                    "snr_db": {
                        "first_window": _window_snr_db(
                            drone_channels,
                            bg,
                            sample_rate=args.sample_rate,
                            start_seconds=0.0,
                            duration_seconds=5.12,
                        ),
                        "after_start_arrival": _window_snr_db(
                            drone_channels,
                            bg,
                            sample_rate=args.sample_rate,
                            start_seconds=args.start_distance_m / cfg.speed_of_sound_mps,
                            duration_seconds=5.12,
                        ),
                    },
                    "best": best,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if summary_rows:
        with (output_dir / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["source_file", "model", "best_score", "best_beam_index"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote attack simulation artifacts to {output_dir}")
