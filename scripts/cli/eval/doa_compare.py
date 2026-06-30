"""Compare AUDI DOA algorithms on multichannel WAV windows."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_SRC = PROJECT_ROOT / "audi-app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

import doa_estimator as doa_mod  # noqa: E402

SHAHAR_MUSIC_METHOD = "shahar_music_triangle_3"


@dataclass(frozen=True)
class Method:
    name: str
    algorithm: str
    mic_indices: tuple[int, ...]
    legacy: bool = False


DEFAULT_METHODS = (
    Method(SHAHAR_MUSIC_METHOD, "ShaharMUSIC", (0, 7, 14), legacy=True),
    Method("music_triangle_3", "MUSIC", (0, 7, 14)),
    Method("normmusic_triangle_3", "NormMUSIC", (0, 7, 14)),
    Method("srp_triangle_3", "SRP-PHAT", (0, 7, 14)),
    Method("normmusic_perimeter_8", "NormMUSIC", (1, 3, 5, 7, 9, 12, 14, 15)),
    Method("srp_perimeter_8", "SRP-PHAT", (1, 3, 5, 7, 9, 12, 14, 15)),
    Method(
        "normmusic_healthy_13",
        "NormMUSIC",
        tuple(idx for idx in range(16) if idx not in {4, 8, 10}),
    ),
)


class StaticRingBuffer:
    def __init__(self, audio: np.ndarray):
        self.audio = audio

    def get_recent(self, samples: int, channel: int | None = None) -> np.ndarray:
        recent = self.audio[-samples:]
        if channel is None:
            return recent
        return recent[:, channel]


def run(noise_path: str | None = None, drone_path: str | None = None) -> None:
    del noise_path, drone_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_path",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "data" / "FPV_flyover_with_attacks.wav",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "doa_compare.csv",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-s", type=float, default=1.28)
    parser.add_argument("--context-padding-s", type=float, default=0.32)
    parser.add_argument(
        "--starts",
        default="165,210,260,320,380,440,500,560",
        help="Comma-separated window start times in seconds",
    )
    parser.add_argument("--azimuth-step-deg", type=float, default=1.0)
    args = parser.parse_args()

    methods = _filter_methods_for_channels(DEFAULT_METHODS, args.wav_path)
    rows: list[dict[str, Any]] = []
    for start_s in _parse_starts(args.starts):
        audio = _load_window(
            args.wav_path,
            start_s,
            args.window_s,
            args.context_padding_s,
            args.sample_rate,
        )
        for method in methods:
            row = estimate_method(
                audio=audio,
                sample_rate=args.sample_rate,
                start_s=start_s,
                window_s=args.window_s,
                context_padding_s=args.context_padding_s,
                azimuth_step_deg=args.azimuth_step_deg,
                method=method,
            )
            rows.append(row)

    _add_music_delta(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nSaved CSV: {args.output}")


def estimate_method(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    window_s: float,
    context_padding_s: float,
    azimuth_step_deg: float,
    method: Method,
) -> dict[str, Any]:
    config = _config_for(
        method,
        sample_rate,
        window_s,
        context_padding_s,
        azimuth_step_deg,
    )
    estimator = doa_mod.DOAEstimator(config, StaticRingBuffer(audio))
    settings = estimator._active_settings()
    active_mics = estimator._active_mic_indices(settings, estimator.config)
    selected = audio[:, list(active_mics)].T
    dominant = estimator._dominant_hps_frequency(selected[0], settings)
    base_row: dict[str, Any] = {
        "start_s": start_s,
        "window_s": window_s,
        "context_padding_s": context_padding_s,
        "analysis_window_s": doa_mod.analysis_window_s(settings),
        "method": method.name,
        "algorithm": method.algorithm,
        "mic_indices": " ".join(str(idx) for idx in active_mics),
        "mic_count": len(active_mics),
        "ok": False,
        "error": "",
        "azimuth_deg": "",
        "confidence": "",
        "dominant_frequency_hz": "",
        "peak_hps_snr_db": "",
        "music_frequencies_hz": "",
        "music_delta_deg": "",
    }
    if dominant is None:
        base_row["error"] = "no HPS peak"
        return base_row

    dominant_frequency_hz, peak_hps_snr_db = dominant
    stft_freqs, stft = estimator._stft_channels(selected, settings)
    harmonic_freqs = estimator._music_frequencies(stft_freqs, dominant_frequency_hz, settings)
    if not harmonic_freqs:
        base_row["error"] = "no usable frequency bins"
        return base_row

    azimuths = np.arange(-180.0, 180.0, settings.azimuth_step_deg)
    if method.legacy:
        covariance = estimator._covariance(stft, stft_freqs, harmonic_freqs, settings)
        spectrum = doa_mod.legacy_music_azimuth_spectrum(
            covariance,
            doa_mod.UMA16_MIC_POSITIONS_M[np.array(active_mics)],
            harmonic_freqs,
            azimuths,
            elevation_deg=settings.elevation_deg,
            n_sources=settings.n_sources,
        )
    else:
        spectrum = doa_mod.pyroom_azimuth_spectrum(
            stft,
            stft_freqs,
            active_mics,
            harmonic_freqs,
            azimuths,
            settings,
        )
    raw_azimuth = float(azimuths[int(np.argmax(spectrum))])
    smoothed = estimator._smooth_azimuth(raw_azimuth, spectrum, peak_hps_snr_db, settings)
    base_row.update(
        {
            "ok": True,
            "azimuth_deg": round(smoothed["azimuth_deg"], 2),
            "confidence": round(smoothed["confidence"], 3),
            "dominant_frequency_hz": round(float(dominant_frequency_hz), 1),
            "peak_hps_snr_db": round(float(peak_hps_snr_db), 2),
            "music_frequencies_hz": " ".join(f"{freq:.1f}" for freq in harmonic_freqs),
        }
    )
    return base_row


def _config_for(
    method: Method,
    sample_rate: int,
    window_s: float,
    context_padding_s: float,
    azimuth_step_deg: float,
) -> dict[str, Any]:
    return {
        "audio": {"sample_rate": sample_rate, "channels": 16},
        "doa": {
            "enabled": True,
            "active_profile": method.name,
            "profiles": {
                method.name: {
                    "mic_indices": list(method.mic_indices),
                    "music": {
                        "algorithm": "MUSIC" if method.legacy else method.algorithm,
                        "window_s": window_s,
                        "context_padding_s": context_padding_s,
                        "azimuth_step_deg": azimuth_step_deg,
                        "half_bins": 1,
                        "n_sources": 1,
                        "elevation_deg": 0.0,
                        "smoothing_predictions": 1,
                        "confidence_jump_deg": 45.0,
                    },
                },
            },
            "n_fft": 2048,
            "hop_length": 256,
            "hps": {
                "harmonics": 3,
                "fmin_hz": 100,
                "peak_search": {"fmin_hz": 100, "fmax_hz": 600},
                "cfar": {"guard_bins": 4, "ref_bins": 20},
            },
        },
    }


def _load_window(
    wav_path: Path,
    start_s: float,
    window_s: float,
    context_padding_s: float,
    sample_rate: int,
) -> np.ndarray:
    info = sf.info(wav_path)
    analysis_start_s = max(0.0, start_s - context_padding_s)
    analysis_window_s = window_s + 2.0 * context_padding_s
    start_frame = int(round(analysis_start_s * info.samplerate))
    frame_count = int(round(analysis_window_s * info.samplerate))
    audio, source_rate = sf.read(
        wav_path,
        start=start_frame,
        frames=frame_count,
        dtype="float32",
        always_2d=True,
    )
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        audio = signal.resample_poly(
            audio,
            up=sample_rate // divisor,
            down=source_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    return audio


def _filter_methods_for_channels(methods: tuple[Method, ...], wav_path: Path) -> tuple[Method, ...]:
    channels = sf.info(wav_path).channels
    return tuple(
        method for method in methods if max(method.mic_indices, default=-1) < channels
    )


def _parse_starts(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _add_music_delta(rows: list[dict[str, Any]]) -> None:
    by_start = {(row["start_s"], row["method"]): row for row in rows if row["ok"]}
    for row in rows:
        if not row["ok"] or row["method"] == SHAHAR_MUSIC_METHOD:
            continue
        baseline = by_start.get((row["start_s"], SHAHAR_MUSIC_METHOD))
        if baseline is None:
            continue
        row["music_delta_deg"] = round(
            doa_mod.angular_distance_deg(
                float(row["azimuth_deg"]),
                float(baseline["azimuth_deg"]),
            ),
            2,
        )


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("DOA comparison")
    print("method                     ok  mean_conf  mean_delta  azimuths")
    print("-------------------------  --  ---------  ----------  ----------------------")
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        ok_rows = [row for row in method_rows if row["ok"]]
        confidences = [float(row["confidence"]) for row in ok_rows]
        deltas = [float(row["music_delta_deg"]) for row in ok_rows if row["music_delta_deg"] != ""]
        azimuths = " ".join(str(row["azimuth_deg"]) for row in ok_rows)
        mean_conf = np.mean(confidences) if confidences else float("nan")
        mean_delta = np.mean(deltas) if deltas else float("nan")
        print(
            f"{method:<25} {len(ok_rows):>2}/{len(method_rows):<2} "
            f"{mean_conf:>9.3f} {mean_delta:>10.2f}  {azimuths}"
        )
