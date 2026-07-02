"""Pyroomacoustics-based 16-mic dataset generation for detector training."""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from scipy.signal import istft, resample_poly, stft
from torch.utils.data import Dataset

from audi.augment import _fit_length, _rms, highpass
from audi.config import SNRBin
from audi.training.dataset import BatchItem, _finite_audio, _rms_normalize, app_window_normalize

UMA16_MIC_POSITIONS_M = np.array(
    [
        [-0.021, -0.063, 0.0],
        [-0.063, -0.063, 0.0],
        [-0.021, -0.021, 0.0],
        [-0.063, -0.021, 0.0],
        [-0.021, 0.021, 0.0],
        [-0.063, 0.021, 0.0],
        [-0.021, 0.063, 0.0],
        [-0.063, 0.063, 0.0],
        [0.063, 0.063, 0.0],
        [0.021, 0.063, 0.0],
        [0.063, 0.021, 0.0],
        [0.021, 0.021, 0.0],
        [0.063, -0.021, 0.0],
        [0.021, -0.021, 0.0],
        [0.063, -0.063, 0.0],
        [0.021, -0.063, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PyRoomSimulationConfig:
    """Configuration for synthetic UMA16 pyroom rendering."""

    rows: int = 4
    cols: int = 4
    spacing_m: float = 0.042
    speed_of_sound_mps: float = 343.0
    min_azimuth_deg: float = -180.0
    max_azimuth_deg: float = 180.0
    min_elevation_deg: float = 5.0
    max_elevation_deg: float = 85.0
    min_distance_m: float = 20.0
    max_distance_m: float = 450.0
    drone_reference_distance_m: float = 10.0
    steering_error_deg: float = 0.0
    stft_n_fft: int = 2048
    hop_length: int = 512
    diagonal_loading: float = 1e-4
    sensor_noise_db: float | None = -45.0
    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    air_absorption: bool = True
    random_beam_probability: float = 0.0
    soft_target_by_beam_alignment: bool = False
    target_alignment_floor: float = 0.5
    beamformer: str = "mvdr"
    deglitch_audio: bool = True
    deglitch_threshold: float = 0.001
    deglitch_loudness_ratio: float = 8.0
    deglitch_diff_ratio: float = 12.0
    deglitch_window_samples: int = 64
    mvdr_cache_dir: str | None = None
    mvdr_cache_seconds: float = 30.0
    spatial_bg_probability: float = 0.25
    spatial_bg_multi_probability: float = 0.5
    spatial_bg_count: int = 3
    spatial_bg_max_attenuation_db: float = -40.0

    def __post_init__(self) -> None:
        if self.rows != 4 or self.cols != 4 or not np.isclose(self.spacing_m, 0.042):
            raise ValueError("MVDR training simulation must use UMA16 4x4 / 42 mm geometry")
        if self.stft_n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("stft_n_fft and hop_length must be positive")
        if self.min_azimuth_deg > self.max_azimuth_deg:
            raise ValueError("min_azimuth_deg must be <= max_azimuth_deg")
        if self.min_elevation_deg > self.max_elevation_deg:
            raise ValueError("min_elevation_deg must be <= max_elevation_deg")
        if self.min_distance_m <= 0.0 or self.min_distance_m > self.max_distance_m:
            raise ValueError("distance range must be positive and ordered")
        if self.drone_reference_distance_m <= 0.0:
            raise ValueError("drone_reference_distance_m must be positive")
        if not 0.0 <= self.random_beam_probability <= 1.0:
            raise ValueError("random_beam_probability must be in [0, 1]")
        if not 0.0 <= self.target_alignment_floor <= 1.0:
            raise ValueError("target_alignment_floor must be in [0, 1]")
        if self.beamformer not in {"mvdr", "mean", "channel0", "random-channel"}:
            raise ValueError(
                "beamformer must be one of: mvdr, mean, channel0, random-channel"
            )
        if self.beamformer != "mvdr" and (
            self.soft_target_by_beam_alignment
            or self.random_beam_probability > 0.0
            or self.steering_error_deg > 0.0
        ):
            raise ValueError("beam calibration options require beamformer='mvdr'")
        if self.deglitch_threshold <= 0.0:
            raise ValueError("deglitch_threshold must be positive")
        if self.deglitch_loudness_ratio <= 0.0:
            raise ValueError("deglitch_loudness_ratio must be positive")
        if self.deglitch_diff_ratio <= 0.0:
            raise ValueError("deglitch_diff_ratio must be positive")
        if self.deglitch_window_samples < 0:
            raise ValueError("deglitch_window_samples must be non-negative")
        if self.mvdr_cache_seconds <= 0.0:
            raise ValueError("mvdr_cache_seconds must be positive")
        if not 0.0 <= self.spatial_bg_probability <= 1.0:
            raise ValueError("spatial_bg_probability must be in [0, 1]")
        if not 0.0 <= self.spatial_bg_multi_probability <= 1.0:
            raise ValueError("spatial_bg_multi_probability must be in [0, 1]")
        if self.spatial_bg_count < 0:
            raise ValueError("spatial_bg_count must be non-negative")


@dataclass(frozen=True)
class WavpackInfo:
    """Decoded audio metadata for one ffmpeg-readable file."""

    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float


@dataclass(frozen=True)
class NoiseSection:
    """Decoded and normalized background audio section."""

    info: WavpackInfo
    channels: np.ndarray
    start_seconds: float


def pyroom_config_payload(cfg: PyRoomSimulationConfig) -> dict[str, Any]:
    """Return a JSON-stable pyroom simulation config payload."""
    payload = asdict(cfg)
    payload["output"] = "full_band_mono_detector_waveform"
    payload["simulator"] = "pyroomacoustics.AnechoicRoom"
    payload["array_geometry"] = "uma16_app_channel_order"
    payload["deglitch_algorithm"] = "app_side_window_prefix_v1"
    payload["mic_positions_m"] = UMA16_MIC_POSITIONS_M.tolist()
    payload["background_source"] = "real_16_channel_wavpack"
    return payload


def pyroom_config_hash(cfg: PyRoomSimulationConfig) -> str:
    """Return a stable hash for the pyroom simulation parameters."""
    import hashlib

    encoded = json.dumps(
        pyroom_config_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def planar_array_positions(
    *, rows: int = 4, cols: int = 4, spacing_m: float = 0.042
) -> np.ndarray:
    """Return UMA16 mic positions in the deployed app's channel order."""
    if rows != 4 or cols != 4 or not np.isclose(spacing_m, 0.042):
        raise ValueError("MVDR simulation must use the deployed UMA16 4x4 / 42 mm geometry")
    return UMA16_MIC_POSITIONS_M.copy()


def direction_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert azimuth/elevation degrees into a unit vector.

    Azimuth follows the DOA app's compass convention: 0 is +y, 90 is +x.
    Elevation is measured up from the array plane; +90 degrees is broadside.
    """
    az = math.radians(90.0 - float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    unit = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    norm = np.linalg.norm(unit)
    if norm <= 0.0:
        raise ValueError("direction vector has zero norm")
    return unit / norm


def sample_direction(cfg: PyRoomSimulationConfig) -> tuple[float, float, float]:
    """Sample a source direction and distance from the configured range."""
    return (
        random.uniform(cfg.min_azimuth_deg, cfg.max_azimuth_deg),
        random.uniform(cfg.min_elevation_deg, cfg.max_elevation_deg),
        random.uniform(cfg.min_distance_m, cfg.max_distance_m),
    )


def jitter_direction(
    azimuth_deg: float, elevation_deg: float, max_error_deg: float
) -> tuple[float, float]:
    """Apply bounded steering jitter to an azimuth/elevation pair."""
    if max_error_deg <= 0.0:
        return azimuth_deg, elevation_deg
    return (
        azimuth_deg + random.uniform(-max_error_deg, max_error_deg),
        min(89.0, max(0.1, elevation_deg + random.uniform(-max_error_deg, max_error_deg))),
    )


def beam_alignment(
    drone_azimuth_deg: float,
    drone_elevation_deg: float,
    beam_azimuth_deg: float,
    beam_elevation_deg: float,
) -> float:
    """Return cosine similarity between drone and beam look vectors."""
    drone_vec = direction_unit_vector(drone_azimuth_deg, drone_elevation_deg)
    beam_vec = direction_unit_vector(beam_azimuth_deg, beam_elevation_deg)
    return float(np.clip(np.dot(drone_vec, beam_vec), -1.0, 1.0))


def soft_target_from_alignment(
    drone_azimuth_deg: float,
    drone_elevation_deg: float,
    beam_azimuth_deg: float,
    beam_elevation_deg: float,
    *,
    floor: float = 0.5,
) -> float:
    """Scale a positive classification target by beam/drone alignment."""
    return float(max(beam_alignment(
        drone_azimuth_deg,
        drone_elevation_deg,
        beam_azimuth_deg,
        beam_elevation_deg,
    ), floor))


def split_wavpack_files(
    data_dir: Path,
    *,
    split: str,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> list[Path]:
    """Return a stable train/validation/test split of WavPack files."""
    files = sorted(Path(data_dir).glob("*.wv"))
    if not files:
        raise SystemExit(f"No .wv files found in {data_dir}")
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * validation_fraction)))
    if split == "validation":
        return sorted(shuffled[:holdout_count])
    if split == "test":
        start = holdout_count
        end = min(len(shuffled), start + holdout_count)
        return sorted(shuffled[start:end] or shuffled[:holdout_count])
    if split == "train":
        return sorted(shuffled[holdout_count * 2 :] or shuffled)
    raise SystemExit(f"Unsupported split for WavPack noise dataset: {split!r}")


def probe_audio_file(path: Path) -> WavpackInfo:
    """Read audio metadata with ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream found in {path}")
    stream = streams[0]
    return WavpackInfo(
        path=path,
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        duration_seconds=float(stream["duration"]),
    )


def load_multichannel_segment(
    info: WavpackInfo,
    *,
    start_seconds: float,
    length_samples: int,
    target_sample_rate: int,
    expected_channels: int = 16,
) -> np.ndarray:
    """Decode one multichannel segment through ffmpeg as ``[channels, samples]``."""
    duration = float(length_samples) / float(target_sample_rate)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, start_seconds):.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(info.path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Decoded empty audio from {info.path}")
    if audio.size % info.channels != 0:
        audio = audio[: audio.size - (audio.size % info.channels)]
    audio = audio.reshape(-1, info.channels).T
    if info.channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels in {info.path}, got {info.channels}"
        )
    if info.sample_rate != target_sample_rate:
        common = gcd(info.sample_rate, target_sample_rate)
        audio = resample_poly(
            audio.astype(np.float64),
            target_sample_rate // common,
            info.sample_rate // common,
            axis=1,
        ).astype(np.float32)
    if audio.shape[1] < length_samples:
        repeats = int(np.ceil(length_samples / audio.shape[1]))
        audio = np.tile(audio, (1, repeats))
    return _finite_audio(audio[:, :length_samples]).reshape(expected_channels, length_samples)


def preprocess_noise_channels(
    channels: np.ndarray,
    *,
    cfg: PyRoomSimulationConfig,
    highpass_hz: float,
    sample_rate: int,
) -> np.ndarray:
    """Apply the same background cleanup used by generated samples."""
    if cfg.deglitch_audio:
        channels = deglitch_multichannel(
            channels,
            threshold=cfg.deglitch_threshold,
            loudness_ratio=cfg.deglitch_loudness_ratio,
            diff_ratio=cfg.deglitch_diff_ratio,
            window_samples=cfg.deglitch_window_samples,
        )
    rms = [_rms(ch) for ch in channels]
    median_rms = float(np.median(rms))
    if median_rms > 1e-8:
        channels = channels / np.float32(median_rms)
    if highpass_hz > 0.0:
        channels = np.stack(
            [
                highpass(ch, cutoff_hz=highpass_hz, sample_rate=sample_rate)
                for ch in channels
            ],
            axis=0,
        )
    return _finite_audio(channels).reshape(channels.shape)


def deglitch_multichannel(
    channels: np.ndarray,
    *,
    threshold: float = 0.001,
    loudness_ratio: float = 8.0,
    diff_ratio: float = 12.0,
    window_samples: int = 64,
) -> np.ndarray:
    """Repair short discontinuity jumps with local interpolation."""
    repaired = _finite_audio(channels).reshape(channels.shape).copy()
    if repaired.shape[-1] < 3:
        return repaired
    for ch_idx in range(repaired.shape[0]):
        y = repaired[ch_idx]
        diff = np.abs(np.diff(y))
        if window_samples > 0:
            abs_y = np.abs(y)
            abs_prefix = _prefix_sum(abs_y)
            diff_prefix = _prefix_sum(diff)
            loud_left = _trailing_mean_inclusive(
                abs_prefix,
                radius=window_samples,
                output_count=diff.size,
            )
            loud_right = _following_mean(
                abs_prefix,
                radius=window_samples,
                output_count=diff.size,
                offset=1,
            )
            diff_left = _preceding_mean(
                diff_prefix,
                radius=window_samples,
                output_count=diff.size,
            )
            diff_right = _following_mean(
                diff_prefix,
                radius=window_samples,
                output_count=diff.size,
                offset=1,
            )
            local_loudness = np.minimum(loud_left, loud_right) + 1e-8
            local_diff_scale = np.maximum(diff_left, diff_right) + 1e-8
        else:
            local_loudness = np.maximum(np.abs(y[1:]), 1e-8)
            local_diff_scale = diff + 1e-8
        jump_mask = (
            (diff > threshold)
            & (diff > local_loudness * loudness_ratio)
            & (diff > local_diff_scale * diff_ratio)
        )
        jumps = np.flatnonzero(jump_mask).astype(np.int64, copy=False)
        if jumps.size == 0:
            continue
        spans: list[tuple[int, int]] = []
        for jump in jumps:
            start = max(0, int(jump) - window_samples)
            end = min(y.size, int(jump) + 2 + window_samples)
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], end))
            else:
                spans.append((start, end))
        for start, end in spans:
            left = y[start - 1] if start > 0 else y[end] if end < y.size else 0.0
            right = y[end] if end < y.size else left
            y[start:end] = np.linspace(
                left,
                right,
                end - start,
                endpoint=False,
                dtype=np.float32,
            )
    return repaired


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    prefix = np.empty(values.size + 1, dtype=np.float32)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float32, out=prefix[1:])
    return prefix


def _trailing_mean_inclusive(
    prefix: np.ndarray,
    *,
    radius: int,
    output_count: int,
) -> np.ndarray:
    values_size = prefix.size - 1
    output_count = max(0, min(int(output_count), values_size))
    out = np.empty(output_count, dtype=np.float32)
    if output_count == 0:
        return out
    radius = max(1, int(radius))
    partial = min(radius - 1, output_count)
    if partial > 0:
        counts = np.arange(1, partial + 1, dtype=np.float32)
        out[:partial] = prefix[1 : partial + 1] / counts
    if output_count >= radius:
        out[radius - 1 :] = (
            prefix[radius : output_count + 1]
            - prefix[: output_count + 1 - radius]
        ) / float(radius)
    return out


def _preceding_mean(
    prefix: np.ndarray,
    *,
    radius: int,
    output_count: int,
) -> np.ndarray:
    values_size = prefix.size - 1
    output_count = max(0, min(int(output_count), values_size))
    out = np.empty(output_count, dtype=np.float32)
    if output_count == 0:
        return out
    radius = max(1, int(radius))
    out[0] = 0.0
    partial = min(radius, output_count - 1)
    if partial > 0:
        counts = np.arange(1, partial + 1, dtype=np.float32)
        out[1 : partial + 1] = prefix[1 : partial + 1] / counts
    if output_count > radius:
        out[radius:] = (
            prefix[radius:output_count]
            - prefix[: output_count - radius]
        ) / float(radius)
    return out


def _following_mean(
    prefix: np.ndarray,
    *,
    radius: int,
    output_count: int,
    offset: int,
) -> np.ndarray:
    values_size = prefix.size - 1
    output_count = max(0, int(output_count))
    out = np.empty(output_count, dtype=np.float32)
    if output_count == 0:
        return out
    radius = max(1, int(radius))
    offset = max(0, int(offset))
    full_count = max(0, min(output_count, values_size - offset - radius + 1))
    if full_count > 0:
        out[:full_count] = (
            prefix[offset + radius : offset + radius + full_count]
            - prefix[offset : offset + full_count]
        ) / float(radius)
    if full_count < output_count:
        starts = np.arange(full_count + offset, output_count + offset)
        starts = np.minimum(starts, values_size)
        counts = np.maximum(values_size - starts, 1).astype(np.float32)
        out[full_count:] = (prefix[values_size] - prefix[starts]) / counts
    return out


def load_hf_audio_array(audio: Any, *, target_sample_rate: int) -> np.ndarray:
    """Decode a Hugging Face Audio value across old and new decoder APIs."""
    if isinstance(audio, dict):
        array = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
        sample_rate = int(audio.get("sampling_rate", target_sample_rate))
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = np.asarray(samples.data, dtype=np.float32).reshape(-1)
        sample_rate = int(samples.sample_rate)
    else:
        raise TypeError(f"Unsupported audio value: {type(audio)!r}")
    if sample_rate == target_sample_rate:
        return _finite_audio(array)
    common = gcd(sample_rate, target_sample_rate)
    return _finite_audio(
        resample_poly(
            array.astype(np.float64),
            target_sample_rate // common,
            sample_rate // common,
        )
    )


def load_hf_split(dataset_path: Path, split: str, *, name: str) -> Any:
    """Load one split from an HF DatasetDict."""
    dd = load_from_disk(str(dataset_path))
    if split not in dd:
        if split == "test" and "validation" in dd:
            return dd["validation"]
        raise SystemExit(f"Split {split!r} not found in {name} dataset {dataset_path}")
    return dd[split]


def load_drone_split(drone_path: Path, split: str) -> Any:
    """Load the current mono drone HF dataset split."""
    return load_hf_split(drone_path, split, name="drone")


def calculate_snr_db_from_mix_and_background(
    mix_channels: np.ndarray,
    background_channels: np.ndarray,
) -> float:
    """Measure SNR from the rendered mix and its clean background channels."""
    signal_channels = _finite_audio(mix_channels - background_channels).reshape(
        background_channels.shape
    )
    signal_rms = _rms(signal_channels.reshape(-1))
    background_rms = _rms(background_channels.reshape(-1))
    if signal_rms <= 1e-8 or background_rms <= 1e-8:
        return -99.9
    return float(20.0 * math.log10(signal_rms / background_rms))


def add_sensor_noise(
    channels: np.ndarray, *, sensor_noise_db: float | None
) -> np.ndarray:
    """Add independent per-mic sensor noise at a level relative to array RMS."""
    if sensor_noise_db is None:
        return channels.astype(np.float32, copy=False)
    rms = _rms(channels.reshape(-1))
    if rms <= 1e-8:
        return channels.astype(np.float32, copy=False)
    noise_rms = rms * (10.0 ** (float(sensor_noise_db) / 20.0))
    noise = np.random.standard_normal(channels.shape).astype(np.float32) * noise_rms
    return _finite_audio(channels + noise).reshape(channels.shape)


def simulate_drone_free_space(
    drone: np.ndarray,
    *,
    positions_m: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float,
    reference_distance_m: float = 1.0,
    sample_rate: int,
    cfg: PyRoomSimulationConfig,
    target_length: int,
) -> np.ndarray:
    """Render a mono source into the 16-mic array using pyroomacoustics.

    Pyroom applies free-space distance loss from a source signal. Our drone
    recordings are microphone signals captured at roughly 10-20 m, so drone
    renders pass that recording distance as ``reference_distance_m`` to make the
    simulated transfer relative to the recording point.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise SystemExit(
            "pyroomacoustics is required for PyRoomDataset simulation. "
            "Run through the project environment, for example `uv run audi-data ...`."
        ) from exc

    direction = direction_unit_vector(azimuth_deg, elevation_deg)
    source_position = direction * float(distance_m)
    room = pra.AnechoicRoom(
        dim=3,
        fs=sample_rate,
        temperature=cfg.temperature_c,
        humidity=cfg.humidity_percent,
        air_absorption=cfg.air_absorption,
    )
    room.add_microphone_array(pra.MicrophoneArray(positions_m.T, fs=sample_rate))
    room.add_source(
        source_position,
        signal=_finite_audio(drone) * np.float32(reference_distance_m),
    )
    premix = room.simulate(return_premix=True)
    rendered = np.asarray(premix[0], dtype=np.float32)
    crop_start = max(0, int((float(distance_m) / cfg.speed_of_sound_mps) * sample_rate))
    crop_end = crop_start + target_length
    if rendered.shape[1] < crop_end:
        rendered = np.pad(rendered, ((0, 0), (0, crop_end - rendered.shape[1])))
    return _finite_audio(rendered[:, crop_start:crop_end]).reshape(
        positions_m.shape[0], target_length
    )


def _stft_channels(
    channels: np.ndarray, *, sample_rate: int, n_fft: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, spec = stft(
        np.asarray(channels, dtype=np.float32),
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    return freqs.astype(np.float64), spec


def estimate_mvdr_inverse_covariance(
    noise_channels: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    diagonal_loading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-frequency inverse covariance from background audio."""
    freqs, noise_spec = _stft_channels(
        noise_channels,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    num_mics = noise_spec.shape[0]
    inverse_covariance = np.zeros(
        (freqs.shape[0], num_mics, num_mics),
        dtype=np.complex128,
    )
    identity = np.eye(num_mics, dtype=np.complex128)

    for freq_idx in range(freqs.shape[0]):
        x = noise_spec[:, freq_idx, :].astype(np.complex128, copy=False)
        covariance = (x @ x.conj().T) / max(1, x.shape[1])
        power = float(np.real(np.trace(covariance)) / max(1, num_mics))
        load = max(power * diagonal_loading, 1e-8)
        loaded = covariance + load * identity
        try:
            inverse_covariance[freq_idx] = np.linalg.inv(loaded)
        except np.linalg.LinAlgError:
            inverse_covariance[freq_idx] = np.linalg.pinv(loaded)
    return freqs, inverse_covariance


def mvdr_weights_from_inverse_covariance(
    freqs: np.ndarray,
    inverse_covariance: np.ndarray,
    *,
    positions_m: np.ndarray,
    look_direction: np.ndarray,
    speed_of_sound_mps: float,
) -> np.ndarray:
    """Project cached inverse covariance into MVDR weights for one beam."""
    num_mics = inverse_covariance.shape[1]
    delays_sec = -(positions_m @ look_direction) / float(speed_of_sound_mps)
    steering = np.exp(-2j * np.pi * freqs[:, None] * delays_sec[None, :])
    solved = np.einsum("fmn,fn->fm", inverse_covariance, steering, optimize=True)
    denom = np.einsum("fm,fm->f", steering.conj(), solved, optimize=True)
    weights = np.empty_like(solved)
    fallback = np.ones(num_mics, dtype=np.complex128) / num_mics
    valid = np.abs(denom) > 1e-12
    weights[valid] = solved[valid] / denom[valid, None]
    weights[~valid] = fallback
    return weights


def estimate_mvdr_weights(
    noise_channels: np.ndarray,
    *,
    positions_m: np.ndarray,
    look_direction: np.ndarray,
    sample_rate: int,
    speed_of_sound_mps: float,
    n_fft: int,
    hop_length: int,
    diagonal_loading: float,
) -> np.ndarray:
    """Estimate full-band MVDR weights from background covariance."""
    freqs, noise_spec = _stft_channels(
        noise_channels,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    num_mics = noise_spec.shape[0]
    inverse_covariance = np.zeros(
        (freqs.shape[0], num_mics, num_mics),
        dtype=np.complex128,
    )
    identity = np.eye(num_mics, dtype=np.complex128)

    for freq_idx in range(freqs.shape[0]):
        x = noise_spec[:, freq_idx, :].astype(np.complex128, copy=False)
        covariance = (x @ x.conj().T) / max(1, x.shape[1])
        power = float(np.real(np.trace(covariance)) / max(1, num_mics))
        load = max(power * diagonal_loading, 1e-8)
        loaded = covariance + load * identity
        try:
            inverse_covariance[freq_idx] = np.linalg.inv(loaded)
        except np.linalg.LinAlgError:
            inverse_covariance[freq_idx] = np.linalg.pinv(loaded)
    return mvdr_weights_from_inverse_covariance(
        freqs,
        inverse_covariance,
        positions_m=positions_m,
        look_direction=look_direction,
        speed_of_sound_mps=speed_of_sound_mps,
    )


def apply_mvdr_weights(
    channels: np.ndarray,
    weights: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    target_length: int,
) -> np.ndarray:
    """Apply previously estimated MVDR weights to a multichannel waveform."""
    _, spec = _stft_channels(
        channels,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    beam_spec = np.einsum("fm,mft->ft", weights.conj(), spec, optimize=True)
    _, beam = istft(
        beam_spec,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )
    return _finite_audio(beam[:target_length])


def mvdr_cache_payload(
    info: WavpackInfo,
    *,
    cfg: PyRoomSimulationConfig,
    highpass_hz: float,
    sample_rate: int,
    cache_seconds: float,
    cache_start_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return a JSON-stable identity for one per-file MVDR covariance cache."""
    stat = info.path.stat()
    return {
        "format": "audi_pyroom_mvdr_inverse_covariance_v2",
        "source_path": str(info.path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sample_rate": info.sample_rate,
        "source_channels": info.channels,
        "sample_rate": int(sample_rate),
        "expected_channels": cfg.rows * cfg.cols,
        "stft_n_fft": cfg.stft_n_fft,
        "hop_length": cfg.hop_length,
        "diagonal_loading": cfg.diagonal_loading,
        "highpass_hz": float(highpass_hz),
        "deglitch_audio": cfg.deglitch_audio,
        "deglitch_threshold": cfg.deglitch_threshold,
        "deglitch_loudness_ratio": cfg.deglitch_loudness_ratio,
        "deglitch_diff_ratio": cfg.deglitch_diff_ratio,
        "deglitch_window_samples": cfg.deglitch_window_samples,
        "deglitch_algorithm": "app_side_window_prefix_v1",
        "cache_seconds": float(cache_seconds),
        "cache_start_seconds": float(cache_start_seconds),
    }


def mvdr_cache_hash(payload: dict[str, Any]) -> str:
    """Return the stable hash for an MVDR covariance cache payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def mvdr_cache_path(cache_dir: Path, info: WavpackInfo, payload: dict[str, Any]) -> Path:
    """Return the cache filename for one background file and parameter set."""
    digest = mvdr_cache_hash(payload)[:16]
    return cache_dir / f"{info.path.stem}.{digest}.mvdr.npz"


def write_mvdr_cache_file(
    info: WavpackInfo,
    *,
    cache_dir: Path,
    cfg: PyRoomSimulationConfig,
    highpass_hz: float,
    sample_rate: int,
    cache_seconds: float,
    cache_start_seconds: float = 0.0,
) -> Path:
    """Precompute and persist one per-file inverse covariance cache."""
    payload = mvdr_cache_payload(
        info,
        cfg=cfg,
        highpass_hz=highpass_hz,
        sample_rate=sample_rate,
        cache_seconds=cache_seconds,
        cache_start_seconds=cache_start_seconds,
    )
    path = mvdr_cache_path(cache_dir, info, payload)
    if path.exists():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    length_samples = max(1, int(round(float(cache_seconds) * sample_rate)))
    channels = load_multichannel_segment(
        info,
        start_seconds=cache_start_seconds,
        length_samples=length_samples,
        target_sample_rate=sample_rate,
        expected_channels=cfg.rows * cfg.cols,
    )
    channels = preprocess_noise_channels(
        channels,
        cfg=cfg,
        highpass_hz=highpass_hz,
        sample_rate=sample_rate,
    )
    freqs, inverse_covariance = estimate_mvdr_inverse_covariance(
        channels,
        sample_rate=sample_rate,
        n_fft=cfg.stft_n_fft,
        hop_length=cfg.hop_length,
        diagonal_loading=cfg.diagonal_loading,
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(
            fh,
            freqs=freqs.astype(np.float64),
            inverse_covariance=inverse_covariance.astype(np.complex64),
            payload_json=np.array(json.dumps(payload, sort_keys=True)),
        )
    tmp_path.replace(path)
    return path


class MvdrCovarianceCache:
    """Lazy loader for per-background MVDR inverse covariance caches."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        cfg: PyRoomSimulationConfig,
        highpass_hz: float,
        sample_rate: int,
        positions_m: np.ndarray,
        create_missing: bool = True,
        cache_start_seconds: float = 0.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        self.highpass_hz = float(highpass_hz)
        self.sample_rate = int(sample_rate)
        self.positions_m = np.asarray(positions_m, dtype=np.float64)
        self.create_missing = bool(create_missing)
        self.cache_start_seconds = float(cache_start_seconds)
        self._loaded: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def ensure(self, info: WavpackInfo) -> Path:
        """Ensure a cache file exists for ``info`` and return its path."""
        path = mvdr_cache_path(
            self.cache_dir,
            info,
            mvdr_cache_payload(
                info,
                cfg=self.cfg,
                highpass_hz=self.highpass_hz,
                sample_rate=self.sample_rate,
                cache_seconds=self.cfg.mvdr_cache_seconds,
                cache_start_seconds=self.cache_start_seconds,
            ),
        )
        if path.exists():
            return path
        if not self.create_missing:
            raise FileNotFoundError(
                f"Missing MVDR cache for {info.path}. Build it with "
                "`uv run audi-data pyroom-mvdr-cache ...`."
            )
        return write_mvdr_cache_file(
            info,
            cache_dir=self.cache_dir,
            cfg=self.cfg,
            highpass_hz=self.highpass_hz,
            sample_rate=self.sample_rate,
            cache_seconds=self.cfg.mvdr_cache_seconds,
            cache_start_seconds=self.cache_start_seconds,
        )

    def weights_for_file(
        self,
        info: WavpackInfo,
        *,
        look_direction: np.ndarray,
    ) -> np.ndarray:
        """Load cached covariance for ``info`` and return weights for one beam."""
        path = self.ensure(info)
        if path not in self._loaded:
            with np.load(path, allow_pickle=False) as data:
                self._loaded[path] = (
                    data["freqs"].astype(np.float64, copy=False),
                    data["inverse_covariance"].astype(np.complex128, copy=False),
                )
        freqs, inverse_covariance = self._loaded[path]
        return mvdr_weights_from_inverse_covariance(
            freqs,
            inverse_covariance,
            positions_m=self.positions_m,
            look_direction=look_direction,
            speed_of_sound_mps=self.cfg.speed_of_sound_mps,
        )


class PyRoomDataset(Dataset[tuple[torch.Tensor, ...]]):
    """File-backed 16-channel background plus pyroomacoustics drone simulation."""

    def __init__(
        self,
        *,
        noise_dir: Path,
        drone_path: Path,
        split: str,
        snr_bins: list[SNRBin],
        target_length_samples: int,
        positive_probability: float,
        highpass_hz: float,
        sample_rate: int,
        cfg: PyRoomSimulationConfig,
        length: int,
        return_components: bool = False,
        validation_fraction: float = 0.15,
        file_split_seed: int = 42,
        spatial_bg_path: Path | None = None,
    ) -> None:
        if not snr_bins:
            raise ValueError("snr_bins must not be empty")
        self.noise_files = [
            probe_audio_file(path)
            for path in split_wavpack_files(
                noise_dir,
                split=split,
                validation_fraction=validation_fraction,
                seed=file_split_seed,
            )
        ]
        self.drone_ds = load_drone_split(drone_path, split)
        self.spatial_bg_ds = (
            load_hf_split(spatial_bg_path, split, name="spatial background")
            if spatial_bg_path is not None and cfg.spatial_bg_probability > 0.0
            else None
        )
        self.snr_bins = list(snr_bins)
        self.target_length_samples = int(target_length_samples)
        self.positive_probability = float(positive_probability)
        self.highpass_hz = float(highpass_hz)
        self.sample_rate = int(sample_rate)
        self.cfg = cfg
        self.length = int(length)
        self.return_components = return_components
        self.positions_m = planar_array_positions(
            rows=cfg.rows,
            cols=cfg.cols,
            spacing_m=cfg.spacing_m,
        )
        self.mvdr_cache = (
            MvdrCovarianceCache(
                cache_dir=Path(cfg.mvdr_cache_dir),
                cfg=cfg,
                highpass_hz=self.highpass_hz,
                sample_rate=self.sample_rate,
                positions_m=self.positions_m,
                create_missing=True,
            )
            if cfg.beamformer == "mvdr" and cfg.mvdr_cache_dir
            else None
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        del idx
        mvdr_noise_section = (
            self._load_noise_section()
            if self.cfg.beamformer == "mvdr" and getattr(self, "mvdr_cache", None) is None
            else None
        )
        output_noise_section = self._load_noise_section()
        output_noise_channels = self._augment_background_channels(
            output_noise_section.channels
        )
        is_positive = random.random() < self.positive_probability

        if is_positive:
            drone_az, drone_el, drone_distance = sample_direction(self.cfg)
            drone = self._load_drone()
            drone_channels = simulate_drone_free_space(
                drone,
                positions_m=self.positions_m,
                azimuth_deg=drone_az,
                elevation_deg=drone_el,
                distance_m=drone_distance,
                reference_distance_m=self.cfg.drone_reference_distance_m,
                sample_rate=self.sample_rate,
                cfg=self.cfg,
                target_length=self.target_length_samples,
            )
            if self.cfg.beamformer != "mvdr":
                beam_az = beam_el = float("nan")
            elif random.random() < self.cfg.random_beam_probability:
                beam_az, beam_el, _ = sample_direction(self.cfg)
            else:
                beam_az, beam_el = jitter_direction(
                    drone_az,
                    drone_el,
                    self.cfg.steering_error_deg,
                )
            target_scale = (
                soft_target_from_alignment(
                    drone_az,
                    drone_el,
                    beam_az,
                    beam_el,
                    floor=self.cfg.target_alignment_floor,
                )
                if self.cfg.soft_target_by_beam_alignment
                else 1.0
            )
            label = torch.tensor(target_scale, dtype=torch.float32)
        else:
            snr_db = -99.9
            drone_az = drone_el = drone_distance = float("nan")
            drone_channels = np.zeros_like(output_noise_channels, dtype=np.float32)
            if self.cfg.beamformer == "mvdr":
                beam_az, beam_el, _ = sample_direction(self.cfg)
            else:
                beam_az = beam_el = float("nan")
            target_scale = float("nan")
            label = torch.tensor(0.0, dtype=torch.float32)
            bin_idx = torch.tensor(-1, dtype=torch.long)

        output_noise_channels = add_sensor_noise(
            output_noise_channels,
            sensor_noise_db=self.cfg.sensor_noise_db,
        )
        mix_channels = _finite_audio(output_noise_channels + drone_channels).reshape(
            output_noise_channels.shape
        )
        if is_positive:
            snr_db = calculate_snr_db_from_mix_and_background(
                mix_channels,
                output_noise_channels,
            )
            bin_idx = torch.tensor(self._bin_idx_for_snr(snr_db), dtype=torch.long)
        if self.cfg.beamformer == "mvdr":
            look_direction = direction_unit_vector(beam_az, beam_el)
            mvdr_cache = getattr(self, "mvdr_cache", None)
            if mvdr_cache is not None:
                weights = mvdr_cache.weights_for_file(
                    output_noise_section.info,
                    look_direction=look_direction,
                )
            else:
                assert mvdr_noise_section is not None
                weights = estimate_mvdr_weights(
                    mvdr_noise_section.channels,
                    positions_m=self.positions_m,
                    look_direction=look_direction,
                    sample_rate=self.sample_rate,
                    speed_of_sound_mps=self.cfg.speed_of_sound_mps,
                    n_fft=self.cfg.stft_n_fft,
                    hop_length=self.cfg.hop_length,
                    diagonal_loading=self.cfg.diagonal_loading,
                )
            beam_mix = apply_mvdr_weights(
                mix_channels,
                weights,
                sample_rate=self.sample_rate,
                n_fft=self.cfg.stft_n_fft,
                hop_length=self.cfg.hop_length,
                target_length=self.target_length_samples,
            )
            if self.return_components:
                beam_noise = apply_mvdr_weights(
                    output_noise_channels,
                    weights,
                    sample_rate=self.sample_rate,
                    n_fft=self.cfg.stft_n_fft,
                    hop_length=self.cfg.hop_length,
                    target_length=self.target_length_samples,
                )
                beam_drone = apply_mvdr_weights(
                    drone_channels,
                    weights,
                    sample_rate=self.sample_rate,
                    n_fft=self.cfg.stft_n_fft,
                    hop_length=self.cfg.hop_length,
                    target_length=self.target_length_samples,
                )
        elif self.cfg.beamformer == "mean":
            beam_mix = np.mean(mix_channels, axis=0)
            if self.return_components:
                beam_noise = np.mean(output_noise_channels, axis=0)
                beam_drone = np.mean(drone_channels, axis=0)
        elif self.cfg.beamformer == "channel0":
            beam_mix = mix_channels[0]
            if self.return_components:
                beam_noise = output_noise_channels[0]
                beam_drone = drone_channels[0]
        else:
            mic_idx = random.randrange(output_noise_channels.shape[0])
            beam_mix = mix_channels[mic_idx]
            if self.return_components:
                beam_noise = output_noise_channels[mic_idx]
                beam_drone = drone_channels[mic_idx]
        mix_tensor = torch.as_tensor(app_window_normalize(beam_mix), dtype=torch.float32)
        snr_tensor = torch.tensor(snr_db, dtype=torch.float32)
        if self.return_components:
            item = BatchItem(
                mix=mix_tensor,
                label=label,
                bin_idx=bin_idx,
                drone=torch.as_tensor(_finite_audio(beam_drone), dtype=torch.float32),
                noise=torch.as_tensor(_finite_audio(beam_noise), dtype=torch.float32),
                snr_db=snr_tensor,
            )
            values = item.to_tuple(return_bin=True, return_components=True)
        else:
            values = (mix_tensor, label, bin_idx, snr_tensor)
        meta = (
            torch.tensor(drone_az, dtype=torch.float32),
            torch.tensor(drone_el, dtype=torch.float32),
            torch.tensor(drone_distance, dtype=torch.float32),
            torch.tensor(beam_az, dtype=torch.float32),
            torch.tensor(beam_el, dtype=torch.float32),
            torch.tensor(target_scale, dtype=torch.float32),
        )
        return values + meta

    def _load_noise_section(self) -> NoiseSection:
        info = random.choice(self.noise_files)
        clip_seconds = self.target_length_samples / float(self.sample_rate)
        max_start = max(0.0, info.duration_seconds - clip_seconds)
        start = random.uniform(0.0, max_start)
        channels = load_multichannel_segment(
            info,
            start_seconds=start,
            length_samples=self.target_length_samples,
            target_sample_rate=self.sample_rate,
            expected_channels=self.cfg.rows * self.cfg.cols,
        )
        channels = preprocess_noise_channels(
            channels,
            cfg=self.cfg,
            highpass_hz=self.highpass_hz,
            sample_rate=self.sample_rate,
        )
        return NoiseSection(info=info, channels=channels, start_seconds=start)

    def _load_noise_channels(self) -> np.ndarray:
        return self._load_noise_section().channels

    def _augment_background_channels(self, channels: np.ndarray) -> np.ndarray:
        spatial_bg_ds = getattr(self, "spatial_bg_ds", None)
        if (
            spatial_bg_ds is None
            or self.cfg.spatial_bg_count <= 0
            or random.random() >= self.cfg.spatial_bg_probability
        ):
            return channels
        augmented = channels.astype(np.float32, copy=True)
        for _ in range(self._pick_spatial_bg_count()):
            bg_audio = self._load_spatial_bg_clip()
            bg_az, bg_el, bg_distance = sample_direction(self.cfg)
            bg_channels = simulate_drone_free_space(
                bg_audio,
                positions_m=self.positions_m,
                azimuth_deg=bg_az,
                elevation_deg=bg_el,
                distance_m=bg_distance,
                sample_rate=self.sample_rate,
                cfg=self.cfg,
                target_length=self.target_length_samples,
            )
            attenuation_db = random.uniform(self.cfg.spatial_bg_max_attenuation_db, 0.0)
            augmented = augmented + bg_channels * np.float32(
                10.0 ** (attenuation_db / 20.0)
            )
        return _finite_audio(augmented).reshape(channels.shape)

    def _pick_spatial_bg_count(self) -> int:
        count = 1 if self.cfg.spatial_bg_count > 0 else 0
        while (
            count < self.cfg.spatial_bg_count
            and random.random() < self.cfg.spatial_bg_multi_probability
        ):
            count += 1
        return count

    def _load_spatial_bg_clip(self) -> np.ndarray:
        spatial_bg_ds = getattr(self, "spatial_bg_ds", None)
        if spatial_bg_ds is None:
            raise RuntimeError("spatial background dataset is not loaded")
        row = spatial_bg_ds[random.randint(0, len(spatial_bg_ds) - 1)]
        bg = load_hf_audio_array(row["audio"], target_sample_rate=self.sample_rate)
        bg = _fit_length(bg, self.target_length_samples)
        if self.highpass_hz > 0.0:
            bg = highpass(bg, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate)
        return _rms_normalize(bg)

    def _load_drone(self) -> np.ndarray:
        row = self.drone_ds[random.randint(0, len(self.drone_ds) - 1)]
        drone = load_hf_audio_array(row["audio"], target_sample_rate=self.sample_rate)
        drone = _fit_length(drone, self.target_length_samples)
        if self.highpass_hz > 0.0:
            drone = highpass(drone, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate)
        return _rms_normalize(drone)

    def _bin_idx_for_snr(self, snr_db: float) -> int:
        bounds = [
            (
                idx,
                float(min(bin_obj.low_db, bin_obj.high_db)),
                float(max(bin_obj.low_db, bin_obj.high_db)),
            )
            for idx, bin_obj in enumerate(self.snr_bins)
        ]
        if not bounds:
            return -1
        lowest = min(bounds, key=lambda item: item[1])
        highest = max(bounds, key=lambda item: item[2])
        if snr_db <= lowest[1]:
            return lowest[0]
        if snr_db >= highest[2]:
            return highest[0]
        closest_idx = 0
        closest_distance = float("inf")
        for idx, lo, hi in bounds:
            if lo <= snr_db <= hi:
                return idx
            distance = min(abs(snr_db - lo), abs(snr_db - hi))
            if distance < closest_distance:
                closest_idx = idx
                closest_distance = distance
        return closest_idx
