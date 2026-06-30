"""MUSIC direction-of-arrival estimation for the live AUDI ring buffer.

The detector triggers this module only after a model-positive drone result.
This file owns DOA profiles, per-mic runtime disabling, MUSIC estimation,
azimuth smoothing, and confidence scoring.
"""

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import signal

logger = logging.getLogger("audi.doa")

SPEED_OF_SOUND_M_S = 343.0

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
class DOASettings:
    name: str = "default"
    mic_indices: tuple[int, ...] = (0, 7, 14)
    window_s: float = 1.0
    azimuth_step_deg: float = 1.0
    elevation_deg: float = 0.0
    n_sources: int = 1
    n_fft: int = 2048
    hop_length: int = 256
    hps_harmonics: int = 3
    hps_fmin_hz: float = 100.0
    peak_fmin_hz: float = 100.0
    peak_fmax_hz: float = 600.0
    cfar_guard_bins: int = 4
    cfar_ref_bins: int = 20
    music_half_bins: int = 1
    diagonal_loading: float = 1e-3
    smoothing_predictions: int = 5
    confidence_jump_deg: float = 45.0


@dataclass(frozen=True)
class DOAConfig:
    enabled: bool = False
    active_profile: str = "default"
    disabled_channels: frozenset[int] = frozenset()
    profiles: dict[str, DOASettings] | None = None


def parse_doa_config(config: dict[str, Any]) -> DOAConfig:
    """Build typed DOA config from the app config dictionary."""
    cfg = dict(config.get("doa", {}) or {})
    profile_cfgs = cfg.get("profiles")
    if isinstance(profile_cfgs, dict) and profile_cfgs:
        profiles = {
            str(name): _parse_doa_settings(str(name), _merge_profile(cfg, profile))
            for name, profile in profile_cfgs.items()
        }
        active_profile = str(cfg.get("active_profile") or next(iter(profiles)))
        if active_profile not in profiles:
            active_profile = next(iter(profiles))
    else:
        active_profile = str(cfg.get("active_profile", "default"))
        profiles = {active_profile: _parse_doa_settings(active_profile, cfg)}

    return DOAConfig(
        enabled=bool(cfg.get("enabled", False)),
        active_profile=active_profile,
        disabled_channels=frozenset(_parse_int_tuple(cfg.get("disabled_channels", ()))),
        profiles=profiles,
    )


class DOAEstimator:
    """On-demand MUSIC estimator over the recorder ring buffer."""

    def __init__(self, config: dict[str, Any], ring_buffer):
        self.config = parse_doa_config(config)
        self.ring_buffer = ring_buffer
        audio_cfg = config.get("audio", {}) or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.input_channels = int(audio_cfg.get("channels", 1))
        self._running = False
        self._lock = threading.RLock()
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._estimate_count = 0
        self._history: list[dict[str, float]] = []

    def start(self) -> None:
        if not self.config.enabled:
            logger.info("DOA estimator disabled")
            return
        self._validate_startup()
        self._running = True
        settings = self._active_settings()
        logger.info(
            "DOA estimator armed: profile=%s mics=%s window=%.2fs",
            self.config.active_profile,
            list(self._active_mic_indices(settings, self.config)),
            settings.window_s,
        )

    def stop(self) -> None:
        self._running = False

    def force_estimate(self) -> dict[str, Any]:
        """Run one estimate synchronously and return the current result."""
        if not self.config.enabled:
            return self._not_ready("DOA disabled")
        result = self._estimate()
        with self._lock:
            self._last_result = result
            self._last_error = result.get("error")
            if result.get("ok"):
                self._estimate_count += 1
        return result

    def set_profile(self, profile: str) -> dict[str, Any]:
        """Switch the active DOA profile at runtime."""
        profile = str(profile)
        profiles = self.config.profiles or {}
        if profile not in profiles:
            raise ValueError(f"unknown DOA profile: {profile}")
        with self._lock:
            updated = replace(self.config, active_profile=profile)
            settings = self._settings_for(updated.active_profile, updated)
            if len(self._active_mic_indices(settings, updated)) < 2:
                raise ValueError("at least two active DOA microphones are required")
            self.config = updated
            self._reset_smoothing()
        return self.status

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> dict[str, Any]:
        """Enable/disable one capture channel for DOA without affecting recording."""
        channel_index = int(channel_index)
        if channel_index < 0 or channel_index >= self.input_channels:
            raise ValueError(
                f"channel_index must be between 0 and {self.input_channels - 1}"
            )
        with self._lock:
            disabled = set(self.config.disabled_channels)
            if enabled:
                disabled.discard(channel_index)
            else:
                disabled.add(channel_index)
            updated = replace(self.config, disabled_channels=frozenset(disabled))
            settings = self._settings_for(updated.active_profile, updated)
            if len(self._active_mic_indices(settings, updated)) < 2:
                raise ValueError("at least two active DOA microphones are required")
            self.config = updated
            self._reset_smoothing()
        return self.status

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            last_result = dict(self._last_result or {})
            config = self.config
            settings = self._settings_for(config.active_profile, config)
            active_mics = self._active_mic_indices(settings, config)
            return {
                "enabled": config.enabled,
                "running": self._running,
                "profiles": sorted((config.profiles or {}).keys()),
                "active_profile": config.active_profile,
                "mic_indices": list(active_mics),
                "profile_mic_indices": list(settings.mic_indices),
                "disabled_channels": sorted(config.disabled_channels),
                "hps_harmonics": settings.hps_harmonics,
                "window_s": settings.window_s,
                "azimuth_step_deg": settings.azimuth_step_deg,
                "smoothing_predictions": settings.smoothing_predictions,
                "confidence_jump_deg": settings.confidence_jump_deg,
                "estimate_count": self._estimate_count,
                "last_error": self._last_error,
                "last_result": last_result,
                "azimuth_deg": last_result.get("azimuth_deg"),
                "raw_azimuth_deg": last_result.get("raw_azimuth_deg"),
                "confidence": last_result.get("confidence"),
                "dominant_frequency_hz": last_result.get("dominant_frequency_hz"),
                "peak_hps_snr_db": last_result.get("peak_hps_snr_db"),
                "spectrum_peak": last_result.get("spectrum_peak"),
                "channels": [
                    {
                        "channel_index": idx,
                        "profile_member": idx in settings.mic_indices,
                        "enabled": idx in active_mics,
                        "disabled": idx in config.disabled_channels,
                    }
                    for idx in settings.mic_indices
                ],
            }

    def _estimate(self) -> dict[str, Any]:
        settings = self._active_settings()
        active_mics = self._active_mic_indices(settings, self.config)
        self._validate_settings(settings, active_mics)

        requested_samples = max(
            settings.n_fft,
            int(round(settings.window_s * self.sample_rate)),
        )
        audio = np.asarray(
            self.ring_buffer.get_recent(requested_samples, channel=None),
            dtype=np.float32,
        )
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        if audio.shape[0] < settings.n_fft:
            return self._not_ready(
                f"need at least {settings.n_fft} samples, have {audio.shape[0]}"
            )
        if max(active_mics) >= audio.shape[1]:
            needed_channels = max(active_mics) + 1
            return self._not_ready(
                f"ring buffer has {audio.shape[1]} channels, need {needed_channels}"
            )

        selected = audio[:, list(active_mics)].T
        dominant = self._dominant_hps_frequency(selected[0], settings)
        if dominant is None:
            return self._not_ready("no HPS peak in configured search band")

        dominant_frequency_hz, peak_hps_snr_db = dominant
        freqs, stft = self._stft_channels(selected, settings)
        harmonic_freqs = self._music_frequencies(freqs, dominant_frequency_hz, settings)
        if not harmonic_freqs:
            return self._not_ready("no usable MUSIC frequency bins")

        covariance = self._covariance(stft, freqs, harmonic_freqs, settings)
        azimuths = np.arange(-180.0, 180.0, settings.azimuth_step_deg)
        spectrum = music_azimuth_spectrum(
            covariance,
            UMA16_MIC_POSITIONS_M[np.array(active_mics)],
            harmonic_freqs,
            azimuths,
            elevation_deg=settings.elevation_deg,
            n_sources=settings.n_sources,
        )
        peak_index = int(np.argmax(spectrum))
        raw_azimuth = float(azimuths[peak_index])
        smoothed = self._smooth_azimuth(
            raw_azimuth,
            spectrum,
            peak_hps_snr_db,
            settings,
        )
        return {
            "ok": True,
            "timestamp": time.time(),
            "profile": self.config.active_profile,
            "azimuth_deg": round(smoothed["azimuth_deg"], 2),
            "raw_azimuth_deg": round(raw_azimuth, 2),
            "azimuth_compass_deg": round((smoothed["azimuth_deg"] + 360.0) % 360.0, 2),
            "confidence": round(smoothed["confidence"], 3),
            "stability": round(smoothed["stability"], 3),
            "jump_deg": round(smoothed["jump_deg"], 2),
            "spectrum_peak_to_median": round(smoothed["spectrum_peak_to_median"], 3),
            "spectrum_peak": round(float(spectrum[peak_index]), 4),
            "dominant_frequency_hz": round(float(dominant_frequency_hz), 1),
            "peak_hps_snr_db": round(float(peak_hps_snr_db), 2),
            "music_frequencies_hz": [round(float(freq), 1) for freq in harmonic_freqs],
            "mic_indices": list(active_mics),
            "window_s": settings.window_s,
        }

    def _validate_startup(self) -> None:
        settings = self._active_settings()
        self._validate_settings(settings, self._active_mic_indices(settings, self.config))

    def _validate_settings(
        self,
        settings: DOASettings,
        active_mics: tuple[int, ...],
    ) -> None:
        if len(active_mics) < 2:
            raise ValueError("doa.mic_indices must contain at least two microphones")
        if settings.n_sources >= len(active_mics):
            raise ValueError("doa.music.n_sources must be smaller than mic count")
        if settings.hop_length >= settings.n_fft:
            raise ValueError("doa.hop_length must be smaller than doa.n_fft")
        invalid = [
            idx
            for idx in settings.mic_indices
            if idx < 0 or idx >= self.input_channels or idx >= len(UMA16_MIC_POSITIONS_M)
        ]
        if invalid:
            raise ValueError(
                "doa.mic_indices outside captured/known channels: "
                f"{invalid} for {self.input_channels} captured channels"
            )

    def _active_settings(self) -> DOASettings:
        return self._settings_for(self.config.active_profile, self.config)

    @staticmethod
    def _settings_for(profile: str, config: DOAConfig) -> DOASettings:
        profiles = config.profiles or {}
        if profile not in profiles:
            raise ValueError(f"unknown DOA profile: {profile}")
        return profiles[profile]

    @staticmethod
    def _active_mic_indices(
        settings: DOASettings,
        config: DOAConfig | None = None,
    ) -> tuple[int, ...]:
        disabled = config.disabled_channels if config else frozenset()
        return tuple(idx for idx in settings.mic_indices if idx not in disabled)

    def _dominant_hps_frequency(
        self,
        samples: np.ndarray,
        settings: DOASettings,
    ) -> tuple[float, float] | None:
        _, _, spectrum = signal.spectrogram(
            samples,
            fs=self.sample_rate,
            window="hann",
            nperseg=settings.n_fft,
            noverlap=settings.n_fft - settings.hop_length,
            scaling="spectrum",
        )
        if spectrum.size == 0:
            return None
        freqs = np.fft.rfftfreq(settings.n_fft, d=1.0 / self.sample_rate)
        cfar_db = 10.0 * np.log10(
            cfar_normalize_frequency(
                spectrum,
                guard_bins=settings.cfar_guard_bins,
                ref_bins=settings.cfar_ref_bins,
            )
            + 1e-12
        )
        max_base_bin = len(freqs) // settings.hps_harmonics
        if max_base_bin <= 1:
            return None
        base_bins = np.arange(max_base_bin)
        hps_bins = base_bins[freqs[base_bins] >= settings.hps_fmin_hz]
        if hps_bins.size == 0:
            return None
        hps_db = cfar_db[hps_bins, :].copy()
        for harmonic in range(2, settings.hps_harmonics + 1):
            hps_db += cfar_db[harmonic * hps_bins, :]
        mean_hps_db = hps_db.mean(axis=1)
        peak_mask = (
            (freqs[hps_bins] >= settings.peak_fmin_hz)
            & (freqs[hps_bins] <= settings.peak_fmax_hz)
        )
        if not np.any(peak_mask):
            return None
        search_freqs = freqs[hps_bins][peak_mask]
        search_scores = mean_hps_db[peak_mask]
        peak_index = int(np.argmax(search_scores))
        return float(search_freqs[peak_index]), float(search_scores[peak_index])

    def _stft_channels(
        self,
        channels_first_audio: np.ndarray,
        settings: DOASettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        stfts = []
        stft_freqs = None
        for samples in channels_first_audio:
            freqs, _, matrix = signal.stft(
                samples,
                fs=self.sample_rate,
                nperseg=settings.n_fft,
                noverlap=settings.n_fft - settings.hop_length,
                boundary=None,
                padded=False,
            )
            stft_freqs = freqs
            stfts.append(matrix)
        return stft_freqs, np.stack(stfts, axis=0).astype(np.complex128)

    def _music_frequencies(
        self,
        stft_freqs: np.ndarray,
        dominant_f0: float,
        settings: DOASettings,
    ) -> list[float]:
        freqs = []
        for harmonic in range(1, settings.hps_harmonics + 1):
            target = dominant_f0 * harmonic
            if target > self.sample_rate / 2:
                break
            center = int(np.searchsorted(stft_freqs, target))
            for delta in range(-settings.music_half_bins, settings.music_half_bins + 1):
                idx = center + delta
                if 0 <= idx < len(stft_freqs):
                    freqs.append(float(stft_freqs[idx]))
        return sorted(set(freqs))

    def _covariance(
        self,
        stft: np.ndarray,
        stft_freqs: np.ndarray,
        harmonic_freqs: list[float],
        settings: DOASettings,
    ) -> np.ndarray:
        channel_count = stft.shape[0]
        covariance = np.zeros((channel_count, channel_count), dtype=np.complex128)
        frame_count = 0
        for freq in harmonic_freqs:
            idx = int(np.searchsorted(stft_freqs, freq))
            if idx < 0 or idx >= stft.shape[1]:
                continue
            frames = stft[:, idx, :]
            covariance += frames @ frames.conj().T
            frame_count += frames.shape[1]
        if frame_count == 0:
            raise ValueError("no STFT frames for MUSIC covariance")
        covariance /= frame_count
        trace = float(np.real(np.trace(covariance)))
        if trace > 0.0 and settings.diagonal_loading > 0.0:
            covariance += (
                np.eye(channel_count, dtype=np.complex128)
                * (trace / channel_count)
                * settings.diagonal_loading
            )
        return covariance

    def _smooth_azimuth(
        self,
        raw_azimuth: float,
        spectrum: np.ndarray,
        peak_hps_snr_db: float,
        settings: DOASettings,
    ) -> dict[str, float]:
        with self._lock:
            previous = self._history[-1]["azimuth_deg"] if self._history else None
            jump_deg = 0.0 if previous is None else angular_distance_deg(raw_azimuth, previous)
            stability = 1.0 if previous is None else _clamp(
                1.0 - jump_deg / settings.confidence_jump_deg
            )
            peak_to_median = float(np.max(spectrum) / (np.median(spectrum) + 1e-12))
            sharpness = _clamp((peak_to_median - 1.0) / 8.0)
            hps_strength = _clamp((float(peak_hps_snr_db) + 3.0) / 18.0)
            confidence = _clamp(
                0.45 * sharpness + 0.35 * hps_strength + 0.20 * stability
            )
            history = [
                *self._history,
                {
                    "raw_azimuth_deg": normalize_signed_deg(raw_azimuth),
                    "confidence": max(confidence, 1e-3),
                },
            ][-settings.smoothing_predictions :]
            smoothed = circular_mean_deg(
                [item["raw_azimuth_deg"] for item in history],
                [item["confidence"] for item in history],
            )
            history[-1]["azimuth_deg"] = smoothed
            self._history = history
        return {
            "azimuth_deg": smoothed,
            "confidence": confidence,
            "stability": stability,
            "jump_deg": jump_deg,
            "spectrum_peak_to_median": peak_to_median,
        }

    def _reset_smoothing(self) -> None:
        self._history.clear()
        self._last_error = None

    @staticmethod
    def _not_ready(reason: str) -> dict[str, Any]:
        return {"ok": False, "timestamp": time.time(), "error": reason}


def cfar_normalize_frequency(
    spectrum: np.ndarray,
    guard_bins: int,
    ref_bins: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cell-averaging CFAR normalization along the frequency axis."""
    values = np.asarray(spectrum, dtype=np.float64)
    freq_bins, time_bins = values.shape
    pad = guard_bins + ref_bins
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
    prefix = np.concatenate(
        [np.zeros((1, time_bins), dtype=np.float64), np.cumsum(padded, axis=0)],
        axis=0,
    )
    bins = np.arange(freq_bins)
    centers = bins + pad
    left = prefix[centers - guard_bins] - prefix[centers - guard_bins - ref_bins]
    right = (
        prefix[centers + guard_bins + 1 + ref_bins]
        - prefix[centers + guard_bins + 1]
    )
    noise = (left + right) / (2.0 * ref_bins)
    return values / (noise + eps)


def music_azimuth_spectrum(
    covariance: np.ndarray,
    mic_positions_m: np.ndarray,
    frequencies_hz: list[float],
    azimuths_deg: np.ndarray,
    elevation_deg: float = 0.0,
    n_sources: int = 1,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    """Return a normalized MUSIC pseudo-spectrum over compass azimuths."""
    _, eigenvectors = np.linalg.eigh(covariance)
    noise_vectors = eigenvectors[:, : -int(n_sources)]
    noise_projector = noise_vectors @ noise_vectors.conj().T

    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_math_rad = np.deg2rad(90.0 - azimuths_deg)
    kx = np.cos(elevation_rad) * np.cos(azimuth_math_rad)
    ky = np.cos(elevation_rad) * np.sin(azimuth_math_rad)
    kz = np.full_like(azimuth_math_rad, np.sin(elevation_rad))

    spectrum = np.zeros(len(azimuths_deg), dtype=np.float64)
    for freq_hz in frequencies_hz:
        wave_number = 2.0 * np.pi * freq_hz / speed_of_sound_m_s
        phase = wave_number * (
            mic_positions_m[:, 0:1] * kx[None, :]
            + mic_positions_m[:, 1:2] * ky[None, :]
            + mic_positions_m[:, 2:3] * kz[None, :]
        )
        steering = np.exp(1j * phase)
        steering /= np.linalg.norm(steering, axis=0, keepdims=True)
        denominator = np.real(
            np.einsum("ca,cd,da->a", steering.conj(), noise_projector, steering)
        )
        spectrum += 1.0 / (denominator + 1e-12)
    spectrum /= max(len(frequencies_hz), 1)
    spectrum /= np.max(spectrum) + 1e-30
    return spectrum


def circular_mean_deg(angles_deg: list[float], weights: list[float]) -> float:
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    weight_arr = np.asarray(weights, dtype=np.float64)
    x = float(np.sum(np.cos(angles_rad) * weight_arr))
    y = float(np.sum(np.sin(angles_rad) * weight_arr))
    return normalize_signed_deg(float(np.rad2deg(np.arctan2(y, x))))


def angular_distance_deg(a_deg: float, b_deg: float) -> float:
    return abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)


def normalize_signed_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _merge_profile(base_cfg: dict[str, Any], profile_cfg: Any) -> dict[str, Any]:
    merged = {
        key: value
        for key, value in base_cfg.items()
        if key not in {"profiles", "active_profile", "disabled_channels"}
    }
    if isinstance(profile_cfg, dict):
        merged.update(profile_cfg)
    return merged


def _parse_doa_settings(name: str, cfg: dict[str, Any]) -> DOASettings:
    hps_cfg = dict(cfg.get("hps", {}) or {})
    music_cfg = dict(cfg.get("music", {}) or {})
    peak_cfg = dict(hps_cfg.get("peak_search", {}) or {})
    cfar_cfg = dict(hps_cfg.get("cfar", {}) or {})

    return DOASettings(
        name=name,
        mic_indices=_parse_int_tuple(cfg.get("mic_indices", (0, 7, 14))),
        window_s=_positive_float(music_cfg.get("window_s", cfg.get("window_s", 1.0)), 0.1),
        azimuth_step_deg=_positive_float(music_cfg.get("azimuth_step_deg", 1.0), 0.1),
        elevation_deg=float(music_cfg.get("elevation_deg", 0.0)),
        n_sources=max(1, int(music_cfg.get("n_sources", 1))),
        n_fft=max(128, int(cfg.get("n_fft", 2048))),
        hop_length=max(1, int(cfg.get("hop_length", 256))),
        hps_harmonics=max(1, int(hps_cfg.get("harmonics", 3))),
        hps_fmin_hz=max(0.0, float(hps_cfg.get("fmin_hz", 100.0))),
        peak_fmin_hz=max(0.0, float(peak_cfg.get("fmin_hz", 100.0))),
        peak_fmax_hz=max(1.0, float(peak_cfg.get("fmax_hz", 600.0))),
        cfar_guard_bins=max(0, int(cfar_cfg.get("guard_bins", 4))),
        cfar_ref_bins=max(1, int(cfar_cfg.get("ref_bins", 20))),
        music_half_bins=max(0, int(music_cfg.get("half_bins", 1))),
        diagonal_loading=max(0.0, float(music_cfg.get("diagonal_loading", 1e-3))),
        smoothing_predictions=max(1, int(music_cfg.get("smoothing_predictions", 5))),
        confidence_jump_deg=_positive_float(music_cfg.get("confidence_jump_deg", 45.0), 1.0),
    )


def _parse_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(item) for item in value)


def _positive_float(value: Any, minimum: float) -> float:
    return max(minimum, float(value))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
