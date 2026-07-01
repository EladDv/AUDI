"""MVDR beamforming utilities for live UMA16 detector inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

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
class BeamDirection:
    index: int
    azimuth_deg: float
    elevation_deg: float

    @property
    def name(self) -> str:
        az = int(round(self.azimuth_deg)) % 360
        el = int(round(self.elevation_deg))
        return f"beam{self.index}_az{az:03d}_el{el:02d}"


@dataclass(frozen=True)
class MVDRBeamformerConfig:
    beam_count: int = 36
    elevation_count: int = 3
    min_elevation_deg: float = 5.0
    max_elevation_deg: float = 70.0
    n_fft: int = 512
    hop_length: int = 160
    diagonal_loading: float = 1e-2
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S
    mic_indices: tuple[int, ...] | None = None
    deglitch_enabled: bool = True
    deglitch_threshold: float = 0.001
    deglitch_loudness_ratio: float = 8.0
    deglitch_diff_ratio: float = 12.0
    deglitch_window_samples: int = 64


def deglitch_multichannel(
    channels: np.ndarray,
    *,
    threshold: float = 0.001,
    loudness_ratio: float = 8.0,
    diff_ratio: float = 12.0,
    window_samples: int = 64,
) -> np.ndarray:
    """Repair short discontinuity jumps with local interpolation."""
    repaired = np.nan_to_num(
        np.asarray(channels, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).copy()
    if repaired.ndim != 2 or repaired.shape[-1] < 3:
        return repaired

    for ch_idx in range(repaired.shape[0]):
        y = repaired[ch_idx]
        diff = np.abs(np.diff(y))
        if window_samples > 0:
            loudness_size = 2 * window_samples + 3
            diff_size = 2 * window_samples + 1
            loudness = median_filter(
                np.abs(y),
                size=loudness_size,
                mode="nearest",
            )
            local_loudness = (
                np.minimum(loudness[:-1], loudness[1:])
                + 1e-8
            )
            local_diff_scale = (
                median_filter(
                    diff,
                    size=diff_size,
                    mode="nearest",
                )
                + 1e-8
            )
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


def build_beam_grid(cfg: MVDRBeamformerConfig) -> tuple[BeamDirection, ...]:
    beam_count = max(1, int(cfg.beam_count))
    elevation_count = max(1, min(int(cfg.elevation_count), beam_count))
    azimuth_count = max(1, math.ceil(beam_count / elevation_count))
    if elevation_count == 1:
        elevations = [
            (float(cfg.min_elevation_deg) + float(cfg.max_elevation_deg)) / 2.0
        ]
    else:
        elevations = np.linspace(
            float(cfg.min_elevation_deg),
            float(cfg.max_elevation_deg),
            elevation_count,
        ).tolist()

    beams: list[BeamDirection] = []
    for elevation in elevations:
        for azimuth_idx in range(azimuth_count):
            if len(beams) >= beam_count:
                return tuple(beams)
            azimuth = 360.0 * azimuth_idx / azimuth_count
            beams.append(
                BeamDirection(
                    index=len(beams),
                    azimuth_deg=float(azimuth),
                    elevation_deg=float(elevation),
                )
            )
    return tuple(beams)


def direction_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    return np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )


class MVDRBeamformer:
    """Apply a bank of MVDR look directions to a multichannel audio window."""

    def __init__(self, cfg: MVDRBeamformerConfig | None = None) -> None:
        self.cfg = cfg or MVDRBeamformerConfig()
        self.beams = build_beam_grid(self.cfg)
        self._cov_inv: np.ndarray | None = None
        self._cov_freqs: np.ndarray | None = None
        self._cov_positions: np.ndarray | None = None
        self._cov_mic_indices: tuple[int, ...] | None = None
        self._cov_n_fft: int | None = None
        self._cov_hop: int | None = None
        self._cov_sample_rate: int | None = None

    @property
    def has_covariance(self) -> bool:
        return self._cov_inv is not None

    def update_covariance(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Estimate the MVDR noise covariance from a background-only window."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] <= 0:
            return False

        mic_indices = self._mic_indices(audio.shape[1])
        if len(mic_indices) < 2:
            return False
        positions = UMA16_MIC_POSITIONS_M[list(mic_indices)]
        channels = self._prepare_channels(audio, mic_indices)

        n_fft = self._n_fft_for_length(channels.shape[-1])
        hop = self._hop_for_n_fft(n_fft)
        _, freqs, stft = signal.stft(
            channels,
            fs=int(sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        self._cov_inv = self._inverse_covariances(stft)
        self._cov_freqs = freqs
        self._cov_positions = positions
        self._cov_mic_indices = mic_indices
        self._cov_n_fft = n_fft
        self._cov_hop = hop
        self._cov_sample_rate = int(sample_rate)
        return True

    def beamform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_audio: np.ndarray | None = None,
    ) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f"expected [frames, channels] audio, got {audio.shape}")
        if audio.shape[0] <= 0:
            return []

        if noise_audio is not None:
            self.update_covariance(noise_audio, sample_rate)

        mic_indices = self._mic_indices(audio.shape[1])
        if len(mic_indices) < 2:
            return []

        if not self.has_covariance:
            self.update_covariance(audio, sample_rate)
        if (
            self._cov_inv is None
            or self._cov_freqs is None
            or self._cov_positions is None
            or self._cov_mic_indices is None
            or self._cov_n_fft is None
            or self._cov_hop is None
        ):
            return []
        if self._cov_sample_rate != int(sample_rate) or self._cov_mic_indices != mic_indices:
            self.update_covariance(audio, sample_rate)
        if (
            self._cov_inv is None
            or self._cov_freqs is None
            or self._cov_positions is None
            or self._cov_mic_indices is None
            or self._cov_n_fft is None
            or self._cov_hop is None
        ):
            return []

        channels = self._prepare_channels(audio, self._cov_mic_indices)
        n_fft = self._cov_n_fft
        hop = self._cov_hop
        _, freqs, stft = signal.stft(
            channels,
            fs=int(sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        if len(freqs) != len(self._cov_freqs):
            self.update_covariance(audio, sample_rate)
            if (
                self._cov_inv is None
                or self._cov_freqs is None
                or self._cov_n_fft is None
                or self._cov_hop is None
            ):
                return []
            n_fft = self._cov_n_fft
            hop = self._cov_hop
            _, freqs, stft = signal.stft(
                channels,
                fs=int(sample_rate),
                window="hann",
                nperseg=n_fft,
                noverlap=n_fft - hop,
                boundary="zeros",
                padded=True,
                axis=-1,
            )
            if len(freqs) != len(self._cov_freqs):
                return []

        outputs: list[dict] = []
        for beam in self.beams:
            weights = self._weights_for_beam(
                self._cov_inv,
                self._cov_freqs,
                self._cov_positions,
                direction_unit_vector(beam.azimuth_deg, beam.elevation_deg),
            )
            beam_stft = np.einsum("fm,mft->ft", np.conj(weights), stft)
            _, beam_audio = signal.istft(
                beam_stft,
                fs=int(sample_rate),
                window="hann",
                nperseg=n_fft,
                noverlap=n_fft - hop,
                input_onesided=True,
            )
            beam_audio = np.asarray(beam_audio[: audio.shape[0]], dtype=np.float32)
            if beam_audio.shape[0] < audio.shape[0]:
                pad_width = (0, audio.shape[0] - beam_audio.shape[0])
                beam_audio = np.pad(beam_audio, pad_width)
            outputs.append(
                {
                    "index": beam.index,
                    "name": beam.name,
                    "audio": beam_audio,
                    "azimuth_deg": beam.azimuth_deg,
                    "elevation_deg": beam.elevation_deg,
                    "mic_indices": list(self._cov_mic_indices),
                }
            )
        return outputs

    def _prepare_channels(
        self,
        audio: np.ndarray,
        mic_indices: tuple[int, ...],
    ) -> np.ndarray:
        channels = audio[:, list(mic_indices)].T.astype(np.float64, copy=False)
        channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.deglitch_enabled:
            channels = deglitch_multichannel(
                channels,
                threshold=self.cfg.deglitch_threshold,
                loudness_ratio=self.cfg.deglitch_loudness_ratio,
                diff_ratio=self.cfg.deglitch_diff_ratio,
                window_samples=self.cfg.deglitch_window_samples,
            ).astype(np.float64, copy=False)
        return channels

    def _n_fft_for_length(self, sample_count: int) -> int:
        return max(2, min(int(self.cfg.n_fft), int(sample_count)))

    def _hop_for_n_fft(self, n_fft: int) -> int:
        return max(1, min(int(self.cfg.hop_length), int(n_fft) - 1))

    def _mic_indices(self, channel_count: int) -> tuple[int, ...]:
        configured = self.cfg.mic_indices
        if configured is None:
            return tuple(range(min(channel_count, len(UMA16_MIC_POSITIONS_M))))
        return tuple(
            idx
            for idx in configured
            if 0 <= int(idx) < channel_count and int(idx) < len(UMA16_MIC_POSITIONS_M)
        )

    def _inverse_covariances(self, stft: np.ndarray) -> np.ndarray:
        mic_count = stft.shape[0]
        cov_inv = np.empty((stft.shape[1], mic_count, mic_count), dtype=np.complex128)
        eye = np.eye(mic_count, dtype=np.complex128)
        for freq_idx in range(stft.shape[1]):
            x = stft[:, freq_idx, :]
            cov = (x @ x.conj().T) / max(1, x.shape[-1])
            trace = float(np.real(np.trace(cov)))
            loading = self.cfg.diagonal_loading * max(trace / mic_count, 1e-8)
            cov_inv[freq_idx] = np.linalg.pinv(cov + loading * eye)
        return cov_inv

    def _weights_for_beam(
        self,
        cov_inv: np.ndarray,
        freqs: np.ndarray,
        positions_m: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        delays = positions_m @ direction / float(self.cfg.speed_of_sound_m_s)
        steering = np.exp(-2j * np.pi * freqs[:, np.newaxis] * delays[np.newaxis, :])
        numerator = np.einsum("fmn,fn->fm", cov_inv, steering)
        denominator = np.einsum("fm,fm->f", np.conj(steering), numerator)
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        return numerator / denominator[:, np.newaxis]
