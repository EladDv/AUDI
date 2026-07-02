"""MVDR beamforming utilities for live UMA16 detector inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

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
    beam_count: int = 12
    elevation_count: int = 1
    min_elevation_deg: float = 5.0
    max_elevation_deg: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    diagonal_loading: float = 1e-4
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S
    mic_indices: tuple[int, ...] | None = None
    deglitch_enabled: bool = True
    deglitch_threshold: float = 0.001
    deglitch_loudness_ratio: float = 8.0
    deglitch_diff_ratio: float = 12.0
    deglitch_window_samples: int = 64
    incremental_step_samples: int | None = None


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
    az = np.deg2rad(90.0 - float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    vector = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction vector has zero norm")
    return vector / norm


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
        self._beam_weights: np.ndarray | None = None
        self._stft_cache_channels: np.ndarray | None = None
        self._stft_cache_spec: np.ndarray | None = None
        self._stft_cache_key: tuple[int, tuple[int, ...], int, int] | None = None
        self._stft_cache_hits = 0
        self._stft_cache_misses = 0
        self._stft_last_reused_frames = 0

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
        freqs, _, stft = signal.stft(
            channels,
            fs=int(sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        cov_inv = self._inverse_covariances(stft)
        beam_weights = self._weights_for_beams(cov_inv, freqs, positions)
        self._cov_inv = cov_inv
        self._cov_freqs = freqs
        self._cov_positions = positions
        self._cov_mic_indices = mic_indices
        self._cov_n_fft = n_fft
        self._cov_hop = hop
        self._cov_sample_rate = int(sample_rate)
        self._beam_weights = beam_weights
        self._reset_stft_cache()
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
            or self._beam_weights is None
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
            or self._beam_weights is None
        ):
            return []

        raw_channels = self._select_channels(audio, self._cov_mic_indices)
        channels = self._repair_channels(raw_channels)
        n_fft = self._cov_n_fft
        hop = self._cov_hop
        freqs, stft = self._stft_channels_cached(
            channels,
            cache_channels=channels,
            sample_rate=int(sample_rate),
            mic_indices=self._cov_mic_indices,
            n_fft=n_fft,
            hop=hop,
        )
        if len(freqs) != len(self._cov_freqs):
            self.update_covariance(audio, sample_rate)
            if (
                self._cov_inv is None
                or self._cov_freqs is None
                or self._cov_n_fft is None
                or self._cov_hop is None
                or self._beam_weights is None
            ):
                return []
            n_fft = self._cov_n_fft
            hop = self._cov_hop
            freqs, stft = self._stft_channels_cached(
                channels,
                cache_channels=channels,
                sample_rate=int(sample_rate),
                mic_indices=self._cov_mic_indices,
                n_fft=n_fft,
                hop=hop,
            )
            if len(freqs) != len(self._cov_freqs):
                return []

        beam_stft = np.einsum(
            "bfm,mft->bft",
            np.conj(self._beam_weights),
            stft,
            optimize=True,
        )
        _, beam_audio_by_beam = signal.istft(
            beam_stft,
            fs=int(sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        beam_audio_by_beam = np.asarray(
            beam_audio_by_beam[:, : audio.shape[0]],
            dtype=np.float32,
        )
        if beam_audio_by_beam.shape[-1] < audio.shape[0]:
            pad_width = (
                (0, 0),
                (0, audio.shape[0] - beam_audio_by_beam.shape[-1]),
            )
            beam_audio_by_beam = np.pad(beam_audio_by_beam, pad_width)

        outputs: list[dict] = []
        for beam, beam_audio in zip(self.beams, beam_audio_by_beam, strict=True):
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
        return self._repair_channels(self._select_channels(audio, mic_indices))

    def _select_channels(
        self,
        audio: np.ndarray,
        mic_indices: tuple[int, ...],
    ) -> np.ndarray:
        channels = audio[:, list(mic_indices)].T.astype(np.float64, copy=False)
        return np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)

    def _repair_channels(
        self,
        channels: np.ndarray,
    ) -> np.ndarray:
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
        delays = -(positions_m @ direction) / float(self.cfg.speed_of_sound_m_s)
        steering = np.exp(-2j * np.pi * freqs[:, np.newaxis] * delays[np.newaxis, :])
        numerator = np.einsum("fmn,fn->fm", cov_inv, steering)
        denominator = np.einsum("fm,fm->f", np.conj(steering), numerator)
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        return numerator / denominator[:, np.newaxis]

    def _reset_stft_cache(self) -> None:
        self._stft_cache_channels = None
        self._stft_cache_spec = None
        self._stft_cache_key = None
        self._stft_last_reused_frames = 0

    def _stft_channels_cached(
        self,
        channels: np.ndarray,
        *,
        cache_channels: np.ndarray,
        sample_rate: int,
        mic_indices: tuple[int, ...],
        n_fft: int,
        hop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(sample_rate), tuple(mic_indices), int(n_fft), int(hop))
        step = self._matching_stft_cache_step(cache_channels, key)
        cached = self._stft_cache_spec
        if step is not None and cached is not None:
            freqs, spec = self._stft_channels_incremental(
                channels,
                cached,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop=hop,
                step=step,
            )
            self._store_stft_cache(cache_channels, spec, key)
            self._stft_cache_hits += 1
            return freqs, spec

        freqs, _, spec = signal.stft(
            channels,
            fs=int(sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        self._store_stft_cache(cache_channels, spec, key)
        self._stft_cache_misses += 1
        self._stft_last_reused_frames = 0
        return freqs, spec

    def _matching_stft_cache_step(
        self,
        channels: np.ndarray,
        key: tuple[int, tuple[int, ...], int, int],
    ) -> int | None:
        cached_channels = self._stft_cache_channels
        cached_spec = self._stft_cache_spec
        if (
            cached_channels is None
            or cached_spec is None
            or self._stft_cache_key != key
            or cached_channels.shape != channels.shape
        ):
            return None

        expected_steps: list[int] = []
        if self.cfg.incremental_step_samples is not None:
            expected_steps.append(int(self.cfg.incremental_step_samples))
        expected_steps.append(0)

        hop = key[3]
        frame_count = cached_spec.shape[-1]
        for step in expected_steps:
            if step < 0 or step >= channels.shape[-1]:
                continue
            if step % hop != 0:
                continue
            if step // hop >= frame_count:
                continue
            if step == 0:
                if np.array_equal(cached_channels, channels):
                    return 0
                continue
            if np.array_equal(cached_channels[:, step:], channels[:, :-step]):
                return step
        return None

    def _store_stft_cache(
        self,
        channels: np.ndarray,
        spec: np.ndarray,
        key: tuple[int, tuple[int, ...], int, int],
    ) -> None:
        self._stft_cache_channels = np.array(channels, dtype=np.float64, copy=True)
        self._stft_cache_spec = np.array(spec, dtype=np.complex128, copy=True)
        self._stft_cache_key = key

    def _stft_channels_incremental(
        self,
        channels: np.ndarray,
        cached: np.ndarray,
        *,
        sample_rate: int,
        n_fft: int,
        hop: int,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_count = cached.shape[-1]
        if step == 0:
            self._stft_last_reused_frames = frame_count
            freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
            return freqs, np.array(cached, dtype=np.complex128, copy=True)

        frame_shift = step // hop
        pad = n_fft // 2
        left_edge_frames = (pad + hop - 1) // hop
        right_safe_last = (channels.shape[-1] - pad) // hop
        reuse_start = min(left_edge_frames, frame_count)
        reuse_stop = min(frame_count, right_safe_last - frame_shift + 1)
        reused_frames = max(0, reuse_stop - reuse_start)

        spec = np.empty_like(cached)
        if reuse_start > 0:
            spec[:, :, :reuse_start] = self._stft_frame_range(
                channels,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop=hop,
                frame_start=0,
                frame_count=reuse_start,
            )[1]
        if reused_frames > 0:
            spec[:, :, reuse_start:reuse_stop] = cached[
                :, :, reuse_start + frame_shift : reuse_stop + frame_shift
            ]
        if reuse_stop < frame_count:
            spec[:, :, reuse_stop:] = self._stft_frame_range(
                channels,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop=hop,
                frame_start=reuse_stop,
                frame_count=frame_count - reuse_stop,
            )[1]

        self._stft_last_reused_frames = reused_frames
        freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
        return freqs, spec

    def _stft_frame_range(
        self,
        channels: np.ndarray,
        *,
        sample_rate: int,
        n_fft: int,
        hop: int,
        frame_start: int,
        frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if frame_count <= 0:
            freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
            return freqs, np.empty((channels.shape[0], n_fft // 2 + 1, 0), dtype=np.complex128)

        pad = n_fft // 2
        padded = np.pad(channels, ((0, 0), (pad, pad)), mode="constant")
        needed = n_fft + (frame_start + frame_count - 1) * hop
        if padded.shape[-1] < needed:
            padded = np.pad(padded, ((0, 0), (0, needed - padded.shape[-1])), mode="constant")
        offsets = (frame_start + np.arange(frame_count, dtype=np.int64)) * hop
        sample_idx = offsets[:, np.newaxis] + np.arange(n_fft, dtype=np.int64)
        frames = padded[:, sample_idx]
        window = signal.get_window("hann", n_fft, fftbins=True).astype(np.float64)
        spectrum = np.fft.rfft(
            frames * window[np.newaxis, np.newaxis, :],
            n=n_fft,
            axis=-1,
        ) / max(float(window.sum()), 1e-12)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
        return freqs, np.moveaxis(spectrum, 1, -1).astype(np.complex128, copy=False)

    def _weights_for_beams(
        self,
        cov_inv: np.ndarray,
        freqs: np.ndarray,
        positions_m: np.ndarray,
    ) -> np.ndarray:
        directions = np.stack(
            [
                direction_unit_vector(beam.azimuth_deg, beam.elevation_deg)
                for beam in self.beams
            ],
            axis=0,
        )
        delays = -(directions @ positions_m.T) / float(self.cfg.speed_of_sound_m_s)
        steering = np.exp(
            -2j * np.pi * freqs[np.newaxis, :, np.newaxis] * delays[:, np.newaxis, :]
        )
        numerator = np.einsum("fmn,bfn->bfm", cov_inv, steering, optimize=True)
        denominator = np.einsum(
            "bfm,bfm->bf",
            np.conj(steering),
            numerator,
            optimize=True,
        )
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        return numerator / denominator[:, :, np.newaxis]
