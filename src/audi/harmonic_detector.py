"""Harmonic folding detector — classical DSP pipeline for drone detection.

No ML required. Designed for weak-signal long-range detection via harmonic
structure extraction + temporal energy stacking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal


@dataclass
class HarmonicConfig:
    """Tunable parameters for the harmonic folding pipeline."""

    # ── Audio ──
    sample_rate: int = 48000
    channels: int = 1

    # ── Preprocessing ──
    highpass_hz: float = 80.0
    bandpass_low: float = 300.0
    bandpass_high: float = 3500.0

    # ── STFT ──
    n_fft: int = 2048
    hop_length: int = 1024
    window: str = "hann"

    # ── Harmonic folding ──
    f0_min: float = 125.0
    f0_max: float = 350.0
    f0_step: float = 2.0       # Hz between candidate f0s
    n_harmonics: int = 12       # h = 1…N
    harmonic_weight: str = "1/h"  # "1/h", "uniform", "1/sqrt(h)"

    # ── Temporal stacking ──
    noise_beta: float = 0.01    # slow noise floor adaptation
    stack_alpha: float = 0.15   # EMA coefficient for evidence accumulation

    # ── Detection ──
    threshold: float = 2.5      # detection threshold on stacked score
    score_scale: float = 1000.0  # scale raw scores to reasonable range

    # ── Output ──
    hop_s: float | None = None  # computed: hop_length / sample_rate


class HarmonicFoldingDetector:
    """Full pipeline: raw audio → stacked drone likelihood score."""

    def __init__(self, cfg: HarmonicConfig | None = None):
        self.cfg = cfg or HarmonicConfig()
        self._noise_level: float = 0.0
        self._stack: float = 0.0
        self._f0_bins: np.ndarray | None = None
        self._harmonic_weights: np.ndarray | None = None

    @property
    def hop_s(self) -> float:
        return self.cfg.hop_length / self.cfg.sample_rate

    # ── Preprocessing ──────────────────────────────────────────

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Highpass + bandpass + RMS normalize + per-frame normalize.

        Args:
            audio: [samples] or [samples, channels]

        Returns:
            [samples] mono float32
        """
        sr = self.cfg.sample_rate
        audio = np.atleast_1d(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # RMS normalize
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-8)
        if rms > 0:
            audio = audio / rms

        # Highpass
        sos_hp = scipy.signal.butter(4, self.cfg.highpass_hz / (sr / 2),
                                      btype="high", output="sos")
        audio = scipy.signal.sosfilt(sos_hp, audio)

        # Bandpass
        sos_bp = scipy.signal.butter(4, [self.cfg.bandpass_low / (sr / 2),
                                          self.cfg.bandpass_high / (sr / 2)],
                                      btype="band", output="sos")
        audio = scipy.signal.sosfilt(sos_bp, audio)

        return audio.astype(np.float32)

    # ── STFT ───────────────────────────────────────────────────

    def stft_power(self, audio: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform → power spectrogram.

        Returns:
            power[freq, time]  —  shape [n_fft//2+1, n_frames]
        """
        n_fft = self.cfg.n_fft
        hop = self.cfg.hop_length
        win = scipy.signal.get_window(self.cfg.window, n_fft)
        # scipy spectrogram returns [freq, time], use mode='magnitude' and square
        f, t, Zxx = scipy.signal.spectrogram(
            audio, fs=self.cfg.sample_rate, window=win,
            nperseg=n_fft, noverlap=n_fft - hop, mode="magnitude"
        )
        return f, t, Zxx ** 2

    # ── Harmonic folding ───────────────────────────────────────

    def _build_harmonic_weights(self) -> np.ndarray:
        """Pre-compute harmonic weights [h=0…N-1]."""
        h = np.arange(1, self.cfg.n_harmonics + 1, dtype=np.float64)
        w = self.cfg.harmonic_weight
        if w == "1/h":
            return 1.0 / h
        elif w == "uniform":
            return np.ones_like(h)
        elif w == "1/sqrt(h)":
            return 1.0 / np.sqrt(h)
        else:
            return 1.0 / h

    def harmonic_score(self, power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """For each time frame, compute max harmonic folding score.

        Args:
            power: [n_freqs, n_frames]
            freqs: [n_freqs]  —  frequency values per bin

        Returns:
            raw_score[t]  shape [n_frames]
        """
        if self._harmonic_weights is None:
            self._harmonic_weights = self._build_harmonic_weights()

        weights = self._harmonic_weights
        n_harmonics = self.cfg.n_harmonics
        n_frames = power.shape[1]
        freq_res = freqs[1] - freqs[0]  # Hz per bin

        # Candidate f0 grid
        f0_candidates = np.arange(
            self.cfg.f0_min, self.cfg.f0_max + self.cfg.f0_step / 2,
            self.cfg.f0_step
        )

        scores = np.zeros(n_frames, dtype=np.float32)

        for f0 in f0_candidates:
            # Accumulate energy at harmonic multiples
            fold = np.zeros(n_frames, dtype=np.float64)
            for h_idx in range(n_harmonics):
                h_freq = f0 * (h_idx + 1)
                # Find nearest freq bin
                bin_idx = int(round(h_freq / freq_res))
                if 0 <= bin_idx < power.shape[0]:
                    fold += weights[h_idx] * power[bin_idx, :]

            # Max over f0 candidates
            scores = np.maximum(scores, fold.astype(np.float32))

        return scores * self.cfg.score_scale

    # ── Temporal stacking ──────────────────────────────────────

    def stack(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply noise-floor-adaptive temporal energy stacking.

        Returns:
            stacked_score[t]  shape [n_frames]
        """
        beta = self.cfg.noise_beta
        alpha = self.cfg.stack_alpha
        n = len(raw_scores)

        noise = np.zeros(n, dtype=np.float64)
        stacked = np.zeros(n, dtype=np.float64)

        noise_est = float(raw_scores[0])
        stack_est = 0.0

        for t in range(n):
            # Update noise floor
            noise_est = beta * raw_scores[t] + (1 - beta) * noise_est
            noise[t] = noise_est

            # Normalize
            x = raw_scores[t] - noise_est

            # Stack
            stack_est = alpha * max(x, 0) + (1 - alpha) * stack_est
            stacked[t] = stack_est

        return stacked.astype(np.float32)

    # ── Full pipeline ──────────────────────────────────────────

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run full pipeline: audio → raw_score + stacked_score.

        Returns:
            (raw_score[t], stacked_score[t], times[t])
        """
        audio = self.preprocess(audio)
        freqs, times, power = self.stft_power(audio)
        raw = self.harmonic_score(power, freqs)
        stacked = self.stack(raw)
        return raw, stacked, times

    def detect(self, audio: np.ndarray) -> np.ndarray:
        """Binary detection: True where stacked_score > threshold."""
        _, stacked, _ = self.process(audio)
        return stacked > self.cfg.threshold

    def reset(self):
        """Reset internal state (noise floor, stack)."""
        self._noise_level = 0.0
        self._stack = 0.0
