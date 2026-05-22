"""DSP feature extraction: compute physics-based features from audio.

Extends the harmonic folding detector with V3/V4/V5-style feature
trajectories. Output can be merged with mel spectrograms as extra
input channels for ML models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DSPFeatureConfig:
    """Which DSP feature sets to extract."""

    # ── Harmonic folding params ──
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    f0_min: float = 125.0
    f0_max: float = 350.0
    f0_step: float = 2.0
    n_harmonics: int = 12

    # ── Stacking params ──
    noise_beta: float = 0.0001   # noise floor adaptation rate
    stack_alpha: float = 0.50    # EMA coefficient

    # ── Feature sets ──
    enable_v3: bool = True   # [harmonic, stacked, velocity, acceleration]
    enable_v4: bool = True   # [energy, centroid, density, spread, structure, log_energy]
    enable_v5: bool = True   # [harmonic, stacked, velocity, snr_estimate]


class DSPFeatureExtractor:
    """Extract DSP feature trajectories from audio.

    Usage:
        extractor = DSPFeatureExtractor()
        features = extractor.extract(audio)  # [n_frames, n_features]
        merged = extractor.merge_with_mel(mel_spec, features)  # [n_features+3, n_mels, n_frames]
    """

    def __init__(self, cfg: DSPFeatureConfig | None = None):
        self.cfg = cfg or DSPFeatureConfig()

    @property
    def hop_s(self) -> float:
        return self.cfg.hop_length / self.cfg.sample_rate

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names produced by extract()."""
        names = []
        if self.cfg.enable_v3:
            names += ["harm_score", "harm_stacked", "harm_vel", "harm_accel"]
        if self.cfg.enable_v4:
            names += ["energy", "centroid", "density", "spread", "structure", "log_energy"]
        if self.cfg.enable_v5:
            # v5 shares harm_score/harm_stacked/harm_vel with v3 — deduplicate
            pass
            names += ["snr_estimate"]
        return names

    # ── Core harmonic folding ──────────────────────────────────

    def harmonic_folding(self, power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Raw harmonic folding score per frame.

        For each f0 candidate, accumulate weighted energy at
        harmonic multiples, then take max over candidates.

        Args:
            power: [n_freqs, n_frames] STFT power.
            freqs: [n_freqs] frequency values.

        Returns:
            raw_score[f] shape [n_frames].
        """
        cfg = self.cfg
        n_frames = power.shape[1]
        freq_res = freqs[1] - freqs[0]
        f0_candidates = np.arange(cfg.f0_min, cfg.f0_max + cfg.f0_step / 2, cfg.f0_step)
        weights = 1.0 / np.arange(1, cfg.n_harmonics + 1, dtype=np.float64)

        scores = np.zeros(n_frames, dtype=np.float32)
        for f0 in f0_candidates:
            fold = np.zeros(n_frames, dtype=np.float64)
            for h_idx in range(cfg.n_harmonics):
                h_freq = f0 * (h_idx + 1)
                bi = int(round(h_freq / freq_res))
                if 0 <= bi < power.shape[0]:
                    fold += weights[h_idx] * power[bi, :]
            scores = np.maximum(scores, fold.astype(np.float32))

        return scores * 1000.0  # scale to reasonable range

    def noise_floor(self, raw: np.ndarray) -> np.ndarray:
        """Estimate slow-adapting noise floor."""
        beta = self.cfg.noise_beta
        nf = np.zeros(len(raw), dtype=np.float64)
        est = float(raw[0])
        for t in range(len(raw)):
            est = beta * raw[t] + (1 - beta) * est
            nf[t] = est
        return nf.astype(np.float32)

    def temporal_stack(self, raw: np.ndarray) -> np.ndarray:
        """EMA evidence accumulation with noise-floor subtraction."""
        alpha = self.cfg.stack_alpha
        noise = self.noise_floor(raw)
        stacked = np.zeros(len(raw), dtype=np.float64)
        state = 0.0
        for t in range(len(raw)):
            x = max(raw[t] - noise[t], 0)
            state = alpha * x + (1 - alpha) * state
            stacked[t] = state
        return stacked.astype(np.float32)

    # ── V4: harmonic lattice features (per-frame) ──────────────

    def lattice_features(self, power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Extract V4-style harmonic lattice features per frame.

        For each frame, finds all harmonic peaks and computes:
        energy, spectral centroid, harmonic density, frequency spread,
        structure score, log-energy.

        Returns:
            [n_frames, 6]  (energy, centroid, density, spread, structure, log_energy)
        """
        cfg = self.cfg
        n_freqs, n_frames = power.shape
        freq_res = freqs[1] - freqs[0]
        f0_candidates = np.arange(cfg.f0_min, cfg.f0_max + cfg.f0_step / 2, cfg.f0_step)

        features = np.zeros((n_frames, 6), dtype=np.float32)

        for t in range(n_frames):
            frame = power[:, t]

            # Find best f0 by harmonic folding
            best_score = 0.0
            best_f0 = cfg.f0_min
            for f0 in f0_candidates:
                score = 0.0
                for h in range(1, cfg.n_harmonics + 1):
                    bi = int(round(f0 * h / freq_res))
                    if 0 <= bi < n_freqs:
                        score += frame[bi] / h
                if score > best_score:
                    best_score = score
                    best_f0 = f0

            # Collect harmonic peaks
            peaks_f = []
            peaks_a = []
            for h in range(1, cfg.n_harmonics + 1):
                bi = int(round(best_f0 * h / freq_res))
                if 0 <= bi < n_freqs:
                    peaks_f.append(best_f0 * h)
                    peaks_a.append(frame[bi])

            if not peaks_f:
                features[t] = [0, 0, 0, 0, 0, 0]
                continue

            f_arr = np.array(peaks_f)
            a_arr = np.array(peaks_a)

            energy = float(np.sum(a_arr))
            centroid = float(np.sum(f_arr * a_arr) / (energy + 1e-8))
            density = len(peaks_f)
            spread = float(np.std(f_arr)) if density > 1 else 0.0
            structure = energy * density / 20.0
            log_energy = float(np.log1p(energy))

            features[t] = [energy, centroid, density, spread, structure, log_energy]

        # Normalize each column to [0, 1] range
        for col in range(6):
            col_max = features[:, col].max()
            if col_max > 0:
                features[:, col] /= col_max

        return features

    # ── SNR estimate (V5) ──────────────────────────────────────

    def snr_estimate(self, raw: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Estimate per-frame SNR: raw_score / noise_floor."""
        return np.clip(raw / (noise + 1e-8), 0, 100).astype(np.float32)

    # ── Full extraction ────────────────────────────────────────

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract all DSP features from audio.

        Args:
            audio: [samples] mono float32.

        Returns:
            features[f, d] shape [n_frames, n_features].
            Column order matches self.feature_names.
        """
        import scipy.signal

        cfg = self.cfg

        # Preprocess
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-8)
        if rms > 0:
            audio = audio / rms

        # BPF 300-3500 Hz
        sos = scipy.signal.butter(4, [300 / (cfg.sample_rate / 2),
                                       3500 / (cfg.sample_rate / 2)],
                                   btype="band", output="sos")
        audio = scipy.signal.sosfilt(sos, audio)

        # STFT
        win = scipy.signal.get_window("hann", cfg.n_fft)
        f, t, Zxx = scipy.signal.spectrogram(
            audio, fs=cfg.sample_rate, window=win,
            nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length,
            mode="magnitude"
        )
        power = Zxx ** 2

        # Harmonic folding
        raw = self.harmonic_folding(power, f)
        noise = self.noise_floor(raw)
        stacked = self.temporal_stack(raw)

        # Build feature matrix
        feature_list = []

        # V3: [harmonic, stacked, velocity, acceleration]
        if cfg.enable_v3:
            vel = np.gradient(stacked)
            accel = np.gradient(vel)
            feature_list += [
                raw[:, None],
                stacked[:, None],
                vel[:, None],
                accel[:, None],
            ]

        # V4: lattice features [6 cols]
        if cfg.enable_v4:
            lattice = self.lattice_features(power, f)
            feature_list.append(lattice)

        # V5: SNR estimate [1 col]
        if cfg.enable_v5:
            snr = self.snr_estimate(raw, noise)
            feature_list.append(snr[:, None])

        features = np.concatenate(feature_list, axis=1).astype(np.float32)
        return features

    # ── Merge with mel spectrogram ─────────────────────────────

    def merge_with_mel(
        self, mel_spec: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Merge DSP features as extra channels on mel spectrogram.

        The mel spec is [n_mels, T_mel] or [C, n_mels, T_mel].
        Features are [T_feat, D]. Time axes are interpolated to match.

        Returns:
            [D + C, n_mels, T]  — mel channels + DSP feature channels.
        """
        mel = np.atleast_3d(mel_spec)
        if mel.shape[0] == 1 and mel.shape[1] > mel.shape[2]:
            mel = mel[0]  # [n_mels, T]
        elif mel.shape[0] == 3:
            pass  # [3, n_mels, T]
        else:
            mel = mel.squeeze()

        if mel.ndim == 2:
            mel = mel[np.newaxis, :, :]  # [1, n_mels, T]

        n_mel_frames = mel.shape[2]
        n_dsp_frames = features.shape[0]
        n_dsp_feats = features.shape[1]

        # Interpolate DSP features to mel time grid
        if n_dsp_frames != n_mel_frames:
            # Linear interpolation along time
            dsp_aligned = np.zeros((n_mel_frames, n_dsp_feats), dtype=np.float32)
            for d in range(n_dsp_feats):
                dsp_aligned[:, d] = np.interp(
                    np.linspace(0, n_dsp_frames - 1, n_mel_frames),
                    np.arange(n_dsp_frames),
                    features[:, d],
                )
        else:
            dsp_aligned = features

        # Tile DSP features across mel bins
        dsp_tiled = np.tile(
            dsp_aligned.T[:, np.newaxis, :], (1, mel.shape[1], 1)
        )  # [D, n_mels, T]

        merged = np.concatenate([mel, dsp_tiled], axis=0)
        return merged.astype(np.float32)
