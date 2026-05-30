"""DSP-augmented DroneDetector — replaces one mel channel with DSP features.

The DSP feature extractor uses its own mel transform (separately configurable
hop_length) and projects features to the backbone mel dimension. When T_dsp ≠ T_mel,
linear interpolation aligns the time axes.

Usage:
    uv run audi-train --arch convnext_small --use-dsp-features
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.training.detector import DroneDetector


class DSPFeatureProjector(nn.Module):
    """Project DSP feature time series to the mel channel dimension.

    Per-timestep MLP → 1D depthwise conv for temporal smoothness.
    """

    def __init__(self, n_dsp: int, n_mels: int = 128, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dsp, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_mels),
        )
        self.smooth = nn.Conv1d(n_mels, n_mels, 3, padding=1, groups=n_mels)

    def forward(self, dsp: torch.Tensor) -> torch.Tensor:
        """Project [B, T, D] → [B, n_mels, T]."""
        return self.smooth(self.mlp(dsp).transpose(1, 2))


class DSPDroneDetector(DroneDetector):
    """DroneDetector where channel 3 is DSP features instead of duplicated mel.

    Input shape unchanged: [B, 3, n_mels, T]. Channels: mel, mel, DSP_projected.

    DSP features are computed from a separate mel spectrogram with its own
    hop_length (default 256, configurable via --dsp-hop-length).
    """

    def __init__(
        self,
        *,
        model: ModelConfig | None = None,
        mel: MelConfig | None = None,
        optimizer: OptimizerConfig | None = None,
        bin_names: list[str] | None = None,
        loss_type: str = "bce",
        label_smoothing: float = 0.0,
        per_bin_weights: bool = False,
        spec_augment_prob: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        dropout: float = 0.0,
        bn_momentum: float = 0.1,
        clip_seconds: float = 2.56,
        # ── DSP ──
        dsp_feature_sets: list[str] | None = None,
        dsp_hop_length: int = 256,
        dsp_f0_min: float = 125.0,
        dsp_f0_max: float = 350.0,
        dsp_f0_step: float = 2.0,
        dsp_n_harmonics: int = 12,
        dsp_noise_beta: float = 0.0001,
        dsp_stack_alpha: float = 0.50,
        dsp_projector_hidden: int = 64,
    ) -> None:
        super().__init__(
            model=model, mel=mel, optimizer=optimizer,
            bin_names=bin_names, loss_type=loss_type,
            label_smoothing=label_smoothing,
            per_bin_weights=per_bin_weights,
            spec_augment_prob=spec_augment_prob,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            dropout=dropout, bn_momentum=bn_momentum,
            clip_seconds=clip_seconds,
        )

        mel_cfg = mel or MelConfig()
        self._dsp_feature_sets = dsp_feature_sets or ["v3", "v4", "v5"]
        self._dsp_f0_min = dsp_f0_min
        self._dsp_f0_max = dsp_f0_max
        self._dsp_f0_step = dsp_f0_step
        self._dsp_n_harmonics = dsp_n_harmonics
        self._dsp_noise_beta = dsp_noise_beta
        self._dsp_stack_alpha = dsp_stack_alpha

        # Separate mel transform for DSP features (own hop)
        self._dsp_mel = T.MelSpectrogram(
            sample_rate=mel_cfg.sample_rate,
            n_fft=mel_cfg.n_fft,
            hop_length=dsp_hop_length,
            n_mels=mel_cfg.n_mels,
        )

        # Count feature dimensions
        n_dsp = 0
        if "v3" in self._dsp_feature_sets:
            n_dsp += 4
        if "v4" in self._dsp_feature_sets:
            n_dsp += 6
        if "v5" in self._dsp_feature_sets:
            n_dsp += 1
        self._n_dsp_features = n_dsp

        self._dsp_projector = DSPFeatureProjector(
            n_dsp=n_dsp, n_mels=mel_cfg.n_mels, hidden=dsp_projector_hidden,
        )

        # Pre-compute harmonic weights
        weights = 1.0 / torch.arange(1, dsp_n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("_dsp_hw", weights, persistent=False)

        # Pre-compute f0 candidates
        f0s = []
        f = dsp_f0_min
        while f <= dsp_f0_max + dsp_f0_step / 2:
            f0s.append(f)
            f += dsp_f0_step
        self._dsp_f0_list = f0s

    # ═══════════════════════════════════════════════════════════
    # DSP feature extraction (GPU, per-batch)
    # ═══════════════════════════════════════════════════════════

    def _dsp_freq_res(self, power: torch.Tensor) -> float:
        """Hz per frequency bin for the DSP mel transform."""
        return self._dsp_mel.sample_rate / 2 / power.shape[1]

    def _dsp_harmonic_folding(self, power: torch.Tensor) -> torch.Tensor:
        """Harmonic folding: max_f0 Σ w_h × power[f0×h]  → [B, T]."""
        B, n_freqs, T = power.shape
        freq_res = self._dsp_freq_res(power)
        device = power.device
        hw = self._dsp_hw
        scores = torch.zeros(B, T, device=device)

        for f0 in self._dsp_f0_list:
            fold = torch.zeros(B, T, device=device)
            for hi in range(len(hw)):
                bi = round(f0 * (hi + 1) / freq_res)
                if 0 <= bi < n_freqs:
                    fold = fold + hw[hi] * power[:, bi, :]
            scores = torch.maximum(scores, fold)

        return scores * 1000.0

    def _dsp_noise_floor(self, raw: torch.Tensor) -> torch.Tensor:
        """EMA noise floor: β·x_t + (1-β)·n_{t-1}."""
        beta = self._dsp_noise_beta
        B, T = raw.shape
        noise = torch.zeros(B, T, device=raw.device)
        est = raw[:, 0].clone()
        for t in range(T):
            est = beta * raw[:, t] + (1 - beta) * est
            noise[:, t] = est
        return noise

    def _dsp_stack(self, raw: torch.Tensor) -> torch.Tensor:
        """EMA stacking with noise-floor subtraction."""
        alpha = self._dsp_stack_alpha
        noise = self._dsp_noise_floor(raw)
        B, T = raw.shape
        stacked = torch.zeros(B, T, device=raw.device)
        state = torch.zeros(B, device=raw.device)
        for t in range(T):
            x = torch.clamp(raw[:, t] - noise[:, t], min=0)
            state = alpha * x + (1 - alpha) * state
            stacked[:, t] = state
        return stacked

    def _dsp_lattice(self, power: torch.Tensor) -> torch.Tensor:
        """V4 lattice features → [B, T, 6].  Fully vectorized over B, T, f0."""
        B, n_freqs, T = power.shape
        device = power.device
        freq_res = self._dsp_freq_res(power)
        n_h = self._dsp_n_harmonics
        hw = self._dsp_hw  # [n_h]  harmonic weights
        f0s = self._dsp_f0_list
        n_f0 = len(f0s)

        # ── Build harmonic bin indices: hb[f0, h] = bin  ──
        hb = torch.full((n_f0, n_h), -1, dtype=torch.long, device=device)
        for fi, f0 in enumerate(f0s):
            for hi in range(n_h):
                bi = round(f0 * (hi + 1) / freq_res)
                if 0 <= bi < n_freqs:
                    hb[fi, hi] = bi

        # ── Compute folding score: S[b, f0, t] = Σ w[h] * power[b, hb[f0,h], t]  ──
        # Gather power at all harmonic bins
        # power: [B, n_freqs, T];  hb: [n_f0, n_h]
        # We want: power_at_harms[b, f0, h, t] = power[b, hb[f0,h], t]

        # Build gather indices: for each f0×h, collect power[b, bin, t] across all t
        # Strategy: reshape hb to [n_f0*n_h], gather, then reshape back

        hb_flat = hb.reshape(-1)  # [n_f0 * n_h]
        valid_mask = hb_flat >= 0  # [n_f0 * n_h]

        # Gather power at valid bins → [B, n_valid, T]
        valid_bins = hb_flat[valid_mask]
        gathered = power[:, valid_bins, :]  # [B, n_valid, T]

        # Build full tensor with zeros at invalid positions
        flat_scores = torch.zeros(B, n_f0 * n_h, T, device=device)
        flat_scores[:, valid_mask, :] = gathered

        # Reshape to [B, n_f0, n_h, T] and apply weights
        scores_per_harm = flat_scores.reshape(B, n_f0, n_h, T)  # [B, n_f0, n_h, T]
        weighted = scores_per_harm * hw[None, None, :, None]     # [B, n_f0, n_h, T]
        f0_scores = weighted.sum(dim=2)  # [B, n_f0, T]  — folding score per candidate

        # ── Best f0 per batch×time: best_idx[b, t]  ──
        best_idx = f0_scores.argmax(dim=1)  # [B, T]

        # ── Collect harmonic peaks for the best f0  ──
        # pf[b, t, h] = f0[best_idx] * (h+1)    — peak frequency
        # pa[b, t, h] = power[b, hb[best_idx, h], t]  — peak amplitude

        # Gather best-f0 bin indices: best_hb[b, t, h] = hb[best_idx[b,t], h]
        best_hb = hb[best_idx]  # [B, T, n_h]  — advanced indexing

        # Gather amplitudes at those bins
        # For each (b, t, h), we need power[b, best_hb[b,t,h], t]
        # Use gather along freq dim
        b_idx = torch.arange(B, device=device)[:, None, None].expand(-1, T, n_h)
        t_idx = torch.arange(T, device=device)[None, :, None].expand(B, -1, n_h)
        pa = power[b_idx, best_hb.clamp(min=0), t_idx]  # [B, T, n_h]
        pa = torch.where(best_hb >= 0, pa, torch.zeros_like(pa))

        # Peak frequencies
        f0_best = torch.tensor(f0s, device=device)[best_idx]  # [B, T]
        h_range = torch.arange(1, n_h + 1, device=device).float()  # [n_h]
        pf = f0_best.unsqueeze(-1) * h_range  # [B, T, n_h]

        # ── Aggregate features  ──
        energy = pa.sum(dim=2)                        # [B, T]
        centroid = (pf * pa).sum(dim=2) / (energy + 1e-8)  # [B, T]
        n_found = (best_hb >= 0).float().sum(dim=2)   # [B, T]
        spread = pf.std(dim=2)                        # [B, T]
        structure = energy * n_found / 20.0            # [B, T]
        log_energy = torch.log1p(energy)                # [B, T]

        feats = torch.stack([energy, centroid, n_found, spread, structure, log_energy], dim=2)
        # [B, T, 6]

        # Per-column normalize
        for c in range(6):
            m = feats[:, :, c].max()
            if m > 0:
                feats[:, :, c] = feats[:, :, c] / m

        return feats

    def _dsp_snr(self, raw: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw / (noise + 1e-8), 0, 100)

    # ═══════════════════════════════════════════════════════════
    # Full DSP extraction
    # ═══════════════════════════════════════════════════════════

    def _extract_dsp(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T_audio] → dsp [B, T_dsp, D]."""
        mel = self._dsp_mel(wav)
        raw = self._dsp_harmonic_folding(mel)
        stacked = self._dsp_stack(raw)
        parts: list[torch.Tensor] = []

        if "v3" in self._dsp_feature_sets:
            vel = F.pad(stacked[:, 1:] - stacked[:, :-1], (0, 1))
            accel = F.pad(vel[:, 1:] - vel[:, :-1], (0, 1))
            parts += [raw, stacked, vel, accel]

        if "v4" in self._dsp_feature_sets:
            parts.append(self._dsp_lattice(mel))

        if "v5" in self._dsp_feature_sets:
            noise = self._dsp_noise_floor(raw)
            parts.append(self._dsp_snr(raw, noise).unsqueeze(-1))

        # Each part is [B, T] or [B, T, D]; unify to [B, T, D]
        unified = []
        for p in parts:
            if p.ndim == 2:
                p = p.unsqueeze(-1)
            unified.append(p)

        return torch.cat(unified, dim=-1)

    # ═══════════════════════════════════════════════════════════
    # Override _to_mel
    # ═══════════════════════════════════════════════════════════

    def _to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, T_audio] → [B, 3, n_mels, T_mel].

        Channel 0-1: standard mel (dB + norm).
        Channel 2:   DSP features projected to mel dimension.
        """
        # Standard mel
        if self._use_pcen:
            mel = self._pcen(self._mel_transform(wav))
        else:
            mel = self._to_db(self._mel_transform(wav))
            if self._mel_mean is not None and self._mel_std is not None:
                mel = (mel - self._mel_mean) / self._mel_std
        # mel: [B, n_mels, T_mel]

        # DSP features
        dsp = self._extract_dsp(wav)                     # [B, T_dsp, D]
        dsp_proj = self._dsp_projector(dsp)               # [B, n_mels, T_dsp]

        # Align time axes (interpolate DSP to mel if hops differ)
        T_mel = mel.shape[2]
        if dsp_proj.shape[2] != T_mel:
            dsp_proj = F.interpolate(dsp_proj, size=T_mel,
                                      mode="linear", align_corners=False)

        # Stack: mel, mel, dsp → [B, 3, n_mels, T_mel]
        spec = torch.stack([mel, mel.clone(), dsp_proj], dim=1)

        # SpecAugment
        if (self._freq_mask is not None and self._time_mask is not None
                and self.training and random.random() < self.spec_augment_prob):
            for _ in range(2):
                spec = self._freq_mask(spec)
            for _ in range(2):
                spec = self._time_mask(spec)

        return spec
