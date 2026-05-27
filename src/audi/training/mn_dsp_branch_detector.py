"""Two-branch MN+DSP detector — separate DSP branch fused with MN embedding.

Unlike DSPDroneDetector (which injects DSP features as mel channels and
processes everything through a single backbone), this model uses two separate
branches:

  WAV [B, T_audio]
    ├── Mel branch: mel → MN backbone → [B, mel_feat_dim]
    └── DSP branch: DSP mel → folding → features → Conv1D encoder → [B, dsp_emb_dim]
         └── Fusion: concat → MLP → [B, 1]

The hypothesis is that DSP features encode physical properties (harmonic
structure, modulation) that are complementary to the learned mel-spectrogram
representations, and a separate branch processes them more effectively than
channel injection.

Usage:
    python scripts/train.py --arch mn10_as --use-dsp-branch --dsp-feature-sets v3,v4,v5
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.training.detector import DroneDetector


class DSPBranchEncoder(nn.Module):
    """Conv1D encoder for DSP feature trajectories → embedding vector.

    Processes the per-frame DSP feature time series through strided
    convolutions, then global-average-pools to a fixed-size embedding.

    Args:
        n_dsp: Number of DSP feature channels (e.g. 11 for v3+v4+v5).
        dsp_emb_dim: Output embedding dimension.
        n_mels: For reference only (unused in encoder).
    """

    def __init__(
        self,
        n_dsp: int,
        dsp_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        c = dsp_emb_dim
        self.conv = nn.Sequential(
            nn.Conv1d(n_dsp, c // 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c // 2, c // 2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c // 2, c, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Linear(c, c)

    def forward(self, dsp: torch.Tensor) -> torch.Tensor:
        """dsp [B, T, D] → embedding [B, dsp_emb_dim]."""
        x = dsp.transpose(1, 2)  # [B, D, T]
        x = self.conv(x)          # [B, dsp_emb_dim, T']
        x = x.mean(dim=-1)        # [B, dsp_emb_dim]
        return F.relu(self.project(x))


class MNBranchDSPDetector(DroneDetector):
    """Two-branch detector: MN backbone for mel + DSP encoder for harmonic features.

    Architecture:
        mel [B, 1, n_mels, T] → MN backbone → mel_emb  [B, mel_feat_dim]
        dsp [B, T_dsp, D]     → DSPBranchEncoder  → dsp_emb  [B, dsp_emb_dim]
        fusion = concat(mel_emb, dsp_emb) → MLP → [B, 1]

    Args:
        model: Backbone config (must be an EfficientAT MN/DyMN model).
        mel: Mel spectrogram config.
        optimizer: Optimizer config.
        bin_names: SNR bin names for per-bin eval.
        dsp_feature_sets: Which DSP features to compute (e.g. ['v3','v4','v5']).
        dsp_hop_length: Hop length for the DSP mel transform.
        dsp_emb_dim: DSP branch output embedding dimension.
        fusion_hidden: Hidden dim of the fusion MLP.
        *args, **kwargs: Passed to DroneDetector.__init__.
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
        # ── DSP branch ──
        dsp_feature_sets: list[str] | None = None,
        dsp_hop_length: int = 256,
        dsp_f0_min: float = 125.0,
        dsp_f0_max: float = 350.0,
        dsp_f0_step: float = 2.0,
        dsp_n_harmonics: int = 12,
        dsp_noise_beta: float = 0.0001,
        dsp_stack_alpha: float = 0.50,
        dsp_emb_dim: int = 256,
        fusion_hidden: int = 512,
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

        # Count DSP feature dimensions
        n_dsp = 0
        if "v3" in self._dsp_feature_sets:
            n_dsp += 4
        if "v4" in self._dsp_feature_sets:
            n_dsp += 6
        if "v5" in self._dsp_feature_sets:
            n_dsp += 1

        # DSP branch encoder
        self._dsp_encoder = DSPBranchEncoder(
            n_dsp=n_dsp,
            dsp_emb_dim=dsp_emb_dim,
        )

        # Determine backbone feature dimension via a test forward pass.
        # EfficientAT backbones return (logits, features) tuples.
        dummy_mel = torch.zeros(2, 1, mel_cfg.n_mels, 100)
        with torch.no_grad():
            raw_output = self.backbone.backbone(dummy_mel)
        if isinstance(raw_output, tuple) and len(raw_output) == 2:
            mel_feat_dim = raw_output[1].shape[-1]
        elif isinstance(raw_output, torch.Tensor):
            mel_feat_dim = raw_output.shape[-1]
        else:
            raise TypeError(
                f"Unexpected backbone output type: {type(raw_output)}. "
                "Expected tuple (logits, features) or Tensor."
            )

        # Fusion head
        fused_dim = mel_feat_dim + dsp_emb_dim
        self._fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, 1),
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
    # DSP feature extraction (mirrors DSPDroneDetector)
    # ═══════════════════════════════════════════════════════════

    def _dsp_freq_res(self, power: torch.Tensor) -> float:
        return self._dsp_mel.sample_rate / 2 / power.shape[1]

    def _dsp_harmonic_folding(self, power: torch.Tensor) -> torch.Tensor:
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
        beta = self._dsp_noise_beta
        B, T = raw.shape
        noise = torch.zeros(B, T, device=raw.device)
        est = raw[:, 0].clone()
        for t in range(T):
            est = beta * raw[:, t] + (1 - beta) * est
            noise[:, t] = est
        return noise

    def _dsp_stack(self, raw: torch.Tensor) -> torch.Tensor:
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
        """V4 lattice features → [B, T, 6]. Fully vectorized."""
        B, n_freqs, T = power.shape
        device = power.device
        freq_res = self._dsp_freq_res(power)
        n_h = self._dsp_n_harmonics
        hw = self._dsp_hw
        f0s = self._dsp_f0_list
        n_f0 = len(f0s)

        hb = torch.full((n_f0, n_h), -1, dtype=torch.long, device=device)
        for fi, f0 in enumerate(f0s):
            for hi in range(n_h):
                bi = round(f0 * (hi + 1) / freq_res)
                if 0 <= bi < n_freqs:
                    hb[fi, hi] = bi

        hb_flat = hb.reshape(-1)
        valid_mask = hb_flat >= 0
        valid_bins = hb_flat[valid_mask]
        gathered = power[:, valid_bins, :]

        flat_scores = torch.zeros(B, n_f0 * n_h, T, device=device)
        flat_scores[:, valid_mask, :] = gathered

        scores_per_harm = flat_scores.reshape(B, n_f0, n_h, T)
        weighted = scores_per_harm * hw[None, None, :, None]
        f0_scores = weighted.sum(dim=2)
        best_idx = f0_scores.argmax(dim=1)

        best_hb = hb[best_idx]

        b_idx = torch.arange(B, device=device)[:, None, None].expand(-1, T, n_h)
        t_idx = torch.arange(T, device=device)[None, :, None].expand(B, -1, n_h)
        pa = power[b_idx, best_hb.clamp(min=0), t_idx]
        pa = torch.where(best_hb >= 0, pa, torch.zeros_like(pa))

        f0_best = torch.tensor(f0s, device=device)[best_idx]
        h_range = torch.arange(1, n_h + 1, device=device).float()
        pf = f0_best.unsqueeze(-1) * h_range

        energy = pa.sum(dim=2)
        centroid = (pf * pa).sum(dim=2) / (energy + 1e-8)
        n_found = (best_hb >= 0).float().sum(dim=2)
        spread = pf.std(dim=2)
        structure = energy * n_found / 20.0
        log_energy = torch.log1p(energy)

        feats = torch.stack([energy, centroid, n_found, spread, structure, log_energy], dim=2)

        for c in range(6):
            m = feats[:, :, c].max()
            if m > 0:
                feats[:, :, c] = feats[:, :, c] / m

        return feats

    def _dsp_snr(self, raw: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw / (noise + 1e-8), 0, 100)

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

        unified = []
        for p in parts:
            if p.ndim == 2:
                p = p.unsqueeze(-1)
            unified.append(p)

        return torch.cat(unified, dim=-1)

    # ═══════════════════════════════════════════════════════════
    # Two-branch forward
    # ═══════════════════════════════════════════════════════════

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Two-branch forward: mel → MN + dsp → encoder → fusion → logit.

        Args:
            wav: [B, T_audio]

        Returns:
            Logits [B] (1-dim per sample).
        """
        # ── Mel branch: standard mel → MN backbone features ──
        mel_1ch = self._mel_transform(wav)  # [B, n_mels, T_mel]
        if self._use_pcen:
            mel_1ch = self._pcen(mel_1ch)
        else:
            mel_1ch = self._to_db(mel_1ch)
            if self._mel_mean is not None and self._mel_std is not None:
                mel_1ch = (mel_1ch - self._mel_mean) / self._mel_std
        # Expand to 1 channel + run MN backbone to get features
        mel_input = mel_1ch.unsqueeze(1)  # [B, 1, n_mels, T_mel]
        _, mel_features = self.backbone.backbone(mel_input)  # [B, mel_feat_dim]

        # ── SpecAugment on mel (before DSP, training only) ──
        spec_3ch = mel_1ch.unsqueeze(1).expand(-1, 3, -1, -1)
        if (self._freq_mask is not None and self._time_mask is not None
                and self.training and random.random() < self.spec_augment_prob):
            for _ in range(2):
                spec_3ch = self._freq_mask(spec_3ch)
            for _ in range(2):
                spec_3ch = self._time_mask(spec_3ch)

        # ── DSP branch ──
        dsp = self._extract_dsp(wav)            # [B, T_dsp, D]
        dsp_emb = self._dsp_encoder(dsp)         # [B, dsp_emb_dim]

        # ── Fusion ──
        fused = torch.cat([mel_features, dsp_emb], dim=-1)  # [B, mel_feat + dsp_emb]
        return self._fusion_head(fused).squeeze(1)           # [B]

    # ═══════════════════════════════════════════════════════════
    # Training step override — bypass the parent's mel→backbone path
    # ═══════════════════════════════════════════════════════════

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        if len(batch) == 3:
            wav, label, bin_idx = batch
        else:
            wav, label = batch
            bin_idx = None

        # For mixup/cutmix, we need spec. Compute mel-only and apply.
        # This won't mix DSP features — only the mel branch gets mixed.
        # That's OK: the DSP features reflect the original (unmixed) audio.
        mel_1ch = self._mel_transform(wav)
        if self._use_pcen:
            mel_1ch = self._pcen(mel_1ch)
        else:
            mel_1ch = self._to_db(mel_1ch)
            if self._mel_mean is not None and self._mel_std is not None:
                mel_1ch = (mel_1ch - self._mel_mean) / self._mel_std
        spec = mel_1ch.unsqueeze(1).expand(-1, 3, -1, -1)

        label, bin_idx, spec_mixed = self._apply_mixup_cutmix_branch(spec, label, bin_idx)
        label_mixed = label  # from mixup, may be soft

        # Mel branch from (possibly mixed) spec
        mel_input = spec_mixed[:, :1]  # [B, 1, n_mels, T]
        _, mel_features = self.backbone.backbone(mel_input)

        # DSP branch from original (unmixed) audio
        dsp = self._extract_dsp(wav)
        dsp_emb = self._dsp_encoder(dsp)

        fused = torch.cat([mel_features, dsp_emb], dim=-1)
        logits = self._fusion_head(fused).squeeze(1)

        loss = self._compute_loss(logits, label_mixed, bin_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _apply_mixup_cutmix_branch(
        self,
        spec: torch.Tensor,
        labels: torch.Tensor,
        bin_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """MixUp/CutMix on 3-channel mel spec, returning mixed spec."""
        if self._mixup_alpha <= 0 and self._cutmix_alpha <= 0:
            return labels, bin_idx, spec

        use_cutmix = self._cutmix_alpha > 0 and (
            self._mixup_alpha <= 0 or random.random() < 0.5
        )
        B = spec.size(0)
        idx = torch.randperm(B, device=spec.device)

        if use_cutmix:
            import numpy as np
            lam = float(np.random.beta(self._cutmix_alpha, self._cutmix_alpha))
            lam = max(lam, 1.0 - lam)
            _, _, H, W = spec.shape
            cut_h = max(1, min(int(H * np.sqrt(1.0 - lam)), H - 1))
            cut_w = max(1, min(int(W * np.sqrt(1.0 - lam)), W - 1))
            cy = random.randint(0, H - cut_h)
            cx = random.randint(0, W - cut_w)
            lam = 1.0 - (cut_h * cut_w) / (H * W)
            spec_mix = spec.clone()
            spec_mix[:, :, cy : cy + cut_h, cx : cx + cut_w] = spec[
                idx, :, cy : cy + cut_h, cx : cx + cut_w
            ]
            labels_mix = lam * labels + (1.0 - lam) * labels[idx]
        else:
            import numpy as np
            lam = float(np.random.beta(self._mixup_alpha, self._mixup_alpha))
            lam = max(lam, 1.0 - lam)
            spec_mix = lam * spec + (1.0 - lam) * spec[idx]
            labels_mix = lam * labels + (1.0 - lam) * labels[idx]

        return labels_mix, bin_idx, spec_mix
