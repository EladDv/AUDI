"""DroneDetector — PyTorch Lightning module for binary drone detection."""

from __future__ import annotations

import random
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.model import build_model
from audi.training.validation import (
    compute_calibration,
    compute_pr_curve,
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
    split_by_bin,
    tpr_at_fpr,
)


class PCEN(torch.nn.Module):
    """Per-Channel Energy Normalization (PyTorch).

    Smooths each mel bin independently via exponential moving average,
    then applies AGC + noise floor. Matches librosa.pcen with
    ``gain=1.0, bias=0.0, power=0.0`` (root compression only).

    Args:
        s: Smoothing coefficient (EMA alpha). Lower = slower adaptation.
        alpha: AGC exponent. <1 compresses dynamic range.
        delta: Noise floor bias.
        r: Root compression exponent.
        eps: Numerical stability.
    """

    def __init__(
        self,
        s: float = 0.025,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps

    def forward(self, mel_power: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to mel power spectrogram.

        Args:
            mel_power: ``[B, n_mels, T]`` mel power (NOT dB).

        Returns:
            ``[B, n_mels, T]`` normalized.
        """
        # EMA smooth along time axis
        s = self.s
        smooth = mel_power[:, :, :1]  # seed with first frame
        frames = [smooth]
        for t in range(1, mel_power.shape[-1]):
            smooth = (1 - s) * smooth + s * mel_power[:, :, t:t + 1]
            frames.append(smooth)
        M_smooth = torch.cat(frames, dim=-1)

        # PCEN: (M / (eps + M_smooth)^alpha + delta)^r - delta^r
        denom = (self.eps + M_smooth) ** self.alpha
        gain = mel_power / torch.clamp(denom, min=self.eps)
        return (gain + self.delta) ** self.r - self.delta ** self.r


class DroneDetector(L.LightningModule):
    """Binary drone-audio detector with mel-spectrogram frontend.

    Featurization: mel spectrogram → dB → scalar normalization → 3-channel expand → backbone.
    Supports MixUp/CutMix on spectrograms, SpecAugment, focal loss, dropout,
    cosine LR schedule, and per-bin evaluation.

    Args:
        model: Backbone model configuration.
        mel: Mel spectrogram parameters.
        optimizer: Optimizer and schedule configuration.
        bin_names: Ordered list of SNR bin names for per-bin evaluation.
        loss_type: Loss function ("bce" or "focal").
        label_smoothing: Label smoothing factor (0.0 = off).
        per_bin_weights: Weight loss by SNR bin difficulty.
        spec_augment_prob: Apply SpecAugment (freq + time masking) during training.
        mixup_alpha: Beta distribution alpha for MixUp (0.0 = off).
        cutmix_alpha: Beta distribution alpha for CutMix (0.0 = off).
        dropout: Dropout probability before classifier head.
        bn_momentum: BatchNorm momentum (higher = less drift).
        clip_seconds: Training clip length in seconds (for eval to match).
        freeze_backbone_epochs: Keep backbone weights and batchnorm statistics
            frozen for this many initial epochs.
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
        freeze_backbone_epochs: int = 0,
        distill_teacher: torch.nn.Module | None = None,
        distill_alpha: float = 0.0,
        distill_temperature: float = 2.0,
    ) -> None:
        super().__init__()

        model_cfg = model or ModelConfig()
        mel_cfg = mel or MelConfig()
        opt_cfg = optimizer or OptimizerConfig()
        self._model_cfg = model_cfg
        self._mel_cfg = mel_cfg
        self._optimizer_cfg = opt_cfg

        if not 0.0 <= distill_alpha <= 1.0:
            raise ValueError(
                f"distill_alpha must be in [0, 1], got {distill_alpha}"
            )
        if distill_temperature <= 0:
            raise ValueError(
                "distill_temperature must be > 0, "
                f"got {distill_temperature}"
            )
        if (
            distill_teacher is not None
            and hasattr(distill_teacher, "_to_mel")
            and (mixup_alpha > 0 or cutmix_alpha > 0)
        ):
            raise ValueError(
                "Full-detector distillation uses waveform teacher logits and "
                "does not support spectrogram MixUp/CutMix. Disable mixup/cutmix "
                "or pass a teacher backbone that consumes student spectrograms."
            )

        self.save_hyperparameters(ignore=["distill_teacher"])

        # ── Frontend ───────────────────────────────────────────────
        self._frontend_type = mel_cfg.frontend_type
        self._use_pcen = mel_cfg.use_pcen
        self._frontend_channels = mel_cfg.n_mels  # default for mel-only

        if mel_cfg.frontend_type != "mel":
            from audi.frontend import build_frontend
            fe, fe_channels, pcen_mod = build_frontend(
                mel_cfg.frontend_type,
                sample_rate=mel_cfg.sample_rate,
                hop_length=mel_cfg.hop_length,
                n_mels=mel_cfg.n_mels,
                n_fft=mel_cfg.n_fft,
                win_length=mel_cfg.win_length,
                use_pcen=mel_cfg.use_pcen,
                stft_bands_hz=mel_cfg.stft_bands_hz,
            )
            self._multi_frontend = fe
            self._frontend_channels = fe_channels
            frontend_tokens = {
                part.strip() for part in mel_cfg.frontend_type.split(",")
            }
            stft_like_tokens = {"stft", "stft_bands"}
            self._input_bn = (
                torch.nn.Identity()
                if frontend_tokens and frontend_tokens <= stft_like_tokens
                else torch.nn.BatchNorm2d(3)
            )
            self._pcen = None
            self._to_db = None
        else:
            self._multi_frontend = None
            self._input_bn = None
            self._mel_transform = T.MelSpectrogram(
                sample_rate=mel_cfg.sample_rate,
                n_fft=mel_cfg.n_fft,
                win_length=mel_cfg.win_length,
                hop_length=mel_cfg.hop_length,
                n_mels=mel_cfg.n_mels,
            )
            self._use_pcen = mel_cfg.use_pcen
            if mel_cfg.use_pcen:
                self._pcen = PCEN(
                    s=mel_cfg.pcen_s,
                    alpha=mel_cfg.pcen_alpha,
                    delta=mel_cfg.pcen_delta,
                    r=mel_cfg.pcen_r,
                    eps=mel_cfg.pcen_eps,
                )
                self._to_db = None
            else:
                self._pcen = None
                self._to_db = T.AmplitudeToDB()

        # Scalar mel normalization
        if mel_cfg.mean_db is not None:
            self.register_buffer(
                "_mel_mean", torch.tensor(float(mel_cfg.mean_db))
            )
        else:
            self._mel_mean = None
        if mel_cfg.std_db is not None:
            self.register_buffer(
                "_mel_std", torch.tensor(float(mel_cfg.std_db))
            )
        else:
            self._mel_std = None

        # ── Backbone ─────────────────────────────────────────────
        self.backbone = build_model(
            arch=model_cfg.arch,
            num_classes=model_cfg.num_classes,
            pretrained=model_cfg.pretrained,
            detector_head_hidden_dims=model_cfg.detector_head_hidden_dims,
            detector_head_dropout=model_cfg.detector_head_dropout,
        )
        if model_cfg.compile:
            self.backbone = torch.compile(self.backbone)  # type: ignore[assignment]  # torch.compile erases type

        # BatchNorm momentum
        if bn_momentum != 0.1:
            for m in self.backbone.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.momentum = bn_momentum

        # ── Regularization ───────────────────────────────────────
        self._dropout_p = dropout
        if dropout > 0:
            self._add_classifier_dropout(dropout)

        # ── Training config ──────────────────────────────────────
        self._lr = opt_cfg.lr
        self._weight_decay = opt_cfg.weight_decay
        self._schedule = opt_cfg.schedule
        self._warmup_epochs = opt_cfg.warmup_epochs
        self._max_epochs = opt_cfg.max_epochs

        self._bin_names = list(bin_names) if bin_names else []
        self._loss_type = loss_type
        self._label_smoothing = float(label_smoothing)
        self._clip_seconds = float(clip_seconds)
        self._freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        # Keep the teacher out of Lightning checkpoints while still using it
        # for no-grad forward passes.
        object.__setattr__(self, "_distill_teacher", distill_teacher)
        self._distill_alpha = float(distill_alpha)
        self._distill_temperature = float(distill_temperature)
        self._freeze_distill_teacher()
        self._per_bin_weights = bool(per_bin_weights)
        self._mixup_alpha = float(mixup_alpha)
        self._cutmix_alpha = float(cutmix_alpha)

        # SpecAugment
        self.spec_augment_prob = float(spec_augment_prob)
        if self.spec_augment_prob > 0:
            self._freq_mask = T.FrequencyMasking(15)
            self._time_mask = T.TimeMasking(25)
        else:
            self._freq_mask = None
            self._time_mask = None

        # Per-bin loss weights
        self._bin_weight_map: dict[str, float] = {}
        if self._per_bin_weights and self._bin_names:
            n = len(self._bin_names)
            self._bin_weight_map = {
                name: 0.5 + 2.5 * i / max(n - 1, 1)
                for i, name in enumerate(self._bin_names)
            }

        # ── Validation buffers ──────────────────────────────────
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._val_bin_idx: list[torch.Tensor] = []

    def _set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze feature extractor params while leaving classifier trainable."""
        model = getattr(self.backbone, "_orig_mod", self.backbone)
        for p in model.parameters():
            p.requires_grad = not frozen
        if frozen:
            trainable_head_found = False
            for attr in ("classifier", "fc", "head", "heads"):
                head = getattr(model, attr, None)
                if isinstance(head, torch.nn.Module):
                    head.train()
                    for p in head.parameters():
                        p.requires_grad = True
                    trainable_head_found = True
            if not trainable_head_found:
                last_linear: torch.nn.Linear | None = None
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        last_linear = module
                if last_linear is not None:
                    last_linear.train()
                    for p in last_linear.parameters():
                        p.requires_grad = True

            feature_extractor = getattr(model, "backbone", model)
            feature_extractor.eval()
        else:
            model.train(self.training)

    def _freeze_distill_teacher(self) -> None:
        teacher = self._distill_teacher
        if teacher is None:
            return
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    def on_train_epoch_start(self) -> None:
        """Apply scheduled backbone freezing at epoch boundaries."""
        self._freeze_distill_teacher()
        if self._freeze_backbone_epochs <= 0:
            return
        self._set_backbone_frozen(self.current_epoch < self._freeze_backbone_epochs)

    # ── Featurization ────────────────────────────────────────────

    def _to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized 3-channel spectrogram."""
        if self._multi_frontend is not None:
            feats = self._multi_frontend(wav)   # (B, C, T)
            if feats.ndim == 4:
                spec = feats  # (B, 3, F, T)
            else:
                spec = feats.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, C, T)
            spec = self._input_bn(spec)
            return torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        mel = self._mel_transform(wav)
        if self._use_pcen:
            mel = self._pcen(mel)
        else:
            mel = self._to_db(mel)
            if self._mel_mean is not None and self._mel_std is not None:
                mel = (mel - self._mel_mean) / self._mel_std
        spec = mel.unsqueeze(1).expand(-1, 3, -1, -1)
        if (
            self._freq_mask is not None
            and self._time_mask is not None
            and self.training
        ):
            if random.random() < self.spec_augment_prob:
                for _ in range(2):
                    spec = self._freq_mask(spec)
                for _ in range(2):
                    spec = self._time_mask(spec)
        return spec

    # ── MixUp / CutMix ───────────────────────────────────────────

    def _apply_mixup_cutmix(
        self,
        spec: torch.Tensor,
        labels: torch.Tensor,
        bin_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply MixUp or CutMix on spectrograms during training."""
        if self._mixup_alpha <= 0 and self._cutmix_alpha <= 0:
            return spec, labels, bin_idx

        use_cutmix = self._cutmix_alpha > 0 and (
            self._mixup_alpha <= 0 or random.random() < 0.5
        )
        B = spec.size(0)
        idx = torch.randperm(B, device=spec.device)

        if use_cutmix:
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
            lam = float(np.random.beta(self._mixup_alpha, self._mixup_alpha))
            lam = max(lam, 1.0 - lam)
            spec_mix = lam * spec + (1.0 - lam) * spec[idx]
            labels_mix = lam * labels + (1.0 - lam) * labels[idx]

        return spec_mix, labels_mix, bin_idx

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Run detection on a waveform or precomputed spectrogram."""
        spec = wav if wav.ndim == 4 else self._to_mel(wav)
        return self.backbone(spec).squeeze(1)

    # ── Loss ─────────────────────────────────────────────────────

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        bin_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the training loss."""
        if self._label_smoothing > 0:
            targets = (
                targets * (1 - self._label_smoothing)
                + 0.5 * self._label_smoothing
            )

        if self._loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            pt = torch.exp(-bce)
            alpha = 0.25 * targets + 0.75 * (1 - targets)
            loss = (alpha * (1 - pt) ** 2.0 * bce).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        if (
            self._per_bin_weights
            and bin_idx is not None
            and self._bin_weight_map
        ):
            weights = torch.ones_like(targets)
            for i, name in enumerate(self._bin_names):
                mask = bin_idx == i
                if mask.any() and name in self._bin_weight_map:
                    weights[mask] = self._bin_weight_map[name]
            bce_raw = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = (bce_raw * weights).mean()

        return loss

    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        spec: torch.Tensor,
        wav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._distill_teacher is None:
            raise RuntimeError("distillation loss requested without a teacher")

        temperature = self._distill_temperature
        teacher = self._distill_teacher
        teacher.eval()
        try:
            teacher_param = next(teacher.parameters())
        except StopIteration:
            teacher_param = None
        if teacher_param is not None and teacher_param.device != spec.device:
            teacher.to(spec.device)
        with torch.no_grad():
            if wav is not None and hasattr(teacher, "_to_mel"):
                teacher_logits = teacher(wav.to(spec.device))
            else:
                teacher_logits = teacher(spec)
            teacher_logits = teacher_logits.reshape_as(student_logits)
            teacher_targets = torch.sigmoid(teacher_logits / temperature)

        return (
            F.binary_cross_entropy_with_logits(
                student_logits / temperature,
                teacher_targets,
            )
            * temperature
            * temperature
        )

    # ── Training step ────────────────────────────────────────────

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with optional MixUp/CutMix."""
        if (
            self._freeze_backbone_epochs > 0
            and self.current_epoch < self._freeze_backbone_epochs
        ):
            self._set_backbone_frozen(True)
        if len(batch) == 3:
            wav, label, bin_idx = batch
        else:
            wav, label = batch
            bin_idx = None

        spec = wav if wav.ndim == 4 else self._to_mel(wav)
        spec, label, bin_idx = self._apply_mixup_cutmix(spec, label, bin_idx)
        logits = self.backbone(spec).squeeze(1)
        hard_loss = self._compute_loss(logits, label, bin_idx)
        loss = hard_loss
        log_values = {"train_loss": loss}
        if self._distill_teacher is not None and self._distill_alpha > 0:
            distill_loss = self._compute_distillation_loss(logits, spec, wav)
            loss = (
                (1.0 - self._distill_alpha) * hard_loss
                + self._distill_alpha * distill_loss
            )
            log_values = {
                "train_loss": loss,
                "train/hard_loss": hard_loss,
                "train/distill_loss": distill_loss,
            }
        self.log_dict(log_values, prog_bar=True)
        return loss

    # ── Validation step ──────────────────────────────────────────

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Collect validation predictions for epoch-end metrics."""
        if len(batch) == 6:
            wav, label, bin_idx, *_ = batch
            self._val_bin_idx.append(bin_idx.cpu())
        elif len(batch) == 3:
            wav, label, bin_idx = batch
            self._val_bin_idx.append(bin_idx.cpu())
        else:
            wav, label = batch

        logit = self(wav)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        acc = ((logit > 0.0).float() == label).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        self._val_logits.append(logit.detach().cpu())
        self._val_labels.append(label.detach().cpu())

    # ── Epoch lifecycle ──────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        """Clear validation buffers."""
        self._val_logits.clear()
        self._val_labels.clear()
        self._val_bin_idx.clear()

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits).numpy()
        labels = torch.cat(self._val_labels).numpy()

        # Per-bin data
        per_bin: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self._val_bin_idx:
            bin_idx = torch.cat(self._val_bin_idx).numpy()
            bin_names_str = [
                self._bin_names[int(i)] if i >= 0 else "" for i in bin_idx
            ]
            per_bin = split_by_bin(logits, labels, bin_names_str)

        # ── AUC ──────────────────────────────────────────────
        fpr, tpr, thresholds, auc = compute_roc_values(logits, labels)
        self.log("val/auc", auc, prog_bar=False)

        # ── TPR at FPR targets ───────────────────────────────
        for target in [0.10, 0.05, 0.02]:
            val = tpr_at_fpr(fpr, tpr, target)
            self.log(
                f"val/tpr_at_fpr_{int(target * 100):02d}", val, prog_bar=False
            )

        # ── TPR at Precision targets ─────────────────────────
        precision = compute_precision(logits, labels, thresholds)
        for pt in [0.99, 0.95, 0.90, 0.80]:
            if pt < precision.min() or pt > precision.max():
                continue
            th_pt, tp_at_pt, _ = find_threshold_at_precision(
                precision, tpr, thresholds, pt
            )
            self.log(
                f"val/threshold_at_precision_{int(pt * 100):02d}",
                th_pt,
                prog_bar=False,
            )
            self.log(
                f"val/tpr_at_precision_{int(pt * 100):02d}",
                tp_at_pt,
                prog_bar=False,
            )

        # ── Per-bin TPR ──────────────────────────────────────
        for bn, (bl, bll) in per_bin.items():
            bfpr, btpr, _bth, _bauc = compute_roc_values(bl, bll)
            val = tpr_at_fpr(bfpr, btpr, 0.10)
            self.log(f"val/bin_{bn}/tpr_at_fpr_10", val, prog_bar=False)

        # ── Average Precision ────────────────────────────────
        _, _, _, ap = compute_pr_curve(logits, labels)
        self.log("val/average_precision", ap, prog_bar=False)

        # ── Calibration ──────────────────────────────────────
        _, _, _, ece = compute_calibration(logits, labels)
        self.log("val/ece", ece, prog_bar=True)

    # ── Optimizer ────────────────────────────────────────────────

    def configure_optimizers(self) -> Any:
        """Configure AdamW with optional cosine schedule + warmup."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=self._lr, weight_decay=self._weight_decay
        )
        if self._schedule == "constant":
            return opt

        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            LRScheduler,
            SequentialLR,
        )

        scheduler: LRScheduler
        if self._warmup_epochs > 0:
            warmup = LinearLR(
                opt,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self._warmup_epochs,
            )
            if self._schedule == "linear":
                # Linear warmup → constant LR (no decay)
                from torch.optim.lr_scheduler import ConstantLR

                main = ConstantLR(opt, factor=1.0, total_iters=self._max_epochs)
            else:
                main = CosineAnnealingLR(opt, T_max=self._max_epochs)
            scheduler = SequentialLR(
                opt, schedulers=[warmup, main], milestones=[self._warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(opt, T_max=self._max_epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ── Dropout helper ───────────────────────────────────────────

    def _add_classifier_dropout(self, p: float) -> None:
        """Find the last Linear layer and prepend a Dropout."""
        last_linear: torch.nn.Linear | None = None
        parent: torch.nn.Module | None = None
        attr_name: str | None = None
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, torch.nn.Linear):
                last_linear = mod
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child = parts
                    parent = self.backbone
                    for pn in parent_name.split("."):
                        parent = getattr(parent, pn)
                    attr_name = child
        if (
            last_linear is not None
            and parent is not None
            and attr_name is not None
        ):
            wrapped = torch.nn.Sequential(torch.nn.Dropout(p), last_linear)
            setattr(parent, attr_name, wrapped)
