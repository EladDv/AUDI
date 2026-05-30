#!/usr/bin/env python3
"""Multi-task blue/red drone classifier with noise-mixing.

Trains a model to simultaneously (1) detect drones in background noise
and (2) classify detected drones as blue (5-inch) or red (10-inch).

Mixes drone audio into background noise at controlled SNRs, matching
the detection training pipeline. Classification loss only applies
to positive samples (drone present).

Model: AudioSet-pretrained EfficientAT MN backbone
  ├── Detection head: Linear(1) → binary drone/no-drone
  └── Classification head: Linear(2) → blue/red
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import load_from_disk
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from audi.checkpoint import load_model_from_checkpoint
from audi.cli_utils import NUM_WORKERS

SR = 16000
CLIP_S = 2.56
CLIP_SAMPLES = int(SR * CLIP_S)  # 40960


# ── Utility ──────────────────────────────────────────────────────────

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary ROC AUC with average ranks for tied scores."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Return red-positive rates at a probability threshold."""
    pred = scores >= threshold
    pos = labels == 1
    neg = ~pos
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    tpr = tp / max(tp + fn, 1)
    fnr = fn / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": float(threshold),
        "tpr": float(tpr),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "precision": float(precision),
    }


def _rates_at_max_fnr(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fnr: float,
) -> dict[str, float]:
    """Pick the strictest threshold whose red FNR stays below target."""
    thresholds = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        rates = _binary_rates(scores, labels, float(threshold))
        if rates["fnr"] <= max_fnr:
            if best is None or rates["threshold"] > best["threshold"]:
                best = rates
    if best is None:
        return _binary_rates(scores, labels, 0.0)
    return best


def _fit_length(audio: np.ndarray, target: int, *, random_crop: bool = True) -> np.ndarray:
    """Fit audio to exactly `target` samples by looping or trimming."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= target:
        start = random.randint(0, len(audio) - target) if random_crop else 0
        return audio[start:start + target]
    reps = int(np.ceil(target / len(audio)))
    return np.tile(audio, reps)[:target].astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────

class BlueRedMixedDataset(Dataset):
    """On-the-fly dataset: mixes blue/red drone audio into background at random SNR.

    Positive samples: drone (blue or red) + background at SNR in [snr_min, snr_max].
    Negative samples: background only.

    Returns:
        (mix, det_label, cls_label)
        - det_label: 1.0 (drone) or 0.0 (no drone)
        - cls_label: 0 (blue), 1 (red), or -1 (negative — ignore for cls loss)
    """

    def __init__(
        self,
        bg_path: Path,
        drone_path: Path,
        split: str,
        snr_min: float = -20.0,
        snr_max: float = 10.0,
        positive_probability: float = 0.5,
        clip_seconds: float = CLIP_S,
        noise2_prob: float = 0.0,
        noise2_count: int = 3,
        noise2_max_attenuation: float = -38.0,
        length: int | None = None,
    ):
        bg_dd = load_from_disk(str(bg_path))
        drone_dd = load_from_disk(str(drone_path))
        self.bg_ds = bg_dd[split]
        self.drone_ds = drone_dd[split]
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.positive_probability = positive_probability
        self.clip_samples = int(round(SR * clip_seconds))
        self.noise2_prob = float(noise2_prob)
        self.noise2_count = max(0, int(noise2_count))
        self.noise2_max_attenuation = float(noise2_max_attenuation)
        self.length = length or max(len(self.bg_ds), len(self.drone_ds))

        # Pre-index blue/red samples for faster sampling
        self._blue_indices = [i for i, r in enumerate(self.drone_ds) if r["label_id"] == 0]
        self._red_indices = [i for i, r in enumerate(self.drone_ds) if r["label_id"] == 1]

    def __len__(self) -> int:
        return self.length

    def _background(self) -> np.ndarray:
        bg_idx = random.randint(0, len(self.bg_ds) - 1)
        bg = np.asarray(self.bg_ds[bg_idx]["audio"]["array"], dtype=np.float32)
        bg = _fit_length(bg, self.clip_samples)
        bg_rms = _rms(bg)
        if bg_rms > 1e-8:
            bg = bg / bg_rms

        if self.noise2_prob <= 0 or self.noise2_count <= 0:
            return bg
        if random.random() >= self.noise2_prob:
            return bg

        for _ in range(random.randint(1, self.noise2_count)):
            extra_idx = random.randint(0, len(self.bg_ds) - 1)
            extra = np.asarray(
                self.bg_ds[extra_idx]["audio"]["array"], dtype=np.float32
            )
            extra = _fit_length(extra, self.clip_samples)
            extra_rms = _rms(extra)
            if extra_rms > 1e-8:
                extra = extra / extra_rms
            att_db = random.uniform(self.noise2_max_attenuation, 0.0)
            bg = bg + extra * (10.0 ** (att_db / 20.0))

        bg_rms = _rms(bg)
        if bg_rms > 1e-8:
            bg = bg / bg_rms
        return bg

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bg = self._background()

        is_drone = random.random() < self.positive_probability

        if not is_drone:
            # Negative: background only
            mix = torch.as_tensor(bg, dtype=torch.float32)
            return mix, torch.tensor(0.0), torch.tensor(-1, dtype=torch.long)

        # ── Positive: drone + background at random SNR ──
        # Pick class (balanced sampling)
        pick_blue = random.random() < 0.5 and len(self._blue_indices) > 0
        cls_label = 0 if pick_blue else 1
        pool = self._blue_indices if pick_blue else self._red_indices
        if not pool:
            # Fallback if one class is empty
            pool = self._blue_indices + self._red_indices
            cls_label = self.drone_ds[pool[0]]["label_id"]

        drone_idx = random.choice(pool)
        drone = np.asarray(self.drone_ds[drone_idx]["audio"]["array"], dtype=np.float32)
        drone = _fit_length(drone, self.clip_samples)
        drone_rms = _rms(drone)
        if drone_rms > 1e-8:
            drone = drone / drone_rms

        # Mix at random SNR
        target_snr = random.uniform(self.snr_min, self.snr_max)
        scale = 10.0 ** (target_snr / 20.0)
        mix = bg + drone * scale

        # Normalize mix to unit RMS (remove energy cue)
        mix_rms = _rms(mix)
        if mix_rms > 1e-8:
            mix = mix / mix_rms

        return (
            torch.as_tensor(mix, dtype=torch.float32),
            torch.tensor(1.0),                   # det_label
            torch.tensor(cls_label, dtype=torch.long),  # cls_label
        )


# ── Model ────────────────────────────────────────────────────────────

class BlueRedDetector(L.LightningModule):
    """Multi-task model: drone detection + blue/red classification.

    Backbone: AudioSet-pretrained EfficientAT MN.
    Head: two branches — detection (1-class) and classification (2-class).
    """

    def __init__(
        self,
        model_arch: str = "mn10_as",
        lr: float = 1e-4,
        dropout: float = 0.2,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        cls_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
        mixup_alpha: float = 0.0,
        freeze_backbone_epochs: int = 2,
        detector_checkpoint: Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.cls_loss_weight = cls_loss_weight
        self.det_loss_weight = det_loss_weight
        self.mixup_alpha = mixup_alpha
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._uses_detector_checkpoint = detector_checkpoint is not None

        # ── Frontend ──
        if detector_checkpoint is None:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            )
            self.to_db = T.AmplitudeToDB()
        else:
            self.mel_transform = None
            self.to_db = None

        # ── Backbone + heads ──
        self._build_model(
            model_arch,
            dropout=dropout,
            detector_checkpoint=detector_checkpoint,
        )
        self._val_red_scores: list[torch.Tensor] = []
        self._val_red_labels: list[torch.Tensor] = []
        if self._uses_detector_checkpoint and self.freeze_backbone_epochs > 0:
            self._set_backbone_frozen(True)

    def _build_model(
        self,
        model_arch: str,
        dropout: float,
        detector_checkpoint: Path | None,
    ):
        if detector_checkpoint is not None:
            detector = load_model_from_checkpoint(detector_checkpoint, quiet=True)
            detector.eval()
            self.detector = detector
            source_arch = self.detector.hparams["model"].arch
            if source_arch != model_arch:
                print(
                    f"Warning: --model-arch={model_arch} but checkpoint is {source_arch}; "
                    "using checkpoint backbone."
                )
            feature_dim = self.detector.backbone.classifier[1].in_features
            self.backbone = self.detector.backbone
            self.pool = None
            self.flatten = None
        else:
            self.detector = None
            feature_dim = self._build_audioset_model(model_arch)

        # Shared embedding
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 640),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

        # Detection head: binary (drone vs no-drone). In detector-checkpoint
        # mode we use the detector's already-trained binary head.
        self.det_head = None if detector_checkpoint is not None else nn.Linear(640, 1)

        # Classification head: 2-class (blue vs red)
        self.cls_head = nn.Linear(640, 2)

    def _build_audioset_model(self, model_arch: str) -> int:
        from models.mn.model import get_model

        width_map = {"mn04_as": 0.4, "mn05_as": 0.5, "mn10_as": 1.0}
        width = width_map.get(model_arch, 1.0)

        with contextlib.redirect_stdout(io.StringIO()):
            full_model = get_model(
                num_classes=527,
                pretrained_name=model_arch,
                width_mult=width,
                head_type="mlp",
                input_dim_f=128,
                input_dim_t=1000,
            )
        feature_dim = full_model.classifier[2].in_features  # 480 for mn05, 960 for mn10

        self.backbone = full_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        return feature_dim

    def _set_backbone_frozen(self, frozen: bool) -> None:
        module = self.detector if self._uses_detector_checkpoint else self.backbone
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = not frozen
        if frozen:
            module.eval()
        else:
            module.train(self.training)

    def _to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        if self.detector is not None:
            return self.detector._to_mel(wav)
        mel = self.to_db(self.mel_transform(wav))
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
        mel = (mel - mean) / std
        return mel.unsqueeze(1)  # (B, 1, 128, T)

    def _extract_features(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.detector is not None:
            if self.freeze_backbone_epochs > 0 and self.current_epoch < self.freeze_backbone_epochs:
                self.detector.eval()
            detector_logits = self.detector.backbone(spec).squeeze(1)
            _, features = self.detector.backbone.backbone(spec[:, :1])
            return features, detector_logits

        feats = self.backbone(spec)
        pooled = self.pool(feats)
        flat = self.flatten(pooled)
        return flat, None

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (det_logits[B, 1], cls_logits[B, 2])."""
        spec = self._to_spec(wav)
        features, detector_logits = self._extract_features(spec)
        emb = self.shared_fc(features)
        cls_logits = self.cls_head(emb)
        if detector_logits is not None:
            return detector_logits.unsqueeze(-1), cls_logits
        return self.det_head(emb), cls_logits

    def training_step(self, batch, batch_idx):
        wav, det_label, cls_label = batch

        det_logits, cls_logits = self(wav)

        # Detection loss is optional for frozen detector-head training.
        if self.det_loss_weight > 0:
            det_loss = F.binary_cross_entropy_with_logits(
                det_logits.squeeze(-1), det_label
            )
        else:
            det_loss = torch.tensor(0.0, device=wav.device)

        # Classification loss (only on positives)
        pos_mask = det_label > 0.5
        if pos_mask.any():
            cls_loss = F.cross_entropy(
                cls_logits[pos_mask], cls_label[pos_mask]
            )
        else:
            cls_loss = torch.tensor(0.0, device=wav.device)

        loss = self.det_loss_weight * det_loss + self.cls_loss_weight * cls_loss

        # Metrics
        det_acc = ((det_logits.squeeze(-1) > 0) == det_label.bool()).float().mean()
        self.log_dict({
            "train_loss": loss,
            "train_det_loss": det_loss,
            "train_det_acc": det_acc,
        }, prog_bar=True)

        if pos_mask.any():
            cls_acc = (cls_logits[pos_mask].argmax(-1) == cls_label[pos_mask]).float().mean()
            self.log("train_cls_acc", cls_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wav, det_label, cls_label = batch

        det_logits, cls_logits = self(wav)

        if self.det_loss_weight > 0:
            det_loss = F.binary_cross_entropy_with_logits(
                det_logits.squeeze(-1), det_label
            )
        else:
            det_loss = torch.tensor(0.0, device=wav.device)

        pos_mask = det_label > 0.5
        cls_loss = torch.tensor(0.0, device=wav.device)
        if pos_mask.any():
            cls_loss = F.cross_entropy(cls_logits[pos_mask], cls_label[pos_mask])

        loss = self.det_loss_weight * det_loss + self.cls_loss_weight * cls_loss

        det_acc = ((det_logits.squeeze(-1) > 0) == det_label.bool()).float().mean()
        self.log_dict({
            "val_loss": loss,
            "val_det_loss": det_loss,
            "val_det_acc": det_acc,
        }, prog_bar=True)

        if pos_mask.any():
            cls_acc = (cls_logits[pos_mask].argmax(-1) == cls_label[pos_mask]).float().mean()
            self.log("val_cls_acc", cls_acc, prog_bar=True)
            red_scores = cls_logits[pos_mask].softmax(dim=-1)[:, 1]
            red_labels = (cls_label[pos_mask] == 1).long()
            self._val_red_scores.append(red_scores.detach().cpu())
            self._val_red_labels.append(red_labels.detach().cpu())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_validation_epoch_start(self):
        self._val_red_scores.clear()
        self._val_red_labels.clear()

    def on_validation_epoch_end(self):
        if not self._val_red_scores:
            return

        scores = torch.cat(self._val_red_scores).numpy()
        labels = torch.cat(self._val_red_labels).numpy()
        if labels.size == 0:
            return

        default = _binary_rates(scores, labels, 0.5)
        metrics = {
            "val_red_auc": _binary_auc(scores, labels),
            "val_red_tpr": default["tpr"],
            "val_red_fnr": default["fnr"],
            "val_red_fpr": default["fpr"],
            "val_red_precision": default["precision"],
        }
        for target_fnr in (0.05, 0.10, 0.20):
            rates = _rates_at_max_fnr(scores, labels, target_fnr)
            suffix = f"{int(target_fnr * 100):02d}"
            metrics[f"val_red_tpr_at_fnr_{suffix}"] = rates["tpr"]
            metrics[f"val_red_fpr_at_fnr_{suffix}"] = rates["fpr"]
            metrics[f"val_red_threshold_at_fnr_{suffix}"] = rates["threshold"]

        self.log_dict(metrics, prog_bar=False)
        print(
            "red-val "
            f"auc={metrics['val_red_auc']:.3f} "
            f"tpr={metrics['val_red_tpr']:.3f} "
            f"fnr={metrics['val_red_fnr']:.3f} "
            f"fpr={metrics['val_red_fpr']:.3f} "
            f"tpr@fnr10={metrics['val_red_tpr_at_fnr_10']:.3f} "
            f"fpr@fnr10={metrics['val_red_fpr_at_fnr_10']:.3f} "
            f"thr@fnr10={metrics['val_red_threshold_at_fnr_10']:.3f}"
        )

    def on_train_epoch_start(self):
        if self.freeze_backbone_epochs > 0 and self.current_epoch < self.freeze_backbone_epochs:
            self._set_backbone_frozen(True)
        elif self.freeze_backbone_epochs > 0 and self.current_epoch == self.freeze_backbone_epochs:
            self._set_backbone_frozen(False)
            print(f"\n[Epoch {self.current_epoch}] Unfroze backbone")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    torch.set_float32_matmul_precision("high")

    ap = argparse.ArgumentParser(
        description="Train multi-task blue/red detector+classifier with noise mixing"
    )
    ap.add_argument("--bg-path", type=Path,
                    default=Path("data/HF_dataset_v7_background"))
    ap.add_argument("--drone-path", type=Path,
                    default=Path("data/hf_blue_red"))
    ap.add_argument("--model-arch", default="mn10_as",
                    choices=["mn04_as", "mn05_as", "mn10_as", "dymn10_as"])
    ap.add_argument("--detector-checkpoint", type=Path, default=None,
                    help="Trained detector checkpoint to use as frozen feature source")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip-seconds", type=float, default=CLIP_S)
    ap.add_argument("--snr-min", type=float, default=-20.0)
    ap.add_argument("--snr-max", type=float, default=10.0)
    ap.add_argument("--positive-prob", type=float, default=0.5)
    ap.add_argument("--noise2-prob", type=float, default=0.0,
                    help="Probability of layering extra background clips")
    ap.add_argument("--noise2-count", type=int, default=3,
                    help="Maximum extra background layers")
    ap.add_argument("--noise2-max-attenuation", type=float, default=-38.0,
                    help="Lowest attenuation in dB for extra background layers")
    ap.add_argument("--cls-loss-weight", type=float, default=0.5,
                    help="Weight for classification loss relative to detection loss")
    ap.add_argument("--det-loss-weight", type=float, default=1.0,
                    help="Set to 0 to train only the blue/red head")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2)
    ap.add_argument("--dataset-length", type=int, default=2000,
                    help="Virtual dataset size per epoch")
    ap.add_argument("--val-length", type=int, default=None,
                    help="Virtual validation size; defaults to max(200, dataset_length / 5)")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/blue_red_detect"))
    ap.add_argument("--save-top-k", type=int, default=1)
    ap.add_argument("--monitor", default="val_cls_acc",
                    help="Metric for ModelCheckpoint, e.g. val_cls_acc or val_det_acc")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    L.seed_everything(args.seed)

    # ── Datasets ──
    train_ds = BlueRedMixedDataset(
        bg_path=args.bg_path,
        drone_path=args.drone_path,
        split="train",
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        positive_probability=args.positive_prob,
        clip_seconds=args.clip_seconds,
        noise2_prob=args.noise2_prob,
        noise2_count=args.noise2_count,
        noise2_max_attenuation=args.noise2_max_attenuation,
        length=args.dataset_length,
    )
    val_ds = BlueRedMixedDataset(
        bg_path=args.bg_path,
        drone_path=args.drone_path,
        split="validation",
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        positive_probability=args.positive_prob,
        clip_seconds=args.clip_seconds,
        noise2_prob=args.noise2_prob,
        noise2_count=args.noise2_count,
        noise2_max_attenuation=args.noise2_max_attenuation,
        length=args.val_length or max(200, args.dataset_length // 5),
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    blue_count = len(train_ds._blue_indices)
    red_count = len(train_ds._red_indices)
    print(f"Blue drones: {blue_count}, Red drones: {red_count}")
    print(f"Background: {len(train_ds.bg_ds)} train clips")
    print(f"Clip length: {args.clip_seconds:.2f}s")
    print(f"SNR range: [{args.snr_min}, {args.snr_max}] dB")
    print(f"Positive probability: {args.positive_prob:.2f}")
    print(
        f"Extra noise: p={args.noise2_prob:.2f}, count={args.noise2_count}, "
        f"attenuation=[{args.noise2_max_attenuation}, 0.0] dB"
    )
    print(f"Virtual dataset size: {args.dataset_length}")

    # ── Model ──
    model = BlueRedDetector(
        model_arch=args.model_arch,
        lr=args.lr,
        dropout=args.dropout,
        cls_loss_weight=args.cls_loss_weight,
        det_loss_weight=args.det_loss_weight,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        detector_checkpoint=args.detector_checkpoint,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {args.model_arch} ({n_params:.2f}M params, {n_trainable:.2f}M trainable)")

    # ── Trainer ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(args.output_dir),
        callbacks=[
            ModelCheckpoint(
                monitor=args.monitor, mode="max", save_top_k=args.save_top_k,
                filename=(
                    "{epoch:02d}-cls{val_cls_acc:.3f}-redauc{val_red_auc:.3f}"
                    "-redfnr{val_red_fnr:.3f}"
                ),
            ),
        ],
    )
    trainer.fit(model, train_dl, val_dl)

    print(f"\nCheckpoints saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
