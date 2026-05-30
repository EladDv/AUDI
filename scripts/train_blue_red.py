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
import random
import sys
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

from audi.cli_utils import NUM_WORKERS

SR = 16000
CLIP_S = 2.56
CLIP_SAMPLES = int(SR * CLIP_S)  # 40960


# ── Utility ──────────────────────────────────────────────────────────

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _fit_length(audio: np.ndarray, target: int) -> np.ndarray:
    """Fit audio to exactly `target` samples by looping or trimming."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= target:
        return audio[:target]
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
        length: int | None = None,
    ):
        bg_dd = load_from_disk(str(bg_path))
        drone_dd = load_from_disk(str(drone_path))
        self.bg_ds = bg_dd[split]
        self.drone_ds = drone_dd[split]
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.positive_probability = positive_probability
        self.length = length or max(len(self.bg_ds), len(self.drone_ds))

        # Pre-index blue/red samples for faster sampling
        self._blue_indices = [i for i, r in enumerate(self.drone_ds) if r["label_id"] == 0]
        self._red_indices = [i for i, r in enumerate(self.drone_ds) if r["label_id"] == 1]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = CLIP_SAMPLES

        # ── Load background ──
        bg_idx = random.randint(0, len(self.bg_ds) - 1)
        bg = np.asarray(self.bg_ds[bg_idx]["audio"]["array"], dtype=np.float32)
        bg = _fit_length(bg, L)
        bg_rms = _rms(bg)
        if bg_rms > 1e-8:
            bg = bg / bg_rms

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
        drone = _fit_length(drone, L)
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
        mixup_alpha: float = 0.0,
        freeze_backbone_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.cls_loss_weight = cls_loss_weight
        self.mixup_alpha = mixup_alpha
        self.freeze_backbone_epochs = freeze_backbone_epochs

        # ── Frontend ──
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )
        self.to_db = T.AmplitudeToDB()

        # ── Backbone + heads ──
        self._build_model(model_arch, dropout=dropout)

    def _build_model(self, model_arch: str, dropout: float):
        from models.mn.model import get_model

        width_map = {"mn04_as": 0.4, "mn05_as": 0.5, "mn10_as": 1.0}
        width = width_map.get(model_arch, 1.0)

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

        # Shared embedding
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 640),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

        # Detection head: binary (drone vs no-drone)
        self.det_head = nn.Linear(640, 1)

        # Classification head: 2-class (blue vs red)
        self.cls_head = nn.Linear(640, 2)

    def _to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.to_db(self.mel_transform(wav))
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
        mel = (mel - mean) / std
        return mel.unsqueeze(1)  # (B, 1, 128, T)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (det_logits[B, 1], cls_logits[B, 2])."""
        spec = self._to_spec(wav)
        feats = self.backbone(spec)
        pooled = self.pool(feats)
        flat = self.flatten(pooled)
        emb = self.shared_fc(flat)
        return self.det_head(emb), self.cls_head(emb)

    def training_step(self, batch, batch_idx):
        wav, det_label, cls_label = batch

        det_logits, cls_logits = self(wav)

        # Detection loss (always)
        det_loss = F.binary_cross_entropy_with_logits(
            det_logits.squeeze(-1), det_label
        )

        # Classification loss (only on positives)
        pos_mask = det_label > 0.5
        if pos_mask.any():
            cls_loss = F.cross_entropy(
                cls_logits[pos_mask], cls_label[pos_mask]
            )
        else:
            cls_loss = torch.tensor(0.0, device=wav.device)

        loss = det_loss + self.cls_loss_weight * cls_loss

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

        det_loss = F.binary_cross_entropy_with_logits(
            det_logits.squeeze(-1), det_label
        )

        pos_mask = det_label > 0.5
        cls_loss = torch.tensor(0.0, device=wav.device)
        if pos_mask.any():
            cls_loss = F.cross_entropy(cls_logits[pos_mask], cls_label[pos_mask])

        loss = det_loss + self.cls_loss_weight * cls_loss

        det_acc = ((det_logits.squeeze(-1) > 0) == det_label.bool()).float().mean()
        self.log_dict({
            "val_loss": loss,
            "val_det_loss": det_loss,
            "val_det_acc": det_acc,
        }, prog_bar=True)

        if pos_mask.any():
            cls_acc = (cls_logits[pos_mask].argmax(-1) == cls_label[pos_mask]).float().mean()
            self.log("val_cls_acc", cls_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_train_epoch_start(self):
        if self.freeze_backbone_epochs > 0 and self.current_epoch < self.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif self.freeze_backbone_epochs > 0 and self.current_epoch == self.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            print(f"\n[Epoch {self.current_epoch}] Unfroze backbone")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    torch.set_float32_matmul_precision("high")

    ap = argparse.ArgumentParser(
        description="Train multi-task blue/red detector+classifier with noise mixing"
    )
    ap.add_argument("--bg-path", type=Path,
                    default=Path("data/HF_dataset_v6_background"))
    ap.add_argument("--drone-path", type=Path,
                    default=Path("data/hf_blue_red"))
    ap.add_argument("--model-arch", default="mn10_as",
                    choices=["mn04_as", "mn05_as", "mn10_as"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--snr-min", type=float, default=-20.0)
    ap.add_argument("--snr-max", type=float, default=10.0)
    ap.add_argument("--positive-prob", type=float, default=0.5)
    ap.add_argument("--cls-loss-weight", type=float, default=0.5,
                    help="Weight for classification loss relative to detection loss")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2)
    ap.add_argument("--dataset-length", type=int, default=2000,
                    help="Virtual dataset size per epoch")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/blue_red_detect"))
    ap.add_argument("--save-top-k", type=int, default=1)
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
        length=args.dataset_length,
    )
    val_ds = BlueRedMixedDataset(
        bg_path=args.bg_path,
        drone_path=args.drone_path,
        split="validation",
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        positive_probability=args.positive_prob,
        length=max(200, args.dataset_length // 5),
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
    print(f"SNR range: [{args.snr_min}, {args.snr_max}] dB")
    print(f"Virtual dataset size: {args.dataset_length}")

    # ── Model ──
    model = BlueRedDetector(
        model_arch=args.model_arch,
        lr=args.lr,
        dropout=args.dropout,
        cls_loss_weight=args.cls_loss_weight,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
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
                monitor="val_det_acc", mode="max", save_top_k=args.save_top_k,
                filename="{epoch:02d}-det{val_det_acc:.3f}-cls{val_cls_acc:.3f}",
            ),
        ],
    )
    trainer.fit(model, train_dl, val_dl)

    print(f"\nCheckpoints saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
