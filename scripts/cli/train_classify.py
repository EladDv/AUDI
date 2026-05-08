#!/usr/bin/env python3
"""Training CLI: DADS classification pretraining."""

from __future__ import annotations

import argparse
import warnings
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
from audi.config import MelConfig
from audi.model import build_model

_SR = MelConfig().sample_rate


class SimpleClassifyDataset(Dataset):
    """Load clips from HF dataset, return (audio, label)."""

    def __init__(self, ds_path: Path, split: str, target_len: int):
        dd = load_from_disk(str(ds_path))
        self.data = dd[split]
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
        if len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        else:
            audio = audio[: self.target_len]
        label = float(sample["is_drone"])
        return torch.as_tensor(audio, dtype=torch.float32), torch.tensor(
            label
        )


class DroneClassifier(L.LightningModule):
    """Binary drone-vs-non-drone classifier (no background mixing)."""

    def __init__(
        self,
        model_arch="resnet18",
        lr=1e-4,
        pretrained=True,
        n_mels=128,
        n_fft=1024,
        hop_length=160,
        dropout=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.mel = T.MelSpectrogram(
            sample_rate=_SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = T.AmplitudeToDB()

        class Cfg:
            pass

        cfg = Cfg()
        cfg.model_arch = model_arch
        cfg.num_classes = 1
        cfg.pretrained_backbone = pretrained
        self.model = build_model(cfg)
        self._dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def _to_mel(self, wav):
        mel = self.to_db(self.mel(wav))
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
        mel = (mel - mean) / std
        return mel.unsqueeze(1).expand(-1, 3, -1, -1)

    def forward(self, wav):
        return self.model(self._to_mel(wav)).squeeze(1)

    def training_step(self, batch, batch_idx):
        wav, label = batch
        logit = self(wav)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        acc = ((logit > 0).float() == label).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav, label = batch
        logit = self(wav)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        acc = ((logit > 0).float() == label).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return {"logits": logit, "labels": label}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def run(argv: list[str] | None = None) -> int:
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", message="audio amplitude out of range")

    ap = argparse.ArgumentParser(
        description="Train a binary drone classifier (no background mixing)."
    )
    ap.add_argument(
        "--drone-path",
        type=Path,
        required=True,
        help="HF dataset with drone/non-drone clips",
    )
    ap.add_argument("--model-arch", default="resnet18")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip-seconds", type=float, default=2.56)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--save-top-k", type=int, default=1)
    args = ap.parse_args(argv)

    clip_samples = int(_SR * args.clip_seconds)
    train_ds = SimpleClassifyDataset(args.drone_path, "train", clip_samples)
    val_ds = SimpleClassifyDataset(args.drone_path, "validation", clip_samples)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = DroneClassifier(
        model_arch=args.model_arch,
        lr=args.lr,
        pretrained=True,
        dropout=args.dropout,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(args.output_dir),
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=args.save_top_k
            )
        ],
    )
    trainer.fit(model, train_dl, val_dl)
    return 0
