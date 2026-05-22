#!/usr/bin/env python3
"""Training CLI: drone detection (the main pipeline)."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from audi.config import (
    AugmentationConfig,
    MelConfig,
    MixConfig,
    ModelConfig,
    OptimizerConfig,
    parse_snr_bins,
)
from audi.training.dataset import make_dataset
from audi.training.detector import DroneDetector


def run(argv: list[str] | None = None) -> int:
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", message="audio amplitude out of range")

    ap = argparse.ArgumentParser(description="Train a drone detection model.")
    # ── Data ──
    ap.add_argument("--noise-path", type=Path, required=True)
    ap.add_argument("--drone-path", type=Path, required=True)
    ap.add_argument(
        "--noise2", type=Path, default=None, help="Secondary noise dataset"
    )
    ap.add_argument(
        "--snr-bin",
        action="append",
        dest="snr_bin",
        default=[
            "easy:-5:0:0.25",
            "medium:-10:-5:0.30",
            "hard:-15:-10:0.30",
            "extreme:-20:-25:0.15",
        ],
    )
    ap.add_argument("--clip-seconds", type=float, default=2.56)
    ap.add_argument("--highpass-hz", type=float, default=125.0)
    ap.add_argument("--positive-probability", type=float, default=0.5)
    # ── Model ──
    ap.add_argument("--arch", type=str, default="cnn14")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bn-momentum", type=float, default=0.1)
    ap.add_argument(
        "--mel-preset",
        type=str,
        default="default",
        choices=["default", "vit_224", "custom"],
        help="Mel spectrogram preset. vit_224: n_mels=224, hop_length=179. "
        "custom: use --n-mels/--n-fft/--hop-length below.",
    )
    ap.add_argument("--n-mels", type=int, default=None,
                    help="Override n_mels (requires --mel-preset custom)")
    ap.add_argument("--n-fft", type=int, default=None,
                    help="Override n_fft (requires --mel-preset custom)")
    ap.add_argument("--hop-length", type=int, default=None,
                    help="Override hop_length (requires --mel-preset custom)")
    ap.add_argument("--use-pcen", action="store_true",
                    help="Use PCEN instead of dB conversion + scalar normalization")
    ap.add_argument("--pcen-s", type=float, default=0.025)
    ap.add_argument("--pcen-alpha", type=float, default=0.98)
    ap.add_argument("--pcen-delta", type=float, default=2.0)
    ap.add_argument("--pcen-r", type=float, default=0.5)
    ap.add_argument("--use-dsp-features", action="store_true",
                    help="Use DSP feature channels in mel input (DSPDroneDetector)")
    ap.add_argument("--dsp-feature-sets", type=str, default="v3,v4,v5",
                    help="Comma-separated DSP feature sets: v3,v4,v5")
    ap.add_argument("--dsp-hop-length", type=int, default=256,
                    help="Hop length for DSP mel transform")
    ap.add_argument("--dsp-projector-hidden", type=int, default=64,
                    help="Hidden dim for DSP→mel projector MLP")
    # ── Optimizer ──
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--lr-schedule", choices=["constant", "cosine", "linear"], default="constant"
    )
    ap.add_argument("--warmup-epochs", type=int, default=0)
    # ── Training ──
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--val-steps-per-epoch", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--accumulate-grad-batches", type=int, default=1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--save-top-k", type=int, default=1)
    # ── Regularization ──
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--per-bin-weights", action="store_true")
    ap.add_argument("--spec-augment-prob", type=float, default=0.0)
    ap.add_argument("--mixup-alpha", type=float, default=0.0)
    ap.add_argument("--cutmix-alpha", type=float, default=0.0)
    # ── Audio Augmentations ──
    ap.add_argument(
        "--augment", action="store_true", help="Enable audio augmentations"
    )
    ap.add_argument("--doppler-prob", type=float, default=0.2,
                    help="Probability of Doppler shift on drone (0-1)")
    # ── Finetuning ──
    ap.add_argument("--finetune-from", type=Path, default=None)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=None)
    args = ap.parse_args(argv)

    L.seed_everything(args.seed)

    snr_bins = parse_snr_bins(args.snr_bin)
    if args.mel_preset == "vit_224" or args.arch.startswith("fastervit"):
        clip_samples = 36704  # 2.294 s → 224 frames at hop_length=160
    else:
        clip_samples = int(MelConfig().sample_rate * args.clip_seconds)

    model_cfg = ModelConfig(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        compile=not args.no_compile,
    )
    opt_cfg = OptimizerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    aug_cfg = AugmentationConfig(
        enable=True, doppler_prob=args.doppler_prob
    ) if args.augment else None

    mix_cfg = MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        noise2_path=args.noise2,
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        positive_probability=args.positive_probability,
        highpass_hz=args.highpass_hz,
        aug=aug_cfg,
    )

    # ── Train dataset ──
    train_ds = make_dataset(cfg=mix_cfg, split="train", return_bin=True)
    train_ds.length = args.batch_size * args.steps_per_epoch

    # ── Validation dataset ──
    val_mix_cfg = MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        positive_probability=0.5,
        highpass_hz=args.highpass_hz,
        aug=None,  # NEVER augment validation
    )
    val_ds = make_dataset(
        cfg=val_mix_cfg, split="validation", return_components=True
    )
    val_ds.length = args.batch_size * args.val_steps_per_epoch

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, **dl_kwargs)

    # ── Model ──
    bin_names = [b.name for b in snr_bins]
    if args.mel_preset == "vit_224" or args.arch.startswith("fastervit"):
        mel_cfg = MelConfig.vit_224()
    elif args.mel_preset == "custom":
        kwargs = {}
        if args.n_mels is not None:
            kwargs["n_mels"] = args.n_mels
        if args.n_fft is not None:
            kwargs["n_fft"] = args.n_fft
        if args.hop_length is not None:
            kwargs["hop_length"] = args.hop_length
        mel_cfg = MelConfig(**kwargs)
    else:
        mel_cfg = MelConfig()
    # PCEN override (works with any preset)
    if args.use_pcen:
        mel_cfg = replace(mel_cfg,
                          use_pcen=True,
                          pcen_s=args.pcen_s,
                          pcen_alpha=args.pcen_alpha,
                          pcen_delta=args.pcen_delta,
                          pcen_r=args.pcen_r)
    detector_cls = DroneDetector
    detector_kwargs: dict = {}
    if args.use_dsp_features:
        from audi.training.dsp_detector import DSPDroneDetector
        detector_cls = DSPDroneDetector
        detector_kwargs.update(
            dsp_feature_sets=[s.strip() for s in args.dsp_feature_sets.split(",") if s.strip()],
            dsp_hop_length=args.dsp_hop_length,
            dsp_projector_hidden=args.dsp_projector_hidden,
        )
    detector = detector_cls(
        model=model_cfg,
        mel=mel_cfg,
        optimizer=opt_cfg,
        **detector_kwargs,
        bin_names=bin_names,
        loss_type=args.loss,
        label_smoothing=args.label_smoothing,
        per_bin_weights=args.per_bin_weights,
        spec_augment_prob=args.spec_augment_prob,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        dropout=args.dropout,
        bn_momentum=args.bn_momentum,
        clip_seconds=args.clip_seconds,
    )

    if args.pretrained_checkpoint is not None:
        from audi.model.panns import load_panns_pretrained

        load_panns_pretrained(
            detector.backbone, str(args.pretrained_checkpoint)
        )

    if args.finetune_from is not None:
        ckpt = torch.load(
            str(args.finetune_from), map_location="cpu", weights_only=False
        )
        detector.load_state_dict(ckpt["state_dict"], strict=False)

    # ── Trainer ──
    callbacks = [
        ModelCheckpoint(
            monitor="val/auc", mode="max", save_top_k=args.save_top_k
        ),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val/auc", patience=args.patience, mode="max")
        )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(args.output_dir),
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=str(args.output_dir), name="", version=""
        ),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(detector, train_dl, val_dl)
    return 0
