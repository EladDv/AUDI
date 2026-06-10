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

from audi.checkpoint import load_model_from_checkpoint, strip_compile_prefix
from audi.config import (
    AugmentationConfig,
    MelConfig,
    MixConfig,
    ModelConfig,
    OptimizerConfig,
    parse_snr_bins,
)
from audi.frontend import parse_frequency_bands_hz
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
        "--hard-noise",
        type=Path,
        default=None,
        help="Hard-negative background dataset sampled as base noise",
    )
    ap.add_argument(
        "--hard-noise-prob",
        type=float,
        default=0.0,
        help="Probability of sampling base noise from --hard-noise",
    )
    ap.add_argument(
        "--noise2", type=Path, default=None, help="Secondary noise dataset"
    )
    ap.add_argument("--noise2-prob", type=float, default=0.25,
                    help="Probability of mixing noise2 into a sample (0-1)")
    ap.add_argument("--noise2-multi-prob", type=float, default=0.5,
                    help="When noise2 is used, probability of mixing multiple noise2 clips")
    ap.add_argument("--noise2-count", type=int, default=3,
                    help="Max number of extra noise2 layers")
    ap.add_argument("--noise2-max-attenuation", type=float, default=-40.0,
                    help="Minimum dB of noise2 relative to base noise (more negative = quieter)")
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
    ap.add_argument(
        "--sample-rate",
        type=int,
        default=MelConfig().sample_rate,
        help="Target audio sample rate for dataset resampling and frontend construction",
    )
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
        "custom: use --n-mels/--n-fft/--hop-length below. "
        "--win-length can override any preset.",
    )
    ap.add_argument(
        "--n-mels",
        type=int,
        default=None,
        help="Override n_mels (requires --mel-preset custom)",
    )
    ap.add_argument(
        "--n-fft",
        type=int,
        default=None,
        help="Override n_fft (requires --mel-preset custom)",
    )
    ap.add_argument(
        "--win-length",
        type=int,
        default=None,
        help="Override frontend win_length",
    )
    ap.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="Override hop_length (requires --mel-preset custom)",
    )
    ap.add_argument(
        "--use-pcen",
        action="store_true",
        help="Research mode: use PCEN instead of dB conversion + scalar normalization",
    )
    ap.add_argument("--pcen-s", type=float, default=0.025)
    ap.add_argument("--pcen-alpha", type=float, default=0.98)
    ap.add_argument("--pcen-delta", type=float, default=2.0)
    ap.add_argument("--pcen-r", type=float, default=0.5)
    ap.add_argument(
        "--frontend-type",
        default="mel",
        help=(
            "Research mode frontend: mel, stft, stft_bands, cqt, cwt, or "
            "comma-separated like mel,cqt or stft_bands,stft_bands,stft_bands"
        ),
    )
    ap.add_argument(
        "--stft-bands-hz",
        "--stft-buckets-hz",
        dest="stft_bands_hz",
        default=None,
        help=(
            "Comma-separated frequency buckets for stft_bands, e.g. "
            "100-500,800-1000,1200-1600"
        ),
    )
    ap.add_argument("--cqt-bins", type=int, default=84)
    ap.add_argument("--cqt-bpo", type=int, default=12)
    ap.add_argument("--cwt-scales", type=int, default=64)
    ap.add_argument(
        "--use-dsp-features",
        action="store_true",
        help="Research mode: use DSP feature channels in mel input (DSPDroneDetector)",
    )
    ap.add_argument(
        "--use-dsp-branch",
        action="store_true",
        help=(
            "Research mode: use DSP features as separate branch fused with backbone embedding "
            "(MNBranchDSPDetector)"
        ),
    )
    ap.add_argument(
        "--dsp-feature-sets",
        type=str,
        default="v3,v4,v5",
        help="Comma-separated DSP feature sets: v3,v4,v5",
    )
    ap.add_argument(
        "--dsp-hop-length",
        type=int,
        default=256,
        help="Hop length for DSP mel transform",
    )
    ap.add_argument(
        "--dsp-projector-hidden",
        type=int,
        default=64,
        help="Hidden dim for DSP→mel projector MLP (DSPDroneDetector)",
    )
    ap.add_argument(
        "--dsp-emb-dim",
        type=int,
        default=256,
        help="DSP branch encoder output dim (MNBranchDSPDetector)",
    )
    ap.add_argument(
        "--fusion-hidden",
        type=int,
        default=512,
        help="Fusion MLP hidden dim (MNBranchDSPDetector)",
    )
    # ── Optimizer ──
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "linear"],
        default="constant",
    )
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone params and BN stats for the first N epochs",
    )
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
    ap.add_argument(
        "--doppler-prob",
        type=float,
        default=0.2,
        help="Probability of Doppler shift on drone (0-1)",
    )
    ap.add_argument("--pitch-prob", type=float, default=0.25)
    ap.add_argument("--stretch-prob", type=float, default=0.25)
    ap.add_argument("--reverb-prob", type=float, default=0.25)
    ap.add_argument("--eq-prob", type=float, default=0.25)
    ap.add_argument("--noise-inject-prob", type=float, default=0.25)
    ap.add_argument("--noise-inject-db", type=float, default=-40.0)
    ap.add_argument("--time-mask-prob", type=float, default=0.25)
    ap.add_argument("--lowpass-prob", type=float, default=0.25)
    ap.add_argument("--atmospheric-prob", type=float, default=0.25)
    # ── Finetuning ──
    ap.add_argument("--finetune-from", type=Path, default=None)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=None)
    ap.add_argument(
        "--distill-from",
        type=Path,
        default=None,
        help="Teacher detector checkpoint used for logit distillation",
    )
    ap.add_argument(
        "--distill-arch",
        type=str,
        default="dymn10_as",
        help=(
            "Legacy teacher architecture hint. Full checkpoints loaded via "
            "--distill-from carry their own architecture."
        ),
    )
    ap.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help="Blend weight for teacher loss: 0=hard labels, 1=teacher only",
    )
    ap.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Temperature for binary logit distillation",
    )
    args = ap.parse_args(argv)
    try:
        stft_bands_hz = parse_frequency_bands_hz(args.stft_bands_hz)
    except ValueError as exc:
        ap.error(str(exc))

    L.seed_everything(args.seed)

    snr_bins = parse_snr_bins(args.snr_bin)
    if args.mel_preset == "vit_224" or args.arch.startswith("fastervit"):
        clip_samples = 36704  # 2.294 s → 224 frames at hop_length=160
    else:
        clip_samples = int(args.sample_rate * args.clip_seconds)

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
    aug_cfg = None
    if args.augment:
        aug_cfg = AugmentationConfig(
            enable=True,
            doppler_prob=args.doppler_prob,
            pitch_prob=args.pitch_prob,
            stretch_prob=args.stretch_prob,
            reverb_prob=args.reverb_prob,
            eq_prob=args.eq_prob,
            noise_inject_prob=args.noise_inject_prob,
            noise_inject_db=args.noise_inject_db,
            time_mask_prob=args.time_mask_prob,
            lowpass_prob=args.lowpass_prob,
            atmospheric_prob=args.atmospheric_prob,
        )

    mix_cfg = MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        hard_noise_path=args.hard_noise,
        hard_noise_prob=args.hard_noise_prob,
        noise2_path=args.noise2,
        noise2_prob=args.noise2_prob,
        noise2_multi_noise_prob=args.noise2_multi_prob,
        noise2_count=args.noise2_count,
        noise2_max_attenuation_db=args.noise2_max_attenuation,
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        positive_probability=args.positive_probability,
        highpass_hz=args.highpass_hz,
        sample_rate=args.sample_rate,
        aug=aug_cfg,
    )

    # ── Train dataset ──
    train_ds = make_dataset(cfg=mix_cfg, split="train", return_bin=True)
    train_ds.length = args.batch_size * args.steps_per_epoch

    # ── Validation dataset ──
    val_mix_cfg = MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        hard_noise_path=args.hard_noise,
        hard_noise_prob=args.hard_noise_prob,
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        positive_probability=0.5,
        highpass_hz=args.highpass_hz,
        sample_rate=args.sample_rate,
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
        mel_cfg = replace(MelConfig.vit_224(), sample_rate=args.sample_rate)
    elif args.mel_preset == "custom":
        kwargs = {"sample_rate": args.sample_rate}
        if args.n_mels is not None:
            kwargs["n_mels"] = args.n_mels
        if args.n_fft is not None:
            kwargs["n_fft"] = args.n_fft
        if args.win_length is not None:
            kwargs["win_length"] = args.win_length
        if args.hop_length is not None:
            kwargs["hop_length"] = args.hop_length
        mel_cfg = MelConfig(**kwargs)
    else:
        mel_cfg = MelConfig(sample_rate=args.sample_rate)
    if args.win_length is not None and args.mel_preset != "custom":
        mel_cfg = replace(mel_cfg, win_length=args.win_length)
    # PCEN override (works with any preset)
    if args.use_pcen:
        mel_cfg = replace(
            mel_cfg,
            use_pcen=True,
            pcen_s=args.pcen_s,
            pcen_alpha=args.pcen_alpha,
            pcen_delta=args.pcen_delta,
            pcen_r=args.pcen_r,
        )
    # Frontend override
    if args.frontend_type != "mel":
        mel_cfg = replace(
            mel_cfg,
            frontend_type=args.frontend_type,
            stft_bands_hz=stft_bands_hz,
            cqt_bins=args.cqt_bins,
            cqt_bpo=args.cqt_bpo,
            cwt_scales=args.cwt_scales,
        )
    detector_cls = DroneDetector
    detector_kwargs: dict = {}
    if args.use_dsp_features and args.use_dsp_branch:
        raise SystemExit(
            "--use-dsp-features and --use-dsp-branch are mutually exclusive. "
            "Choose one DSP integration strategy."
        )
    if args.use_dsp_features:
        from audi.training.dsp_detector import DSPDroneDetector

        detector_cls = DSPDroneDetector
        detector_kwargs.update(
            dsp_feature_sets=[
                s.strip() for s in args.dsp_feature_sets.split(",") if s.strip()
            ],
            dsp_hop_length=args.dsp_hop_length,
            dsp_projector_hidden=args.dsp_projector_hidden,
        )
    elif args.use_dsp_branch:
        from audi.training.mn_dsp_branch_detector import MNBranchDSPDetector

        detector_cls = MNBranchDSPDetector
        detector_kwargs.update(
            dsp_feature_sets=[
                s.strip() for s in args.dsp_feature_sets.split(",") if s.strip()
            ],
            dsp_hop_length=args.dsp_hop_length,
            dsp_emb_dim=args.dsp_emb_dim,
            fusion_hidden=args.fusion_hidden,
        )
    distill_teacher = None
    if args.distill_from is not None:
        teacher_detector = load_model_from_checkpoint(
            args.distill_from,
            device="cpu",
            quiet=True,
        )
        distill_teacher = teacher_detector

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
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        distill_teacher=distill_teacher,
        distill_alpha=args.distill_alpha if distill_teacher is not None else 0.0,
        distill_temperature=args.distill_temperature,
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
        detector.load_state_dict(
            strip_compile_prefix(ckpt["state_dict"]), strict=False
        )

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
