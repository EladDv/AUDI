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
from audi.training.dataset import (
    EpochSliceDataset,
    HFDetectionDataset,
    PrecomputedDetectionDataset,
    PrecomputedFeatureDataset,
    make_dataset,
    validate_precomputed_feature_manifest,
    validate_precomputed_manifest,
)
from audi.training.detector import DroneDetector


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    dims: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        dim = int(raw)
        if dim <= 0:
            raise argparse.ArgumentTypeError("hidden dimensions must be positive")
        dims.append(dim)
    return tuple(dims)


def _load_matching_finetune_state(
    detector: DroneDetector,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    current = detector.state_dict()
    current_by_normalized_key = {
        normalized_key: key
        for key in current
        for normalized_key in (strip_compile_prefix({key: current[key]}).keys())
    }
    compatible = {}
    skipped = []
    for key, value in strip_compile_prefix(state_dict).items():
        current_key = current_by_normalized_key.get(key)
        target = current.get(current_key) if current_key is not None else None
        if target is None or target.shape != value.shape:
            skipped.append(key)
            continue
        compatible[current_key] = value
    detector.load_state_dict(compatible, strict=False)
    return skipped


class EpochSliceCallback(L.Callback):
    """Advance EpochSliceDataset to a fresh slice at each train epoch."""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        loader = trainer.train_dataloader
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)


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
    ap.add_argument("--arch", type=str, default="mn10_as")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument(
        "--detector-head-hidden-dims",
        type=_parse_hidden_dims,
        default=(),
        help="Comma-separated hidden dims for the EfficientAT detector head",
    )
    ap.add_argument(
        "--detector-head-dropout",
        type=float,
        default=0.0,
        help="Dropout applied between deep detector head layers",
    )
    ap.add_argument("--bn-momentum", type=float, default=0.1)
    ap.add_argument(
        "--mel-preset",
        type=str,
        default="default",
        choices=["default", "custom"],
        help="Mel spectrogram preset. custom uses --n-mels/--n-fft/--hop-length below. "
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
            "Research mode frontend: mel, stft, stft_bands, or comma-separated "
            "like stft_bands,stft_bands,stft_bands"
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
    ap.add_argument("--precomputed-train-path", type=Path, default=None)
    ap.add_argument("--precomputed-val-path", type=Path, default=None)
    ap.add_argument("--precomputed-feature-train-path", type=Path, default=None)
    ap.add_argument("--precomputed-feature-val-path", type=Path, default=None)
    ap.add_argument(
        "--hf-dataset-path",
        type=Path,
        default=None,
        help="Already-mixed HF detector DatasetDict with train/validation splits.",
    )
    ap.add_argument(
        "--one-pass-samples-per-epoch",
        type=int,
        default=None,
        help="Use a disjoint contiguous train slice of this many samples each epoch.",
    )
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
    ap.add_argument(
        "--distill-from",
        type=Path,
        default=None,
        help="Teacher detector checkpoint used for logit distillation",
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
    clip_samples = int(args.sample_rate * args.clip_seconds)

    model_cfg = ModelConfig(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        compile=not args.no_compile,
        detector_head_hidden_dims=args.detector_head_hidden_dims,
        detector_head_dropout=args.detector_head_dropout,
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

    if args.mel_preset == "custom":
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
    if args.use_pcen:
        mel_cfg = replace(
            mel_cfg,
            use_pcen=True,
            pcen_s=args.pcen_s,
            pcen_alpha=args.pcen_alpha,
            pcen_delta=args.pcen_delta,
            pcen_r=args.pcen_r,
        )
    if args.frontend_type != "mel":
        mel_cfg = replace(
            mel_cfg,
            frontend_type=args.frontend_type,
            stft_bands_hz=stft_bands_hz,
        )

    # ── Train dataset ──
    precomputed_train = (
        args.precomputed_feature_train_path is not None
        or args.precomputed_train_path is not None
    )
    if args.hf_dataset_path is not None:
        train_ds = HFDetectionDataset(
            args.hf_dataset_path,
            split="train",
            target_length_samples=clip_samples,
            sample_rate=args.sample_rate,
            return_bin=True,
        )
    elif args.precomputed_feature_train_path is not None:
        validate_precomputed_feature_manifest(
            args.precomputed_feature_train_path, mix_cfg, mel_cfg, split="train"
        )
        train_ds = PrecomputedFeatureDataset(
            args.precomputed_feature_train_path, return_bin=True
        )
    elif args.precomputed_train_path is not None:
        validate_precomputed_manifest(
            args.precomputed_train_path, mix_cfg, split="train"
        )
        train_ds = PrecomputedDetectionDataset(
            args.precomputed_train_path, return_bin=True
        )
    else:
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
    if args.hf_dataset_path is not None:
        val_ds = HFDetectionDataset(
            args.hf_dataset_path,
            split="validation",
            target_length_samples=clip_samples,
            sample_rate=args.sample_rate,
            return_bin=True,
        )
    elif args.precomputed_feature_val_path is not None:
        validate_precomputed_feature_manifest(
            args.precomputed_feature_val_path, val_mix_cfg, mel_cfg, split="validation"
        )
        val_ds = PrecomputedFeatureDataset(
            args.precomputed_feature_val_path, return_bin=True
        )
    elif args.precomputed_val_path is not None:
        validate_precomputed_manifest(
            args.precomputed_val_path, val_mix_cfg, split="validation"
        )
        val_ds = PrecomputedDetectionDataset(
            args.precomputed_val_path, return_components=True
        )
    else:
        val_ds = make_dataset(
            cfg=val_mix_cfg, split="validation", return_components=True
        )
        val_ds.length = args.batch_size * args.val_steps_per_epoch

    if args.one_pass_samples_per_epoch is not None:
        required = args.one_pass_samples_per_epoch * args.epochs
        if len(train_ds) < required:
            raise SystemExit(
                f"--one-pass-samples-per-epoch requires at least {required} train samples "
                f"for {args.epochs} epochs, but dataset has {len(train_ds)}"
            )
        train_ds = EpochSliceDataset(
            train_ds, samples_per_epoch=args.one_pass_samples_per_epoch
        )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # Precomputed datasets are stored as large shard tensors and cache one shard
    # at a time. Sample-level shuffling makes workers repeatedly deserialize
    # different full shards, which is much slower than sequential shard reads.
    train_dl = DataLoader(train_ds, shuffle=not precomputed_train, **dl_kwargs)
    val_dl = DataLoader(val_ds, **dl_kwargs)

    # ── Model ──
    bin_names = [b.name for b in snr_bins]
    detector_cls = DroneDetector
    detector_kwargs: dict = {}
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

    if args.finetune_from is not None:
        ckpt = torch.load(
            str(args.finetune_from), map_location="cpu", weights_only=False
        )
        skipped = _load_matching_finetune_state(detector, ckpt["state_dict"])
        if skipped:
            preview = ", ".join(skipped[:6])
            suffix = "..." if len(skipped) > 6 else ""
            print(
                f"Finetune load skipped {len(skipped)} incompatible tensors: "
                f"{preview}{suffix}"
            )

    # ── Trainer ──
    callbacks = [
        EpochSliceCallback(),
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
