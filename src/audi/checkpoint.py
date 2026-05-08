"""Checkpoint loading utilities for AUDI models.

Centralises the checkpoint-bridging logic (old flat-param vs new config-object
hparams) and _orig_mod prefix stripping that was duplicated across evaluate.py,
qat_train.py, export_tflite.py, eval_ensemble.py, and eval_app/model_utils.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.training.detector import DroneDetector


def strip_compile_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` prefix from torch.compile-trained state dict keys.

    Without this, ``load_state_dict(strict=False)`` silently drops ALL backbone
    weights for compiled checkpoints, giving AUC ≈ 0.5.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _resolve_model_config(hp: dict) -> ModelConfig:
    """Bridge old (flat dict) and new (config object) hparams for model."""
    model_hp = hp.get("model", {})
    if isinstance(model_hp, dict):
        return ModelConfig(
            arch=model_hp.get("arch", hp.get("model_arch", "cnn14")),
            pretrained=model_hp.get(
                "pretrained", hp.get("pretrained_backbone", True)
            ),
            compile=False,
        )
    # Config object stored in checkpoint
    return ModelConfig(
        arch=model_hp.arch,
        pretrained=model_hp.pretrained,
        compile=False,
    )


def _resolve_mel_config(hp: dict) -> MelConfig:
    """Bridge old (flat dict) and new (config object) hparams for mel."""
    mel_hp = hp.get("mel", {})
    if isinstance(mel_hp, dict):
        return MelConfig(
            n_mels=mel_hp.get("n_mels", hp.get("n_mels", 128)),
            n_fft=mel_hp.get("n_fft", hp.get("n_fft", 1024)),
            hop_length=mel_hp.get("hop_length", hp.get("hop_length", 160)),
            mean_db=mel_hp.get("mean_db", hp.get("mel_mean")),
            std_db=mel_hp.get("std_db", hp.get("mel_std")),
        )
    # Config object stored in checkpoint
    return MelConfig(
        n_mels=getattr(mel_hp, "n_mels", hp.get("n_mels", 128)),
        n_fft=getattr(mel_hp, "n_fft", hp.get("n_fft", 1024)),
        hop_length=getattr(mel_hp, "hop_length", hp.get("hop_length", 160)),
        mean_db=getattr(mel_hp, "mean_db", hp.get("mel_mean")),
        std_db=getattr(mel_hp, "std_db", hp.get("mel_std")),
    )


def _resolve_optimizer_config(hp: dict) -> OptimizerConfig:
    """Bridge old and new hparams for optimizer."""
    opt_hp = hp.get("optimizer", {})
    if isinstance(opt_hp, dict):
        return OptimizerConfig(
            lr=opt_hp.get("lr", hp.get("lr", 1e-3)),
            weight_decay=opt_hp.get("weight_decay", hp.get("weight_decay", 0.01)),
            schedule=opt_hp.get("schedule", hp.get("lr_schedule", "constant")),
            warmup_epochs=opt_hp.get("warmup_epochs", hp.get("warmup_epochs", 0)),
        )
    return opt_hp


def get_clip_seconds(hp: dict) -> float:
    """Extract training clip length from checkpoint hyperparameters.

    Returns ``hp[\"clip_seconds\"]`` for new checkpoints, falls back to
    2.56 for old checkpoints that predate the parameter.
    """
    cs = hp.get("clip_seconds")
    if cs is not None:
        return float(cs)
    return 2.56  # legacy default


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str = "cpu",
    *,
    bin_names: list[str] | None = None,
    quiet: bool = False,
) -> DroneDetector:
    """Load a DroneDetector from a training checkpoint in eval mode.

    Handles both old flat-hparam and new config-object checkpoints, and
    strips _orig_mod. prefixes from compiled models.

    Args:
        ckpt_path: Path to the ``.ckpt`` file.
        device: Target device (``"cpu"`` or ``"cuda"``).
        bin_names: SNR bin names. If None, taken from checkpoint hparams
                   (``hp["bin_names"]``) or defaulted to ``[]``.
        quiet: Suppress bridge-format info prints.

    Returns:
        Loaded model in eval mode on the target device.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    model_cfg = _resolve_model_config(hp)
    mel_cfg = _resolve_mel_config(hp)
    opt_cfg = _resolve_optimizer_config(hp)

    if bin_names is None:
        bin_names = hp.get("bin_names", [])

    if not quiet:
        print(f"Loading {model_cfg.arch} from {ckpt_path.name}")

    model = DroneDetector(
        model=model_cfg,
        mel=mel_cfg,
        optimizer=opt_cfg,
        bin_names=list(bin_names),
        loss_type=hp.get("loss_type", "bce"),
        label_smoothing=hp.get("label_smoothing", 0.0),
        per_bin_weights=hp.get("per_bin_weights", False),
        spec_augment_prob=float(hp.get("spec_augment_prob", 0.0)),
        mixup_alpha=hp.get("mixup_alpha", 0.0),
        cutmix_alpha=hp.get("cutmix_alpha", 0.0),
        dropout=hp.get("dropout", 0.0),
        bn_momentum=hp.get("bn_momentum", 0.1),
        clip_seconds=get_clip_seconds(hp),
    )
    model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
    return model.to(device).eval()
