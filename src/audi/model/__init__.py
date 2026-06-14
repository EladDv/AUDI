"""EfficientAT MN/DyMN audio classification backbones."""

from __future__ import annotations

from torch import nn

from audi.model.efficientat import EFFICIENTAT_MODELS, build_efficientat

SUPPORTED_MODEL_ARCHS: tuple[str, ...] = EFFICIENTAT_MODELS


def build_model(
    *,
    arch: str,
    num_classes: int = 1,
    pretrained: bool = True,
    detector_head_hidden_dims: tuple[int, ...] = (),
    detector_head_dropout: float = 0.0,
) -> nn.Module:
    """Factory: construct an audio backbone by architecture name.

    Args:
        arch: EfficientAT architecture identifier (e.g. "mn10_as", "dymn10_as").
        num_classes: Number of output logits.
        pretrained: Whether to load pretrained weights.

    Returns:
        An nn.Module that accepts ``[B, 3, n_mels, T]`` spectrograms.

    Raises:
        ValueError: If ``arch`` is not recognized.
    """
    arch_lower = arch.strip().lower()

    if arch_lower in EFFICIENTAT_MODELS:
        return build_efficientat(
            arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            head_hidden_dims=detector_head_hidden_dims,
            head_dropout=detector_head_dropout,
        )

    supported = ", ".join(SUPPORTED_MODEL_ARCHS)
    raise ValueError(f"Unknown arch {arch!r}. Choose from: {supported}")
