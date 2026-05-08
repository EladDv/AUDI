"""Abstract base for audio classification backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from audi.model.vision import (
    _FASTERVIT_ARCHS,
    EFFICIENTNET_LITE_ARCHS,
    MOBILENETV4_CONV_ARCHS,
    MOBILENETV4_HYBRID_ARCHS,
    MOBILEVIT_ARCHS,
    MOBILEVITV2_ARCHS,
    _build_convnext,
    _build_efficientnet,
    _build_fastervit,
    _build_resnet,
    _build_timm,
)


class AudioBackbone(nn.Module, ABC):
    """Abstract base for audio classification backbones.

    All backbones must:
      - Accept ``[B, 3, n_mels, T]`` spectrograms as input.
      - Return ``[B, num_classes]`` logits.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            spec: Spectrogram tensor of shape ``[B, 3, n_mels, T]``.

        Returns:
            Logits tensor of shape ``[B, num_classes]``.
        """
        ...


SUPPORTED_MODEL_ARCHS: tuple[str, ...] = (
    "cnn8",
    "cnn10",
    "cnn14",
    "resnet18",
    "resnet34",
    "resnet50",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
) + tuple(f"efficientnet_b{i}" for i in range(8)) \
  + EFFICIENTNET_LITE_ARCHS \
  + MOBILENETV4_CONV_ARCHS \
  + MOBILENETV4_HYBRID_ARCHS \
  + MOBILEVIT_ARCHS \
  + MOBILEVITV2_ARCHS \
  + _FASTERVIT_ARCHS


def build_model(
    *, arch: str, num_classes: int = 1, pretrained: bool = True
) -> nn.Module:
    """Factory: construct an audio backbone by architecture name.

    Args:
        arch: Architecture identifier (e.g. "cnn14", "resnet18").
        num_classes: Number of output logits.
        pretrained: Whether to load pretrained weights.

    Returns:
        An nn.Module that accepts ``[B, 3, n_mels, T]`` spectrograms.

    Raises:
        ValueError: If ``arch`` is not recognized.
    """
    arch_lower = arch.strip().lower()

    if arch_lower in {"cnn8", "cnn10", "cnn14"}:
        from audi.model.panns import _build_panns

        return _build_panns(arch_lower, num_classes=num_classes)

    if arch_lower in {"resnet18", "resnet34", "resnet50"}:
        return _build_resnet(
            arch_lower, num_classes=num_classes, pretrained=pretrained
        )

    if arch_lower in {"convnext_tiny", "convnext_small", "convnext_base"}:
        return _build_convnext(
            arch_lower, num_classes=num_classes, pretrained=pretrained
        )

    if "efficientnet_b" in arch_lower:
        return _build_efficientnet(
            arch_lower, num_classes=num_classes, pretrained=pretrained
        )

    if arch_lower in EFFICIENTNET_LITE_ARCHS + MOBILENETV4_CONV_ARCHS \
            + MOBILENETV4_HYBRID_ARCHS + MOBILEVIT_ARCHS + MOBILEVITV2_ARCHS:
        return _build_timm(
            arch_lower, num_classes=num_classes, pretrained=pretrained
        )

    if arch_lower in _FASTERVIT_ARCHS:
        return _build_fastervit(
            arch_lower, num_classes=num_classes, pretrained=pretrained
        )

    supported = ", ".join(SUPPORTED_MODEL_ARCHS)
    raise ValueError(f"Unknown arch {arch!r}. Choose from: {supported}")
