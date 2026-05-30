"""TorchVision-based, timm-based, and custom backbones."""

from __future__ import annotations

from typing import Final

from torch import nn

try:
    from torchvision import models
except ModuleNotFoundError:
    models = None

EFFICIENTNET_MODEL_ARCHS: Final[tuple[str, ...]] = tuple(
    f"efficientnet_b{i}" for i in range(8)
)

# ── Timm-based model families ────────────────────────────────────────────

EFFICIENTNET_LITE_ARCHS: Final[tuple[str, ...]] = tuple(
    f"efficientnet_lite{i}" for i in range(5)
)

MOBILENETV4_CONV_ARCHS: Final[tuple[str, ...]] = (
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large",
)

MOBILENETV4_HYBRID_ARCHS: Final[tuple[str, ...]] = (
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_large",
)

MOBILEVIT_ARCHS: Final[tuple[str, ...]] = (
    "mobilevit_xxs",
    "mobilevit_xs",
    "mobilevit_s",
)

MOBILEVITV2_ARCHS: Final[tuple[str, ...]] = tuple(
    f"mobilevitv2_{size:03d}" for size in (50, 75, 100, 125, 150, 175, 200)
)

_FASTERVIT_ARCHS: Final[tuple[str, ...]] = ("fastervit_0",)

# Map our arch names -> timm model names
_TIMM_NAME_MAP: Final[dict[str, str]] = {
    **{f"efficientnet_lite{i}": f"tf_efficientnet_lite{i}" for i in range(5)},
    **{a: a for a in MOBILENETV4_CONV_ARCHS},
    **{a: a for a in MOBILENETV4_HYBRID_ARCHS},
    **{a: a for a in MOBILEVIT_ARCHS},
    **{a: a for a in MOBILEVITV2_ARCHS},
}


def _efficientnet_weights_name(arch: str) -> str:
    suffix = arch.removeprefix("efficientnet_b")
    return f"EfficientNet_B{suffix}_Weights"


def _build_timm(
    arch: str, *, num_classes: int, pretrained: bool
) -> nn.Module:
    """Build a timm-based backbone (efficientnet_lite, mobilenetv4, mobilevit)."""
    import timm

    timm_name = _TIMM_NAME_MAP.get(arch, arch)
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def _build_fastervit(
    arch: str, *, num_classes: int, pretrained: bool
) -> nn.Module:
    """Build a FasterViT backbone."""
    import fastervit

    name_map = {"fastervit_0": "faster_vit_0_224"}
    model_name = name_map.get(arch, arch)
    model = fastervit.create_model(model_name, pretrained=pretrained)
    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    return model


# ── TorchVision backbones ────────────────────────────────────────────────


def _require_torchvision():
    if models is None:
        raise ModuleNotFoundError(
            "torchvision is required for torchvision backbones"
        )
    return models


def _build_resnet(
    arch: str, *, num_classes: int, pretrained: bool
) -> nn.Module:
    """Build a ResNet backbone with 3-channel mel-spectrogram input."""
    tv_models = _require_torchvision()
    weight = "IMAGENET1K_V1" if pretrained else None
    model_fn = getattr(tv_models, arch)
    model = model_fn(weights=weight)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_convnext(
    arch: str, *, num_classes: int, pretrained: bool
) -> nn.Module:
    """Build a ConvNeXt backbone."""
    tv_models = _require_torchvision()
    weight = "IMAGENET1K_V1" if pretrained else None
    model_fn = getattr(tv_models, arch)
    model = model_fn(weights=weight)
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet(
    arch: str, *, num_classes: int, pretrained: bool
) -> nn.Module:
    """Build an EfficientNet backbone."""
    tv_models = _require_torchvision()
    model_fn = getattr(tv_models, arch)
    if pretrained:
        weights_cls = getattr(tv_models, _efficientnet_weights_name(arch))
        model = model_fn(weights=weights_cls.DEFAULT)
    else:
        model = model_fn(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
