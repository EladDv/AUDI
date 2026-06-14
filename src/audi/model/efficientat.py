"""EfficientAT MN/DyMN audio backbone wrappers.

Wraps pre-trained MobileNet (MN) and Dynamic MobileNet (DyMN) models
from the EfficientAT repository for binary drone detection.

These models were pre-trained on AudioSet with 1-channel mel spectrograms
and 527 output classes.  This module adapts them to:
  - Input:  [B, 3, n_mels, T]  (takes channel 0 = raw mel, discards delta/delta-delta)
  - Output: [B, 1]             (binary drone logit)
"""

from __future__ import annotations

import contextlib
import io

import torch
from torch import nn

# ── AudioSet pre-trained model names ──────────────────────────────────

# From the README "Pre-Trained Models" table (AudioSet checkpoints only)
MN_AS_MODELS: tuple[str, ...] = (
    "mn01_as",
    "mn02_as",
    "mn04_as",
    "mn05_as",
    "mn10_as",
    "mn20_as",
    "mn30_as",
    "mn40_as",
    "mn40_as_ext",
    "mn40_as_no_im_pre",
    # Hop size variants (same architecture, different time resolution)
    "mn10_as_hop_15",
    "mn10_as_hop_20",
    "mn10_as_hop_25",
    # Mel band variants (same architecture, different freq resolution)
    "mn10_as_mels_40",
    "mn10_as_mels_64",
    "mn10_as_mels_256",
)

DYMN_AS_MODELS: tuple[str, ...] = (
    "dymn04_as",
    "dymn10_as",
    "dymn20_as",
)

MN_SCRATCH_MODELS: tuple[str, ...] = (
    "mn06",
    "mn08",
)

DYMN_SCRATCH_MODELS: tuple[str, ...] = (
    "dymn05",
    "dymn06",
    "dymn08_rt",
)

STATIC_DYMN_MODELS: tuple[str, ...] = (
    "sdymn04",
    "sdymn05",
    "sdymn10",
    "sdymn20",
    "sdymn30",
    "sdymn40",
    "sdymn04_ca",
    "sdymn05_ca",
    "sdymn10_ca",
    "sdymn20_ca",
    "sdymn30_ca",
    "sdymn40_ca",
)

# EfficientAT MobileNet-family model names usable via --arch
EFFICIENTAT_MODELS: tuple[str, ...] = (
    MN_AS_MODELS
    + DYMN_AS_MODELS
    + MN_SCRATCH_MODELS
    + DYMN_SCRATCH_MODELS
    + STATIC_DYMN_MODELS
)

# Map pretrained name → width_mult
_NAME_TO_WIDTH: dict[str, float] = {
    "mn01": 0.1,
    "mn02": 0.2,
    "mn04": 0.4,
    "mn05": 0.5,
    "mn06": 0.6,
    "mn08": 0.8,
    "mn10": 1.0,
    "mn20": 2.0,
    "mn30": 3.0,
    "mn40": 4.0,
    "dymn04": 0.4,
    "dymn05": 0.5,
    "dymn06": 0.6,
    "dymn08": 0.8,
    "dymn10": 1.0,
    "dymn20": 2.0,
    "sdymn04": 0.4,
    "sdymn05": 0.5,
    "sdymn10": 1.0,
    "sdymn20": 2.0,
    "sdymn30": 3.0,
    "sdymn40": 4.0,
}


def _parse_width(name: str) -> float:
    """Extract width_mult from a pretrained model name."""
    for prefix in ("sdymn", "dymn", "mn"):
        if name.startswith(prefix):
            key = name[: len(prefix) + 2]  # e.g. "mn10", "dymn04"
            return _NAME_TO_WIDTH.get(key, 1.0)
    return 1.0


def _is_mn(name: str) -> bool:
    return name.startswith("mn") and name[:4] in _NAME_TO_WIDTH


def _is_dymn(name: str) -> bool:
    return name.startswith("dymn") and name[:6] in _NAME_TO_WIDTH


def _is_static_dymn(name: str) -> bool:
    base = name.removesuffix("_ca")
    return base.startswith("sdymn") and base[:7] in _NAME_TO_WIDTH


def _quiet_get_model(get_model, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_model(*args, **kwargs)


def _build_classifier_head(
    input_dim: int,
    num_classes: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
) -> nn.Module:
    if not hidden_dims:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)


class _EfficientATWrapper(nn.Module):
    """Generic wrapper that adapts an EfficientAT model for our pipeline.

    Architecture:
        [B, 3, n_mels, T] → slice [:, 0:1] → [B, 1, n_mels, T]
                         → EfficientAT backbone (features only)
                         → new classifier head → [B, 1]

    Only the raw mel channel (index 0) is used; delta and delta-delta
    channels are discarded.  The original 527-class AudioSet classifier
    is discarded; a fresh binary classifier is attached.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 1,
        dropout: float = 0.2,
        head_hidden_dims: tuple[int, ...] = (),
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = _build_classifier_head(
            feature_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout if head_hidden_dims else dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :1]  # [B, 3, H, W] → [B, 1, H, W] — take mel channel only
        # EfficientAT backbones return (logits, features) tuple — we want features
        _, features = self.backbone(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.classifier(features)


def _build_mn(
    pretrained_name: str,
    num_classes: int = 1,
    *,
    pretrained: bool = True,
    head_hidden_dims: tuple[int, ...] = (),
    head_dropout: float = 0.0,
) -> _EfficientATWrapper:
    """Build an AudioSet-pretrained MobileNetV3 wrapper.

    The original first conv (1→16, stride 2) receives our 1-channel
    projected input.  The 527-way classifier is discarded and replaced
    with a binary head.

    Args:
        pretrained_name: e.g. "mn10_as", "mn04_as", "mn20_as".
        num_classes: Number of output logits (default 1 for binary).

    Returns:
        Wrapped model accepting [B, 3, n_mels, T] and returning [B, num_classes].
    """
    from models.mn.model import get_model

    width = _parse_width(pretrained_name)
    # Build with 527 AudioSet classes to match pretrained weights
    model = _quiet_get_model(
        get_model,
        num_classes=527,
        pretrained_name=pretrained_name if pretrained else None,
        width_mult=width,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=1000,
    )
    # The last channel before classifier in MN is
    # lastconv_output_channels = 6 * lastconv_input_channels.
    # where lastconv_input_channels = inverted_residual_setting[-1].out_channels
    # For mn10 (width=1.0): lastconv_output_channels = 6 * 160 = 960
    feature_dim = model.classifier[2].in_features  # Linear: lastconv_output_channels → last_channel
    # The wrapper consumes backbone features only; drop the original AudioSet
    # classifier so small distilled students stay small at checkpoint/export time.
    model.classifier = nn.Identity()
    # Discard the original classifier
    wrapper = _EfficientATWrapper(
        model,
        feature_dim,
        num_classes=num_classes,
        head_hidden_dims=head_hidden_dims,
        head_dropout=head_dropout,
    )
    return wrapper


def _build_dymn(
    pretrained_name: str,
    num_classes: int = 1,
    *,
    pretrained: bool = True,
    head_hidden_dims: tuple[int, ...] = (),
    head_dropout: float = 0.0,
) -> _EfficientATWrapper:
    """Build an AudioSet-pretrained Dynamic MobileNet wrapper."""
    from models.dymn.model import get_model

    width = _parse_width(pretrained_name)
    reduced_tail = pretrained_name.endswith("_rt")
    model = _quiet_get_model(
        get_model,
        num_classes=527,
        pretrained_name=pretrained_name if pretrained else None,
        width_mult=width,
        reduced_tail=reduced_tail,
    )
    # DyMN classifier[2] is the first Linear (lastconv_output_channels → last_channel)
    feature_dim = model.classifier[2].in_features
    # The wrapper consumes backbone features only; drop the original AudioSet
    # classifier so small distilled students stay small at checkpoint/export time.
    model.classifier = nn.Identity()
    wrapper = _EfficientATWrapper(
        model,
        feature_dim,
        num_classes=num_classes,
        head_hidden_dims=head_hidden_dims,
        head_dropout=head_dropout,
    )
    return wrapper


def _build_static_dymn(
    arch_name: str,
    num_classes: int = 1,
    *,
    head_hidden_dims: tuple[int, ...] = (),
    head_dropout: float = 0.0,
) -> _EfficientATWrapper:
    """Build a static DyMN-derived student.

    ``sdymn*`` keeps the DyMN width/channel topology but removes the dynamic
    convolution, dynamic ReLU, and coordinate attention branches. ``sdymn*_ca``
    keeps coordinate attention as the higher-capacity static-derived variant.
    These students are trained from scratch/distillation rather than loaded
    from EfficientAT pretrained checkpoints.
    """
    from models.dymn.model import get_model

    keep_ca = arch_name.endswith("_ca")
    base_name = arch_name.removesuffix("_ca")
    width = _parse_width(base_name)
    model = _quiet_get_model(
        get_model,
        num_classes=527,
        pretrained_name=None,
        width_mult=width,
        reduced_tail=False,
        no_dyconv=True,
        no_dyrelu=True,
        no_ca=not keep_ca,
    )
    feature_dim = model.classifier[2].in_features
    model.classifier = nn.Identity()
    return _EfficientATWrapper(
        model,
        feature_dim,
        num_classes=num_classes,
        head_hidden_dims=head_hidden_dims,
        head_dropout=head_dropout,
    )


def build_efficientat(
    pretrained_name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    cache_dir: str | None = None,
    head_hidden_dims: tuple[int, ...] = (),
    head_dropout: float = 0.0,
) -> _EfficientATWrapper:
    """Factory: build an EfficientAT model wrapper by pretrained name.

    Args:
        pretrained_name: Any key from EFFICIENTAT_MODELS (e.g. "mn10_as").
        num_classes: Number of output logits (default 1 for binary).
        pretrained: Whether to load AudioSet weights. Set False for offline
            shape tests and smoke builds.
        cache_dir: Optional directory for downloaded checkpoints.

    Returns:
        Wrapped model accepting [B, 3, n_mels, T] and returning [B, num_classes].

    Raises:
        ValueError: If pretrained_name is not recognized.
    """
    if cache_dir is not None:
        import models.dymn.model as _dymn
        import models.mn.model as _mn

        _mn.model_dir = cache_dir
        _dymn.model_dir = cache_dir

    if pretrained and pretrained_name in (
        MN_SCRATCH_MODELS + DYMN_SCRATCH_MODELS + STATIC_DYMN_MODELS
    ):
        raise ValueError(
            f"{pretrained_name!r} has no EfficientAT pretrained checkpoint. "
            "Use pretrained=False or pass --no-pretrained."
        )

    if _is_mn(pretrained_name):
        return _build_mn(
            pretrained_name,
            num_classes=num_classes,
            pretrained=pretrained,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
        )
    elif _is_dymn(pretrained_name):
        return _build_dymn(
            pretrained_name,
            num_classes=num_classes,
            pretrained=pretrained,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
        )
    elif _is_static_dymn(pretrained_name):
        return _build_static_dymn(
            pretrained_name,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
        )
    else:
        supported = ", ".join(EFFICIENTAT_MODELS)
        raise ValueError(
            f"Unknown EfficientAT model {pretrained_name!r}. Choose from: {supported}"
        )


# Quick smoke test
if __name__ == "__main__":
    print("Building mn10_as (no download, dry-run architecture check)...")
    from models.mn.model import get_model

    backbone = _quiet_get_model(
        get_model,
        num_classes=527,
        pretrained_name=None,
        width_mult=1.0,
        head_type="mlp",
    )
    feature_dim = backbone.classifier[2].in_features
    wrapper = _EfficientATWrapper(backbone, feature_dim)
    x = torch.randn(2, 3, 128, 100)
    y = wrapper(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}  (expected [2, 1])")
    print(f"  Params: {sum(p.numel() for p in wrapper.parameters()) / 1e6:.2f}M")
    print("OK")
