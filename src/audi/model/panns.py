"""PANNs-style CNN backbones: cnn8, cnn10, cnn14."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

PANN_BLOCK_WIDTHS: Final[dict[str, tuple[int, ...]]] = {
    "cnn8": (64, 128, 256, 512),
    "cnn10": (64, 128, 256, 512, 1024),
    "cnn14": (64, 128, 256, 512, 1024, 2048),
}


class _ConvBlock2d(nn.Module):
    """Two conv2d + BN + ReLU layers followed by average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(
        self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class PANNsCNN(nn.Module):
    """PANNs-style CNN stack adapted for multi-channel input.

    Args:
        num_classes: Number of output logits.
        in_channels: Input channels (default 3 for expanded mel spectrogram).
        block_widths: Channel widths for each conv block.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 3,
        block_widths: tuple[int, ...],
    ) -> None:
        super().__init__()
        widths = tuple(block_widths)
        if len(widths) < 3:
            raise ValueError("PANNsCNN needs at least three conv blocks")
        emb = widths[-1]
        self.bn0 = nn.BatchNorm2d(in_channels)
        c_in = in_channels
        for i, c_out in enumerate(widths):
            self.add_module(f"block{i + 1}", _ConvBlock2d(c_in, c_out))
            c_in = c_out
        self.fc1 = nn.Linear(emb, emb, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(emb, num_classes, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn0(x)
        i = 1
        while hasattr(self, f"block{i}"):
            x = getattr(self, f"block{i}")(x)
            i += 1
        x = x.mean(dim=(2, 3))
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        return self.classifier(x)


def _build_panns(arch: str, *, num_classes: int) -> PANNsCNN:
    """Build a PANNs CNN by architecture name."""
    key = arch.lower()
    if key not in PANN_BLOCK_WIDTHS:
        raise KeyError(arch)
    return PANNsCNN(num_classes=num_classes, block_widths=PANN_BLOCK_WIDTHS[key])


def load_panns_pretrained(model: PANNsCNN, checkpoint_path: str) -> PANNsCNN:
    """Load AudioSet-pretrained weights into a PANNsCNN.

    The official checkpoint from zenodo.org/records/3987831 stores weights
    under ``model`` with keys like ``conv_block1.conv1.weight``.  This
    function remaps them to the naming used by this module
    (``block1.conv1.weight``, etc.) and discards the 527-way AudioSet
    classifier (``fc_audioset``) and the original spectrogram / logmel
    extractors.
    """
    import logging

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    audio_keys: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("spectrogram_extractor") or k.startswith("logmel_extractor"):
            continue
        if k.startswith("fc_audioset"):
            continue
        if k.startswith("conv_block"):
            audio_keys[k.replace("conv_block", "block")] = v
        else:
            audio_keys[k] = v

    missing, unexpected = model.load_state_dict(audio_keys, strict=False)
    n_transferred = len(audio_keys)

    if hasattr(model, "bn0"):
        nn.init.ones_(model.bn0.weight)
        nn.init.zeros_(model.bn0.bias)
        model.bn0.reset_running_stats()

    _log = logging.getLogger(__name__)
    _log.info("Loaded AudioSet weights: %d params transferred", n_transferred)
    if missing:
        _log.info("  Missing (our own layers — OK): %s", missing)
    _log.info(
        "  Skipped (original extractor / 527-way classifier): %d keys",
        len(unexpected),
    )
    return model
