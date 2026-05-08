"""Tests for EfficientNet backbone."""

import torch

from audi.model.vision import EFFICIENTNET_MODEL_ARCHS, _build_efficientnet


class TestEfficientNet:
    def test_build_basic(self):
        """Minimal forward pass works."""
        model = _build_efficientnet("efficientnet_b0", num_classes=1, pretrained=False)
        x = torch.randn(1, 3, 128, 100)
        out = model(x)
        assert out.shape == (1, 1)

    def test_all_arches(self):
        """All efficientnet variants can be built (no pretrained)."""
        for arch in EFFICIENTNET_MODEL_ARCHS:
            model = _build_efficientnet(arch, num_classes=1, pretrained=False)
            x = torch.randn(1, 3, 128, 100)
            out = model(x)
            assert out.shape == (1, 1), f"Failed for {arch}"

    def test_pretrained_builds(self):
        """Building with pretrained=True downloads weights but doesn't crash."""
        model = _build_efficientnet("efficientnet_b0", num_classes=1, pretrained=True)
        assert isinstance(model, torch.nn.Module)
