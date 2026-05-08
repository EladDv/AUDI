"""Tests for audi.model module."""

import pytest
import torch

from audi.model import SUPPORTED_MODEL_ARCHS, build_model


class TestBuildModel:
    def test_build_cnn14(self):
        model = build_model(arch="cnn14", num_classes=1)
        assert isinstance(model, torch.nn.Module)
        # Forward pass shape check
        x = torch.randn(1, 3, 128, 100)
        out = model(x)
        assert out.shape == (1, 1)

    def test_build_resnet18(self):
        model = build_model(arch="resnet18", num_classes=1, pretrained=True)
        x = torch.randn(1, 3, 128, 100)
        out = model(x)
        assert out.shape == (1, 1)

    def test_build_unknown_arch(self):
        with pytest.raises(ValueError, match="Unknown arch"):
            build_model(arch="nonexistent")

    def test_all_supported_archs_build(self):
        for arch in SUPPORTED_MODEL_ARCHS:
            model = build_model(arch=arch, num_classes=1)
            x = torch.randn(1, 3, 128, 100)
            out = model(x)
            assert out.shape == (1, 1), f"Failed for {arch}"
