"""Tests for audi.model module."""

import os

import pytest
import torch

from audi.model import SUPPORTED_MODEL_ARCHS, build_model
from audi.model.efficientat import DYMN_AS_MODELS, MN_AS_MODELS


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

    @pytest.mark.parametrize(
        "arch",
        [
            "cnn14",
            "resnet18",
            "convnext_small",
            "mn04_as",
            "mn10_as",
            "mn40_as",
            "dymn04_as",
            "dymn10_as",
            "dymn20_as",
        ],
    )
    def test_representative_supported_archs_build_without_downloads(self, arch):
        model = build_model(arch=arch, num_classes=1, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 128, 100)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_full_efficientat_model_map_is_intentional(self):
        assert {"mn04_as", "mn10_as", "mn40_as"}.issubset(MN_AS_MODELS)
        assert {"dymn04_as", "dymn10_as", "dymn20_as"}.issubset(DYMN_AS_MODELS)
        assert set(MN_AS_MODELS + DYMN_AS_MODELS).issubset(SUPPORTED_MODEL_ARCHS)

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.environ.get("AUDI_RUN_SLOW_MODEL_TESTS") != "1",
        reason="set AUDI_RUN_SLOW_MODEL_TESTS=1 to build every supported arch",
    )
    def test_all_supported_archs_build(self):
        for arch in SUPPORTED_MODEL_ARCHS:
            model = build_model(arch=arch, num_classes=1, pretrained=False)
            model.eval()
            x = torch.randn(1, 3, 128, 100)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1), f"Failed for {arch}"
