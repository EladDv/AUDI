"""Tests for audi.model module."""

import os

import pytest
import torch

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.model import SUPPORTED_MODEL_ARCHS, build_model
from audi.model.efficientat import DYMN_AS_MODELS, MN_AS_MODELS
from audi.training.detector import DroneDetector


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

    @pytest.mark.parametrize(
        "arch",
        [
            "mn04_as",
            "mn05_as",
            "mn06",
            "mn08",
            "mn10_as",
            "dymn04_as",
            "dymn05",
            "dymn06",
            "dymn08_rt",
        ],
    )
    def test_under_4m_mn_students_build_without_downloads(self, arch):
        model = build_model(arch=arch, num_classes=1, pretrained=False)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        assert params < 4_000_000
        x = torch.randn(1, 3, 128, 100)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_full_efficientat_model_map_is_intentional(self):
        expected_mn = {"mn01_as", "mn02_as", "mn04_as", "mn10_as", "mn40_as"}
        assert expected_mn.issubset(MN_AS_MODELS)
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


class ConstantBackbone(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(logit)))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(spec.shape[0], 1)


class ConstantTeacher(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.logit = float(logit)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (spec.shape[0], 1),
            self.logit,
            device=spec.device,
            dtype=spec.dtype,
        )


def test_detector_distillation_uses_teacher_logits_with_temperature():
    teacher = ConstantTeacher(logit=2.0)
    detector = DroneDetector(
        model=ModelConfig(arch="mn01_as", pretrained=False, compile=False),
        mel=MelConfig(),
        optimizer=OptimizerConfig(),
        distill_teacher=teacher,
        distill_alpha=1.0,
        distill_temperature=2.0,
    )
    detector.backbone = ConstantBackbone(logit=0.0)
    detector._to_mel = lambda wav: wav[:, None, None, :]

    wav = torch.randn(3, 8)
    labels = torch.tensor([0.0, 1.0, 1.0])
    loss = detector.training_step((wav, labels), 0)

    teacher_prob = torch.sigmoid(torch.full_like(labels, 2.0) / 2.0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros_like(labels), teacher_prob
    ) * 4.0
    assert torch.isclose(loss, expected)
    assert not any(p.requires_grad for p in teacher.parameters())


def test_detector_distillation_teacher_is_not_saved_in_student_state_dict():
    detector = DroneDetector(
        model=ModelConfig(arch="mn01_as", pretrained=False, compile=False),
        mel=MelConfig(),
        optimizer=OptimizerConfig(),
        distill_teacher=ConstantTeacher(logit=1.0),
        distill_alpha=0.5,
    )

    assert not any(key.startswith("_distill_teacher.") for key in detector.state_dict())
