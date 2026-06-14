"""Tests for audi.model module."""

import os

import pytest
import torch
from lightning import LightningModule

from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.frontend import build_frontend, parse_frequency_bands_hz
from audi.model import SUPPORTED_MODEL_ARCHS, build_model
from audi.model.efficientat import DYMN_AS_MODELS, MN_AS_MODELS, STATIC_DYMN_MODELS
from audi.training.detector import DroneDetector


class TinyWrappedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 4),
        )
        self.classifier = torch.nn.Linear(4, 1)


class TestBuildModel:
    def test_build_unknown_arch(self):
        with pytest.raises(ValueError, match="Unknown arch"):
            build_model(arch="nonexistent")

    @pytest.mark.parametrize(
        "arch",
        [
            "mn04_as",
            "mn10_as",
            "mn40_as",
            "dymn04_as",
            "dymn10_as",
            "dymn20_as",
            "sdymn10",
            "sdymn10_ca",
        ],
    )
    def test_representative_supported_archs_build_without_downloads(self, arch):
        model = build_model(arch=arch, num_classes=1, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 128, 100)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_efficientat_detector_head_can_be_deep(self):
        model = build_model(
            arch="mn10_as",
            num_classes=1,
            pretrained=False,
            detector_head_hidden_dims=(512, 256, 128),
            detector_head_dropout=0.2,
        )

        assert isinstance(model.classifier, torch.nn.Sequential)
        linear_shapes = [
            (m.in_features, m.out_features)
            for m in model.classifier
            if isinstance(m, torch.nn.Linear)
        ]
        assert linear_shapes == [(960, 512), (512, 256), (256, 128), (128, 1)]
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
        expected_static_dymn = {
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
        }
        assert expected_mn.issubset(MN_AS_MODELS)
        assert {"dymn04_as", "dymn10_as", "dymn20_as"}.issubset(DYMN_AS_MODELS)
        assert expected_static_dymn == set(STATIC_DYMN_MODELS)
        assert set(MN_AS_MODELS + DYMN_AS_MODELS + STATIC_DYMN_MODELS).issubset(
            SUPPORTED_MODEL_ARCHS
        )

    @pytest.mark.parametrize(
        "arch",
        ["sdymn04", "sdymn10", "sdymn20", "sdymn40", "sdymn10_ca", "sdymn40_ca"],
    )
    def test_static_dymn_variants_build_without_downloads(self, arch):
        model = build_model(arch=arch, num_classes=1, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 128, 100)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_static_dymn_removes_dynamic_modules(self):
        from models.dymn.dy_block import CoordAtt, DynamicConv, DyReLUB

        model = build_model(arch="sdymn10", num_classes=1, pretrained=False)
        module_types = {type(module) for module in model.modules()}

        assert DynamicConv not in module_types
        assert DyReLUB not in module_types
        assert CoordAtt not in module_types

    def test_static_dymn_ca_keeps_coord_attention_only(self):
        from models.dymn.dy_block import CoordAtt, DynamicConv, DyReLUB

        model = build_model(arch="sdymn10_ca", num_classes=1, pretrained=False)
        module_types = {type(module) for module in model.modules()}

        assert DynamicConv not in module_types
        assert DyReLUB not in module_types
        assert CoordAtt in module_types

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


class ConstantWaveformTeacher(torch.nn.Module):
    """Teacher shaped like DroneDetector: consumes waveform, not spectrogram."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.logit = float(logit)
        self._to_mel = torch.nn.Identity()
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(wav.shape)
        if wav.ndim != 2:
            raise AssertionError(f"Expected waveform [B, T], got {tuple(wav.shape)}")
        return torch.full(
            (wav.shape[0],),
            self.logit,
            device=wav.device,
            dtype=wav.dtype,
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


def test_detector_distillation_can_use_full_waveform_teacher():
    teacher = ConstantWaveformTeacher(logit=2.0)
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
    assert teacher.last_input_shape == (3, 8)


def test_full_teacher_distillation_rejects_spectrogram_mixup_cutmix():
    with pytest.raises(ValueError, match="MixUp/CutMix"):
        DroneDetector(
            model=ModelConfig(arch="mn01_as", pretrained=False, compile=False),
            mel=MelConfig(),
            optimizer=OptimizerConfig(),
            distill_teacher=ConstantWaveformTeacher(logit=1.0),
            distill_alpha=0.5,
            mixup_alpha=0.2,
        )


def test_detector_distillation_teacher_is_not_saved_in_student_state_dict():
    detector = DroneDetector(
        model=ModelConfig(arch="mn01_as", pretrained=False, compile=False),
        mel=MelConfig(),
        optimizer=OptimizerConfig(),
        distill_teacher=ConstantTeacher(logit=1.0),
        distill_alpha=0.5,
    )

    assert not any(key.startswith("_distill_teacher.") for key in detector.state_dict())


def test_stft_frontend_stack_is_three_channel_linear_spectrogram():
    frontend, channels, pcen = build_frontend(
        "stft,stft,stft",
        sample_rate=4000,
        n_mels=128,
        n_fft=512,
        win_length=256,
        hop_length=40,
    )
    wav = torch.randn(2, 20480)

    spec = frontend(wav)

    assert channels == 3
    assert pcen is None
    assert frontend.frontends[0].win_length == 256
    assert spec.shape == (2, 3, 128, 513)
    assert torch.isfinite(spec).all()


def test_stft_frontend_handles_short_and_large_waveforms():
    frontend, _channels, _pcen = build_frontend(
        "stft,stft,stft",
        sample_rate=4000,
        n_mels=128,
        n_fft=512,
        hop_length=40,
    )

    short = torch.ones(2, 64)
    large = torch.full((2, 20480), 1e30)

    assert torch.isfinite(frontend(short)).all()
    assert torch.isfinite(frontend(large)).all()


@pytest.mark.parametrize(
    ("sample_rate", "n_fft", "hop_length", "bands_hz"),
    [
        (4000, 4096, 40, None),
        (
            16000,
            4096,
            160,
            ((100.0, 500.0), (800.0, 1000.0), (1200.0, 1600.0)),
        ),
        (
            16000,
            2048,
            160,
            ((100.0, 500.0), (800.0, 1000.0), (1200.0, 2000.0)),
        ),
    ],
)
def test_banded_stft_frontend_resizes_selected_frequency_bands_to_128(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    bands_hz: tuple[tuple[float, float], ...] | None,
):
    frontend, channels, pcen = build_frontend(
        "stft_bands,stft_bands,stft_bands",
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        stft_bands_hz=bands_hz,
    )
    wav = torch.randn(2, int(sample_rate * 5.12))

    spec = frontend(wav)

    assert channels == 3
    assert pcen is None
    if bands_hz is not None:
        assert frontend.frontends[0].frequency_bands_hz == bands_hz
    assert spec.shape == (2, 3, 128, 513)
    assert torch.isfinite(spec).all()


def test_parse_frequency_bands_hz_accepts_comma_separated_ranges():
    assert parse_frequency_bands_hz("100-500,800-1000,1200-2000") == (
        (100.0, 500.0),
        (800.0, 1000.0),
        (1200.0, 2000.0),
    )


def test_detector_uses_identity_input_norm_for_stft_frontend():
    for frontend_type in ("stft,stft,stft", "stft_bands,stft_bands,stft_bands"):
        detector = DroneDetector(
            model=ModelConfig(arch="mn01_as", pretrained=False, compile=False),
            mel=MelConfig(
                sample_rate=4000,
                n_mels=128,
                n_fft=512,
                win_length=256,
                hop_length=40,
                frontend_type=frontend_type,
            ),
            optimizer=OptimizerConfig(),
        )

        assert isinstance(detector._input_bn, torch.nn.Identity)
        assert detector._multi_frontend.frontends[0].win_length == 256


def test_freeze_backbone_keeps_classifier_trainable():
    detector = DroneDetector.__new__(DroneDetector)
    LightningModule.__init__(detector)
    detector.backbone = TinyWrappedBackbone()

    detector._set_backbone_frozen(True)

    assert all(not p.requires_grad for p in detector.backbone.backbone.parameters())
    assert all(p.requires_grad for p in detector.backbone.classifier.parameters())
    assert not detector.backbone.backbone.training
    assert detector.backbone.classifier.training
