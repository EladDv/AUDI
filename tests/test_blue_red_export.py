"""Tests for maintained blue/red TFLite export wrappers."""

from __future__ import annotations

import pytest
import torch

from scripts.export_blue_red_tflite import (
    CombinedDetectorBlueRed,
    _dynamic_conv_export_forward,
)


class _FakeBackbone(torch.nn.Module):
    def forward(self, spec):
        features = spec.mean(dim=(2, 3))
        return torch.zeros(spec.size(0), 527), features


class _FakeDetectorBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.classifier = torch.nn.Linear(1, 1)


class _FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeDetectorBackbone()


class _FakeBlueRed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = _FakeDetector()
        self.shared_fc = torch.nn.Identity()
        self.cls_head = torch.nn.Linear(1, 2)


def test_combined_export_output_order_is_detector_blue_red():
    model = _FakeBlueRed()
    with torch.no_grad():
        model.detector.backbone.classifier.weight.fill_(1.0)
        model.detector.backbone.classifier.bias.fill_(0.1)
        model.cls_head.weight.copy_(torch.tensor([[2.0], [3.0]]))
        model.cls_head.bias.copy_(torch.tensor([0.2, 0.3]))

    wrapper = CombinedDetectorBlueRed(model)
    spec = torch.ones(2, 3, 4, 5)
    out = wrapper(spec)

    assert out.shape == (2, 3)
    torch.testing.assert_close(out[0], torch.tensor([1.1, 2.2, 3.3]))


def test_combined_export_rejects_checkpoint_without_detector():
    model = torch.nn.Module()
    model.detector = None

    with pytest.raises(ValueError, match="--detector-checkpoint"):
        CombinedDetectorBlueRed(model)


def test_dymn_dynamic_conv_export_forward_supports_batch_one_only():
    from models.dymn.dy_block import DynamicConv

    conv = DynamicConv(
        in_channels=1,
        out_channels=1,
        context_dim=1,
        kernel_size=1,
    )
    x = torch.randn(1, 1, 4, 5)
    g = [torch.randn(1, 1)]

    out = _dynamic_conv_export_forward(conv, x, g)
    assert out.shape == (1, 1, 4, 5)

    with pytest.raises(ValueError, match="batch size 1"):
        _dynamic_conv_export_forward(conv, torch.randn(2, 1, 4, 5), [torch.randn(2, 1)])
