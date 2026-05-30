import pytest
import torch
from lightning import LightningModule

pytest.importorskip("torchvision")

from audi.training.detector import DroneDetector


class TinyWrappedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 4),
        )
        self.classifier = torch.nn.Linear(4, 1)


def test_freeze_backbone_keeps_classifier_trainable():
    detector = DroneDetector.__new__(DroneDetector)
    LightningModule.__init__(detector)
    detector.backbone = TinyWrappedBackbone()

    detector._set_backbone_frozen(True)

    assert all(not p.requires_grad for p in detector.backbone.backbone.parameters())
    assert all(p.requires_grad for p in detector.backbone.classifier.parameters())
    assert not detector.backbone.backbone.training
    assert detector.backbone.classifier.training
