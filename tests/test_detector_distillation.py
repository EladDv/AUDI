import torch
from lightning import LightningModule

from audi.training.detector import DroneDetector


class ConstantTeacher(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.logits[: wav.size(0)]


def test_distillation_loss_matches_teacher_probabilities():
    detector = DroneDetector.__new__(DroneDetector)
    LightningModule.__init__(detector)
    detector._teacher = ConstantTeacher(torch.tensor([2.0, -2.0]))
    detector._distillation_weight = 1.0
    detector._distillation_temperature = 2.0

    wav = torch.zeros(2, 16)
    student_logits = torch.zeros(2)

    loss = detector._compute_distillation_loss(wav, student_logits)

    assert loss.item() > 0.0


def test_save_checkpoint_strips_distillation_teacher_state():
    detector = DroneDetector.__new__(DroneDetector)
    LightningModule.__init__(detector)
    checkpoint = {
        "state_dict": {
            "_teacher.backbone.weight": torch.ones(1),
            "backbone.weight": torch.zeros(1),
        }
    }

    detector.on_save_checkpoint(checkpoint)

    assert "_teacher.backbone.weight" not in checkpoint["state_dict"]
    assert "backbone.weight" in checkpoint["state_dict"]


def test_optimizer_excludes_frozen_teacher_params():
    detector = DroneDetector.__new__(DroneDetector)
    LightningModule.__init__(detector)
    detector.student = torch.nn.Linear(2, 1)
    detector._teacher = torch.nn.Linear(2, 1)
    for p in detector._teacher.parameters():
        p.requires_grad = False
    detector._lr = 1e-3
    detector._weight_decay = 0.0
    detector._schedule = "constant"

    opt = detector.configure_optimizers()
    opt_param_ids = {
        id(p) for group in opt.param_groups for p in group["params"]
    }
    teacher_param_ids = {id(p) for p in detector._teacher.parameters()}

    assert teacher_param_ids.isdisjoint(opt_param_ids)
