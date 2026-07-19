import torch
from torch import nn

from infrared_detection.compression.distillation import DetectionDistillationLoss


def test_kd_freezes_teacher_and_backpropagates_to_student():
    teacher = nn.Linear(4, 3)
    student = nn.Linear(4, 3)
    criterion = DetectionDistillationLoss(teacher, student, weights={"prediction": 1.0, "feature": 1.0})

    inputs = torch.randn(2, 4)
    student_outputs = {"prediction": student(inputs), "feature": student(inputs)}
    teacher_outputs = {"prediction": teacher(inputs), "feature": teacher(inputs)}
    loss = criterion(student_outputs, teacher_outputs)
    loss.backward()

    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert student.weight.grad is not None
    assert torch.isfinite(student.weight.grad).all()


def test_kd_reports_stable_component_values():
    teacher = nn.Identity()
    student = nn.Identity()
    criterion = DetectionDistillationLoss(teacher, student)

    losses = criterion.component_losses(
        {"prediction": torch.ones(2, 3), "feature": torch.ones(2, 4)},
        {"prediction": torch.zeros(2, 3), "feature": torch.zeros(2, 4)},
    )

    assert set(losses) == {"prediction", "feature", "total"}
    assert losses["total"].item() == 2.0
