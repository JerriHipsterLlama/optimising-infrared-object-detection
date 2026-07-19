"""Detection-aware knowledge distillation as an independent experiment factor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DetectionDistillationLoss(nn.Module):
    """Combine prediction and intermediate-feature distillation losses."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_layers: Iterable[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ):
        super().__init__()
        del feature_layers
        self.teacher = teacher.eval()
        self.student = student
        self.weights = {"prediction": 1.0, "feature": 1.0, **dict(weights or {})}
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if torch.is_tensor(item)]
            if not tensors:
                raise TypeError("Distillation output contains no tensors.")
            return tensors[0]
        raise TypeError("Distillation outputs must contain tensors.")

    def component_losses(self, student_outputs: Mapping[str, Any], teacher_outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for name in ("prediction", "feature"):
            if name not in student_outputs or name not in teacher_outputs:
                continue
            student_tensor = self._as_tensor(student_outputs[name])
            teacher_tensor = self._as_tensor(teacher_outputs[name]).detach()
            if student_tensor.shape != teacher_tensor.shape:
                raise ValueError(f"KD tensor shape mismatch for {name}: {student_tensor.shape} vs {teacher_tensor.shape}")
            losses[name] = F.mse_loss(student_tensor, teacher_tensor) * float(self.weights[name])
        if not losses:
            raise ValueError("KD requires at least one prediction or feature output.")
        losses["total"] = torch.stack(list(losses.values())).sum()
        return losses

    def forward(self, student_outputs: Mapping[str, Any], teacher_outputs: Mapping[str, Any]) -> torch.Tensor:
        return self.component_losses(student_outputs, teacher_outputs)["total"]


def train_student_with_kd(teacher: nn.Module, student: nn.Module, dataloader: Iterable[Any], config: Mapping[str, Any]) -> nn.Module:
    """Run a small generic KD loop for adapters that provide output mappings."""

    teacher.eval()
    student.train()
    criterion = DetectionDistillationLoss(teacher, student, weights=config.get("weights"))
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config.get("lr", 1e-4)))
    device = torch.device(config.get("device", "cpu"))
    teacher.to(device)
    student.to(device)
    for _ in range(int(config.get("epochs", 1))):
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_prediction = teacher(inputs)
            student_prediction = student(inputs)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion({"prediction": student_prediction}, {"prediction": teacher_prediction})
            loss.backward()
            optimizer.step()
    return student
