"""Knowledge-distillation utilities for detection experiments."""

from .detection_kd import DetectionDistillationLoss, train_student_with_kd

__all__ = ["DetectionDistillationLoss", "train_student_with_kd"]
