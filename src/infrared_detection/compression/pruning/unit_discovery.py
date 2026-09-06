"""Discover and validate independent structural pruning units."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import torch
from torch import nn


Region = Literal["backbone", "neck", "detect_head"]


@dataclass(frozen=True)
class PruningUnit:
    name: str
    region: Region
    original_channels: int


@dataclass(frozen=True)
class DetectContract:
    number_of_scales: int
    prediction_rank: int
    prediction_channels: int
    boxes_rank: int
    box_channels: int
    scores_rank: int
    class_channels: int
    nc: int
    reg_max: int
    no: int
    nl: int


def requested_prune_count(channels: int, ratio: float) -> int:
    """Convert a requested ratio into a valid dense-model channel count."""

    if channels < 2:
        raise ValueError("A pruning unit must have at least two output channels.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("Pruning ratio must be between zero and one.")
    count = round(channels * ratio)
    if not 1 <= count < channels:
        raise ValueError("Requested ratio does not remove a valid channel count.")
    return count


def _top_level_modules(model: nn.Module) -> list[nn.Module]:
    modules = getattr(model, "model", None)
    if not isinstance(modules, (nn.Sequential, nn.ModuleList)):
        raise ValueError("Expected a YOLO model with an ordered top-level model container.")
    return list(modules)


def discover_pruning_units(model: nn.Module) -> tuple[PruningUnit, ...]:
    """Enumerate convolution roots and infer detector subsystem boundaries."""

    top_level = _top_level_modules(model)
    detect_indices = [
        index for index, module in enumerate(top_level) if type(module).__name__ == "Detect"
    ]
    if len(detect_indices) != 1:
        raise ValueError("Expected exactly one top-level Detect module.")
    detect_index = detect_indices[0]
    neck_indices = [
        index
        for index, module in enumerate(top_level[:detect_index])
        if type(module).__name__ in {"Upsample", "Concat"}
    ]
    if not neck_indices:
        raise ValueError("Could not infer the YOLO neck boundary.")
    neck_index = neck_indices[0]

    units: list[PruningUnit] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        match = re.match(r"^model\.(\d+)(?:\.|$)", name)
        if match is None:
            raise ValueError(f"Cannot assign architectural region to convolution {name!r}.")
        top_index = int(match.group(1))
        region: Region
        if top_index == detect_index:
            region = "detect_head"
        elif top_index >= neck_index:
            region = "neck"
        else:
            region = "backbone"
        units.append(PruningUnit(name, region, int(module.weight.shape[0])))
    return tuple(units)


def _detect_module(model: nn.Module) -> nn.Module:
    matches = [module for module in _top_level_modules(model) if type(module).__name__ == "Detect"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one top-level Detect module.")
    return matches[0]


def capture_detect_contract(model: nn.Module, output: object) -> DetectContract:
    """Capture the prediction and class semantics that pruning must preserve."""

    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("Expected YOLO Detect evaluation output as a two-item tuple.")
    prediction, raw = output
    if not isinstance(prediction, torch.Tensor) or not isinstance(raw, dict):
        raise ValueError("Expected tensor predictions and a Detect output mapping.")
    boxes = raw.get("boxes")
    scores = raw.get("scores")
    features = raw.get("feats")
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        raise ValueError("Detect output mapping must contain boxes and scores tensors.")
    if not isinstance(features, (list, tuple)):
        raise ValueError("Detect output mapping must contain feature scales.")
    detect = _detect_module(model)
    return DetectContract(
        number_of_scales=len(features),
        prediction_rank=prediction.ndim,
        prediction_channels=int(prediction.shape[1]),
        boxes_rank=boxes.ndim,
        box_channels=int(boxes.shape[1]),
        scores_rank=scores.ndim,
        class_channels=int(scores.shape[1]),
        nc=int(getattr(detect, "nc")),
        reg_max=int(getattr(detect, "reg_max")),
        no=int(getattr(detect, "no")),
        nl=int(getattr(detect, "nl")),
    )


def validate_detect_contract(
    expected: DetectContract,
    model: nn.Module,
    output: object,
) -> DetectContract:
    """Reject a pruned model whose Detect output semantics changed."""

    actual = capture_detect_contract(model, output)
    if actual != expected:
        raise ValueError(f"Detect contract changed: expected {expected!r}, got {actual!r}.")
    return actual
