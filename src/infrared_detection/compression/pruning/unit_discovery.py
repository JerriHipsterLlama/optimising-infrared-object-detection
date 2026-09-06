"""Discover and validate independent structural pruning units."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Literal

import torch
from torch import nn

from .dependency_graph import DependencyGraph, build_yolo_dependency_graph
from .importance import rank_output_channels
from .yolo_pruner import synchronize_module_channel_metadata


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


@dataclass(frozen=True)
class DependencyOperation:
    module_name: str
    module_type: str
    operation: str
    indices: tuple[int, ...]

    @property
    def affected_count(self) -> int:
        return len(self.indices)


@dataclass
class StructuralProbe:
    status: Literal["VALID", "GROUPED", "INVALID", "SEMANTICS_CHANGED"]
    unit_name: str
    original_channels: int
    requested_pruned_channels: int
    requested_remaining_channels: int
    actual_remaining_channels: int | None
    requested_pruning_ratio: float
    actual_pruning_ratio: float | None
    operations: tuple[DependencyOperation, ...]
    touched_modules: tuple[str, ...]
    reason: str | None
    error: str | None
    model: nn.Module | None


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


def _handler_name(handler: object) -> str:
    name = getattr(handler, "__name__", None)
    if isinstance(name, str):
        return name
    return type(handler).__name__


def _dependency_operations(
    model: nn.Module,
    group: Any,
) -> tuple[DependencyOperation, ...]:
    names_by_id = {id(module): name or "$model" for name, module in model.named_modules()}
    operations: list[DependencyOperation] = []
    for position, item in enumerate(group):
        module = item.dep.target.module
        module_name = names_by_id.get(id(module))
        if module_name is None:
            module_name = f"$op:{position}:{type(module).__name__}"
        operations.append(
            DependencyOperation(
                module_name=module_name,
                module_type=type(module).__name__,
                operation=_handler_name(item.dep.handler),
                indices=tuple(int(index) for index in item.idxs),
            )
        )
    return tuple(operations)


def serialize_dependency_group(operations: tuple[DependencyOperation, ...]) -> str:
    """Serialize every dependency operation without unstable object addresses."""

    payload = [
        {
            "module_name": operation.module_name,
            "module_type": operation.module_type,
            "operation": operation.operation,
            "indices": list(operation.indices),
            "affected_count": operation.affected_count,
        }
        for operation in operations
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _touched_modules(operations: tuple[DependencyOperation, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(operation.module_name for operation in operations))


def _has_coupled_output(
    model: nn.Module,
    unit_name: str,
    operations: tuple[DependencyOperation, ...],
) -> bool:
    modules = dict(model.named_modules())
    for operation in operations:
        if operation.module_name == unit_name:
            continue
        module = modules.get(operation.module_name)
        if isinstance(module, (nn.Conv2d, nn.Linear)) and operation.operation in {
            "prune_out_channels",
            "prune_out_features",
        }:
            return True
    return False


def _empty_probe(
    *,
    status: Literal["INVALID"],
    unit_name: str,
    original_channels: int,
    ratio: float,
    reason: str,
    error: str | None = None,
) -> StructuralProbe:
    return StructuralProbe(
        status=status,
        unit_name=unit_name,
        original_channels=original_channels,
        requested_pruned_channels=0,
        requested_remaining_channels=original_channels,
        actual_remaining_channels=None,
        requested_pruning_ratio=ratio,
        actual_pruning_ratio=None,
        operations=(),
        touched_modules=(),
        reason=reason,
        error=error,
        model=None,
    )


def validate_and_prune_unit(
    model: nn.Module,
    example_input: torch.Tensor,
    unit_name: str,
    ratio: float,
    baseline_contract: DetectContract | None,
    *,
    criterion: str = "l1",
    graph_builder: Callable[[nn.Module, torch.Tensor], DependencyGraph] = build_yolo_dependency_graph,
) -> StructuralProbe:
    """Audit and apply one independent structural pruning experiment."""

    modules = dict(model.named_modules())
    root = modules.get(unit_name)
    if not isinstance(root, nn.Conv2d):
        return _empty_probe(
            status="INVALID",
            unit_name=unit_name,
            original_channels=0,
            ratio=ratio,
            reason="missing_convolution",
        )
    original_channels = int(root.weight.shape[0])
    try:
        prune_count = requested_prune_count(original_channels, ratio)
    except ValueError as error:
        return _empty_probe(
            status="INVALID",
            unit_name=unit_name,
            original_channels=original_channels,
            ratio=ratio,
            reason="insufficient_channels",
            error=str(error),
        )
    requested_remaining = original_channels - prune_count

    try:
        import torch_pruning as tp

        ranked = rank_output_channels(root, criterion=criterion)
        prune_indices = [index for index, _score in ranked[:prune_count]]
        dependency = graph_builder(model, example_input)
        group = dependency.graph.get_pruning_group(
            root,
            tp.prune_conv_out_channels,
            idxs=prune_indices,
        )
        operations = _dependency_operations(model, group)
        touched = _touched_modules(operations)
        if not dependency.graph.check_pruning_group(group):
            return StructuralProbe(
                "INVALID", unit_name, original_channels, prune_count,
                requested_remaining, None, ratio, None, operations, touched,
                "dependency_group_rejected", None, None,
            )
        if _has_coupled_output(model, unit_name, operations):
            return StructuralProbe(
                "GROUPED", unit_name, original_channels, prune_count,
                requested_remaining, None, ratio, None, operations, touched,
                "coupled_output_channels", None, None,
            )

        group.prune()
        synchronize_module_channel_metadata(model)
        actual_remaining = int(root.weight.shape[0])
        actual_ratio = (original_channels - actual_remaining) / original_channels
        if actual_remaining != requested_remaining:
            return StructuralProbe(
                "INVALID", unit_name, original_channels, prune_count,
                requested_remaining, actual_remaining, ratio, actual_ratio,
                operations, touched, "unexpected_channel_count", None, None,
            )
        model.eval()
        with torch.inference_mode():
            output = model(example_input)
        if baseline_contract is not None:
            try:
                validate_detect_contract(baseline_contract, model, output)
            except ValueError as error:
                return StructuralProbe(
                    "SEMANTICS_CHANGED", unit_name, original_channels, prune_count,
                    requested_remaining, actual_remaining, ratio, actual_ratio,
                    operations, touched, "detect_contract_changed", str(error), None,
                )
        return StructuralProbe(
            "VALID", unit_name, original_channels, prune_count,
            requested_remaining, actual_remaining, ratio, actual_ratio,
            operations, touched, None, None, model,
        )
    except Exception as error:
        return StructuralProbe(
            "INVALID", unit_name, original_channels, prune_count,
            requested_remaining, None, ratio, None, (), (),
            "dependency_or_forward_failure", str(error), None,
        )
