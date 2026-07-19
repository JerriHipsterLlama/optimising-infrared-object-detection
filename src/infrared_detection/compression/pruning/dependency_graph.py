"""Optional torch-pruning dependency graph adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class DependencyGraph:
    model: nn.Module
    graph: Any
    modules: dict[str, nn.Module]


def build_yolo_dependency_graph(model: nn.Module, example_input: torch.Tensor) -> DependencyGraph:
    """Build a graph that propagates channel surgery through YOLO dependencies."""

    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError(
            "Structured pruning requires torch-pruning. Install the project dependency before pruning."
        ) from exc

    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    return DependencyGraph(model=model, graph=graph, modules=modules)
