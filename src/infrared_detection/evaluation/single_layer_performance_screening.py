"""Deterministic planning primitives for single-layer performance screening."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import re
from typing import Any, Iterable, Sequence

from torch import nn

from infrared_detection.compression.pruning import rank_filters_by_minimum_weight


@dataclass(frozen=True)
class FilterRanking:
    original_index: int
    score: float


@dataclass(frozen=True)
class LayerPlan:
    name: str
    filters_before: int
    ranking: tuple[FilterRanking, ...]


_RESEARCH_FIELDS = (
    "candidate_id", "model_variant", "hardware", "status", "layer", "filter_rank",
    "original_filter_index", "physical_filter_index", "minimum_weight_score", "filters_before",
    "filters_removed", "filters_remaining", "map50_95", "map50_95_drop", "map50", "precision",
    "recall", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "fps",
)


def resolve_layer_patterns(model: nn.Module, patterns: Iterable[str]) -> list[str]:
    """Resolve every wildcard pattern in model order, rejecting invalid selections."""

    modules = [(name, module) for name, module in model.named_modules() if name]
    selected_names: set[str] = set()
    for pattern in patterns:
        matches = [(name, module) for name, module in modules if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"Layer pattern {pattern!r} matched no modules.")
        non_convolutions = [name for name, module in matches if not isinstance(module, nn.Conv2d)]
        if non_convolutions:
            raise ValueError(f"Layer pattern {pattern!r} matched non-Conv2d modules: {non_convolutions!r}.")
        selected_names.update(name for name, _ in matches)
    return [name for name, _ in modules if name in selected_names]


def build_layer_plan(model: nn.Module, layer_name: str) -> LayerPlan:
    """Build one immutable dense-model ranking plan for a convolution layer."""

    modules = dict(model.named_modules())
    module = modules.get(layer_name)
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f"Layer {layer_name!r} must resolve to a torch.nn.Conv2d module.")
    filters_before = int(module.out_channels)
    if filters_before < 2:
        raise ValueError("A planned layer must contain at least two output filters.")
    ranking = tuple(FilterRanking(original_index=index, score=float(score)) for index, score in rank_filters_by_minimum_weight(module))
    return LayerPlan(name=layer_name, filters_before=filters_before, ranking=ranking)


def physical_index(remaining_original_indices: Sequence[int], original_index: int) -> int:
    """Translate a dense-model filter index to its current tensor position."""

    try:
        return remaining_original_indices.index(original_index)
    except ValueError as exc:
        raise ValueError(f"Original filter index {original_index} is not remaining.") from exc


def planned_rows(model_variant: str, hardware: str, layer_plans: Iterable[LayerPlan]) -> list[dict[str, Any]]:
    """Create deterministic, unevaluated rows for every valid pruning step."""

    rows: list[dict[str, Any]] = []
    for plan in layer_plans:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", plan.name).strip("-")
        for filter_rank, entry in enumerate(plan.ranking[:-1], start=1):
            row = {field: None for field in _RESEARCH_FIELDS}
            row.update(
                candidate_id=f"{model_variant}-{hardware}-{slug}-rank-{filter_rank}",
                model_variant=model_variant,
                hardware=hardware,
                status="planned",
                layer=plan.name,
                filter_rank=filter_rank,
                original_filter_index=entry.original_index,
                minimum_weight_score=entry.score,
                filters_before=plan.filters_before,
                filters_removed=filter_rank,
                filters_remaining=plan.filters_before - filter_rank,
            )
            rows.append(row)
    return rows
