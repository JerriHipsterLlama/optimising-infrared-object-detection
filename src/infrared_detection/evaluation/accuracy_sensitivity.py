"""Accuracy sensitivity calculations and screening orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RATIOS = (0.125, 0.25, 0.375, 0.5)


@dataclass(frozen=True)
class PointMetrics:
    accuracy_degradation: float
    map50_95_change_from_baseline: float
    point_sensitivity: float


def calculate_point_metrics(
    baseline_map50_95: float,
    candidate_map50_95: float,
    actual_pruning_ratio: float,
) -> PointMetrics:
    """Calculate signed degradation, change, and ratio-normalized sensitivity."""

    if actual_pruning_ratio <= 0:
        raise ValueError("Achieved pruning ratio must be positive.")
    degradation = float(baseline_map50_95) - float(candidate_map50_95)
    return PointMetrics(
        accuracy_degradation=degradation,
        map50_95_change_from_baseline=-degradation,
        point_sensitivity=degradation / float(actual_pruning_ratio),
    )


def normalized_degradation_auc(points: Sequence[tuple[float, float]]) -> float:
    """Return trapezoidal degradation AUC normalized by maximum achieved ratio."""

    ordered = sorted((float(ratio), float(drop)) for ratio, drop in points)
    if not ordered or ordered[0][0] <= 0 or any(
        right[0] <= left[0] for left, right in pairwise(ordered)
    ):
        raise ValueError("Achieved ratios must be positive and strictly increasing.")
    curve = [(0.0, 0.0), *ordered]
    area = sum(
        (right_ratio - left_ratio) * (left_drop + right_drop) / 2.0
        for (left_ratio, left_drop), (right_ratio, right_drop) in pairwise(curve)
    )
    return area / ordered[-1][0]


def _ratio_slug(ratio: float) -> str:
    return format(float(ratio), "g").replace(".", "_")


def build_unit_rankings(
    rows: Iterable[Mapping[str, Any]],
    requested_ratios: Sequence[float],
) -> list[dict[str, Any]]:
    """Build complete-curve AUC rows and leave incomplete curves unranked."""

    expected = tuple(float(ratio) for ratio in requested_ratios)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        unit = row.get("pruning_unit")
        if unit:
            grouped.setdefault(str(unit), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for unit_name in sorted(grouped):
        unit_rows = grouped[unit_name]
        completed = [row for row in unit_rows if row.get("status") == "COMPLETED"]
        by_requested = {
            float(row["requested_pruning_ratio"]): row for row in completed
        }
        complete = len(by_requested) == len(expected) and set(by_requested) == set(expected)
        first = unit_rows[0]
        aggregate_row: dict[str, Any] = {
            "rank": None,
            "pruning_unit": unit_name,
            "architectural_region": first.get("architectural_region"),
            "curve_status": "COMPLETE" if complete else "INCOMPLETE",
            "original_channels": first.get("original_channels"),
            "successful_ratio_count": len(completed),
            "maximum_achieved_ratio": None,
            "normalized_auc_sensitivity": None,
        }
        for ratio in expected:
            point = by_requested.get(ratio)
            slug = _ratio_slug(ratio)
            aggregate_row[f"map50_95_at_{slug}"] = None if point is None else point.get("map50_95")
            aggregate_row[f"degradation_at_{slug}"] = (
                None if point is None else point.get("accuracy_degradation")
            )
        if complete:
            points = [
                (float(by_requested[ratio]["actual_pruning_ratio"]), float(by_requested[ratio]["accuracy_degradation"]))
                for ratio in expected
            ]
            aggregate_row["maximum_achieved_ratio"] = max(ratio for ratio, _drop in points)
            aggregate_row["normalized_auc_sensitivity"] = normalized_degradation_auc(points)
        aggregate.append(aggregate_row)

    complete_rows = sorted(
        (row for row in aggregate if row["curve_status"] == "COMPLETE"),
        key=lambda row: (-float(row["normalized_auc_sensitivity"]), row["pruning_unit"]),
    )
    for rank, row in enumerate(complete_rows, start=1):
        row["rank"] = rank
    incomplete_rows = sorted(
        (row for row in aggregate if row["curve_status"] != "COMPLETE"),
        key=lambda row: row["pruning_unit"],
    )
    return [*complete_rows, *incomplete_rows]
