"""Pareto-front analysis for compression experiments."""

from __future__ import annotations

from typing import Any, Iterable


REQUIRED_REPORT_COLUMNS = {
    "variant",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "parameter_count",
    "serialized_bytes",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_memory_mb",
    "fps",
    "energy_mj_per_inference",
}


def pareto_optimal_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows not dominated on accuracy, latency, and memory."""

    rows = list(rows)
    valid = [row for row in rows if all(row.get(key) is not None for key in ("map50_95", "latency_p50_ms", "peak_memory_mb"))]
    result = []
    for candidate in valid:
        dominated = any(
            other is not candidate
            and other["map50_95"] >= candidate["map50_95"]
            and other["latency_p50_ms"] <= candidate["latency_p50_ms"]
            and other["peak_memory_mb"] <= candidate["peak_memory_mb"]
            and (
                other["map50_95"] > candidate["map50_95"]
                or other["latency_p50_ms"] < candidate["latency_p50_ms"]
                or other["peak_memory_mb"] < candidate["peak_memory_mb"]
            )
            for other in valid
        )
        if not dominated:
            result.append(candidate)
    return result

