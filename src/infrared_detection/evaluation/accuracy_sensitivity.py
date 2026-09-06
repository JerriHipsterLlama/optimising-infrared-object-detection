"""Accuracy sensitivity calculations and screening orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import yaml


DEFAULT_RATIOS = (0.125, 0.25, 0.375, 0.5)

DETAILED_FIELDS = (
    "candidate_id", "status", "pruning_unit", "architectural_region",
    "original_channels", "requested_pruned_channels", "requested_remaining_channels",
    "actual_remaining_channels", "requested_pruning_ratio", "actual_pruning_ratio",
    "map50_95", "map50", "precision", "recall", "map50_95_change_from_baseline",
    "accuracy_degradation", "point_sensitivity", "touched_modules",
    "dependency_group_json", "status_reason", "error", "checkpoint", "dataset",
    "split", "importance", "validation_fingerprint", "elapsed_validation_seconds",
)


def aggregate_fields(ratios: Sequence[float]) -> tuple[str, ...]:
    fields = [
        "rank", "pruning_unit", "architectural_region", "curve_status",
        "original_channels", "successful_ratio_count", "maximum_achieved_ratio",
        "normalized_auc_sensitivity",
    ]
    for ratio in ratios:
        slug = format(float(ratio), "g").replace(".", "_")
        fields.extend((f"map50_95_at_{slug}", f"degradation_at_{slug}"))
    return tuple(fields)


AGGREGATE_FIELDS = aggregate_fields(DEFAULT_RATIOS)


@dataclass(frozen=True)
class PointMetrics:
    accuracy_degradation: float
    map50_95_change_from_baseline: float
    point_sensitivity: float


@dataclass(frozen=True)
class SensitivityConfig:
    checkpoint: Path
    dataset_yaml: Path
    split: str
    validation: Mapping[str, Any]
    ratios: tuple[float, ...]
    importance: str
    example_image_size: int
    output_dir: Path


@dataclass(frozen=True)
class ResumeArtifacts:
    rows: tuple[dict[str, str], ...]
    manifest: Mapping[str, Any]
    reusable_candidate_ids: set[str]


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


def _resolved_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_sensitivity_config(config_path: str | Path) -> SensitivityConfig:
    """Load and validate the standalone screening configuration."""

    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sensitivity configuration must be a mapping.")
    checkpoint = _resolved_path(payload["model"]["checkpoint"])
    dataset = _resolved_path(payload["data"]["dataset_yaml"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset}")
    ratios = tuple(float(value) for value in payload["pruning"]["ratios"])
    if not ratios or len(set(ratios)) != len(ratios) or any(
        not 0.0 < ratio < 1.0 for ratio in ratios
    ):
        raise ValueError("Pruning ratios must be nonempty, unique, and between zero and one.")
    importance = str(payload["pruning"].get("importance", "l1"))
    if importance != "l1":
        raise ValueError("Supported importance criteria: l1")
    example_size = int(payload["pruning"]["example_image_size"])
    if example_size <= 0:
        raise ValueError("Example image size must be positive.")
    validation = dict(payload.get("validation", {}))
    if not validation:
        raise ValueError("Validation settings must be provided.")
    return SensitivityConfig(
        checkpoint=checkpoint,
        dataset_yaml=dataset,
        split=str(payload["data"].get("split", "val")),
        validation=MappingProxyType(validation),
        ratios=ratios,
        importance=importance,
        example_image_size=example_size,
        output_dir=_resolved_path(payload["experiment"]["output_dir"]),
    )


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def experiment_fingerprint(
    config: SensitivityConfig,
    *,
    versions: Mapping[str, str],
) -> str:
    payload = {
        "checkpoint": _file_identity(config.checkpoint),
        "dataset": _file_identity(config.dataset_yaml),
        "split": config.split,
        "validation": dict(config.validation),
        "ratios": config.ratios,
        "importance": config.importance,
        "importance_version": "l1-filter-sum-v1",
        "example_image_size": config.example_image_size,
        "versions": dict(versions),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return stream.getvalue()


def write_artifacts(
    output_dir: str | Path,
    rows: Iterable[Mapping[str, Any]],
    rankings: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> None:
    """Atomically refresh detailed results, aggregate rankings, and manifest."""

    directory = Path(output_dir)
    _atomic_write_text(directory / "results.csv", _csv_text(rows, DETAILED_FIELDS))
    _atomic_write_text(
        directory / "unit_sensitivity_ranking.csv",
        _csv_text(rankings, aggregate_fields(ratios)),
    )
    _atomic_write_text(
        directory / "manifest.json",
        json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
    )


def load_resume_artifacts(
    output_dir: str | Path,
    *,
    expected_fingerprint: str,
) -> ResumeArtifacts:
    directory = Path(output_dir)
    paths = [
        directory / "results.csv",
        directory / "unit_sensitivity_ranking.csv",
        directory / "manifest.json",
    ]
    existing = [path.exists() for path in paths]
    if not any(existing):
        return ResumeArtifacts((), {}, set())
    if not all(existing):
        raise ValueError("Resume artifacts are incomplete.")
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("Experiment fingerprint does not match saved artifacts.")
    with paths[0].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DETAILED_FIELDS:
            raise ValueError("Detailed result CSV header does not match this tool version.")
        rows = tuple(dict(row) for row in reader)
    reusable_statuses = {"BASELINE", "COMPLETED", "GROUPED", "INVALID", "SEMANTICS_CHANGED"}
    reusable = {
        row["candidate_id"] for row in rows if row.get("status") in reusable_statuses
    }
    return ResumeArtifacts(rows, manifest, reusable)


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
