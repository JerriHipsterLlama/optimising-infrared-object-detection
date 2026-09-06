"""Accuracy sensitivity calculations and screening orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
import yaml

from infrared_detection.compression.pruning.unit_discovery import (
    DetectContract,
    PruningUnit,
    StructuralProbe,
    capture_detect_contract,
    discover_pruning_units,
    serialize_dependency_group,
    validate_and_prune_unit,
)


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


@dataclass(frozen=True)
class ValidationMetrics:
    map50_95: float
    map50: float
    precision: float
    recall: float


@dataclass(frozen=True)
class SensitivityAdapters:
    @classmethod
    def defaults(cls) -> "SensitivityAdapters":
        """Build production Ultralytics and Torch-Pruning operations lazily."""

        def load_model(checkpoint: Path) -> Any:
            from infrared_detection.evaluation.single_layer_performance_screening import (
                _install_legacy_pathlib_checkpoint_compatibility,
            )
            from ultralytics import YOLO

            _install_legacy_pathlib_checkpoint_compatibility()
            return YOLO(str(checkpoint))

        def evaluate(wrapper: Any, arguments: Mapping[str, Any]) -> ValidationMetrics:
            result = wrapper.val(**dict(arguments))
            box = result.box
            return ValidationMetrics(
                map50_95=float(box.map),
                map50=float(box.map50),
                precision=float(box.mp),
                recall=float(box.mr),
            )

        def make_example(model: nn.Module, size: int) -> torch.Tensor:
            parameter = next(model.parameters())
            return torch.randn(
                1, 3, size, size, device=parameter.device, dtype=parameter.dtype
            )

        def versions() -> Mapping[str, str]:
            import torch_pruning
            import ultralytics

            return {
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "torch_pruning": getattr(torch_pruning, "__version__", "unknown"),
            }

        return cls(
            load_model=load_model,
            unwrap_model=lambda wrapper: wrapper.model,
            replace_model=lambda wrapper, model: setattr(wrapper, "model", model),
            evaluate=evaluate,
            make_example_input=make_example,
            capture_contract=capture_detect_contract,
            forward=lambda model, example: model.eval()(example),
            versions=versions,
        )

    load_model: Callable[[Path], Any]
    unwrap_model: Callable[[Any], nn.Module]
    replace_model: Callable[[Any, nn.Module], None]
    evaluate: Callable[[Any, Mapping[str, Any]], ValidationMetrics]
    make_example_input: Callable[[nn.Module, int], torch.Tensor]
    discover_units: Callable[[nn.Module], tuple[PruningUnit, ...]] = discover_pruning_units
    capture_contract: Callable[[nn.Module, Any], DetectContract | None] | None = None
    forward: Callable[[nn.Module, torch.Tensor], Any] | None = None
    probe: Callable[..., StructuralProbe] = validate_and_prune_unit
    versions: Callable[[], Mapping[str, str]] = lambda: {}


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


def _validation_arguments(config: SensitivityConfig) -> dict[str, Any]:
    arguments = dict(config.validation)
    arguments.update(
        data=str(config.dataset_yaml),
        split=config.split,
        plots=False,
        save_json=False,
        verbose=False,
    )
    return arguments


def _baseline_row(config: SensitivityConfig, metrics: ValidationMetrics) -> dict[str, Any]:
    return {
        "candidate_id": "baseline",
        "status": "BASELINE",
        "map50_95": metrics.map50_95,
        "map50": metrics.map50,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "checkpoint": str(config.checkpoint),
        "dataset": str(config.dataset_yaml),
        "split": config.split,
        "importance": config.importance,
    }


def _float_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Required numeric result field is missing: {key}")
    return float(value)


def _candidate_id(unit_name: str, ratio: float) -> str:
    return f"{unit_name}@{format(float(ratio), 'g')}"


def _probe_row(
    config: SensitivityConfig,
    unit: PruningUnit,
    probe: StructuralProbe,
    validation_fingerprint: str,
) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(unit.name, probe.requested_pruning_ratio),
        "status": probe.status,
        "pruning_unit": unit.name,
        "architectural_region": unit.region,
        "original_channels": probe.original_channels,
        "requested_pruned_channels": probe.requested_pruned_channels,
        "requested_remaining_channels": probe.requested_remaining_channels,
        "actual_remaining_channels": probe.actual_remaining_channels,
        "requested_pruning_ratio": probe.requested_pruning_ratio,
        "actual_pruning_ratio": probe.actual_pruning_ratio,
        "touched_modules": ";".join(probe.touched_modules),
        "dependency_group_json": serialize_dependency_group(probe.operations),
        "status_reason": probe.reason,
        "error": probe.error,
        "checkpoint": str(config.checkpoint),
        "dataset": str(config.dataset_yaml),
        "split": config.split,
        "importance": config.importance,
        "validation_fingerprint": validation_fingerprint,
    }


def run_accuracy_sensitivity(
    config_path: str | Path,
    *,
    adapters: SensitivityAdapters | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run independent dense-checkpoint structural sensitivity experiments."""

    if adapters is None:
        adapters = SensitivityAdapters.defaults()
    config = load_sensitivity_config(config_path)
    versions = dict(adapters.versions())
    fingerprint = experiment_fingerprint(config, versions=versions)
    resume = load_resume_artifacts(config.output_dir, expected_fingerprint=fingerprint)
    rows_by_id = {row["candidate_id"]: dict(row) for row in resume.rows}
    validation = _validation_arguments(config)
    validation_fingerprint = hashlib.sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    baseline_wrapper = adapters.load_model(config.checkpoint)
    baseline_model = adapters.unwrap_model(baseline_wrapper)
    units = adapters.discover_units(baseline_model)
    baseline_example = adapters.make_example_input(baseline_model, config.example_image_size)
    baseline_contract = None
    if adapters.capture_contract is not None and adapters.forward is not None:
        baseline_contract = adapters.capture_contract(
            baseline_model,
            adapters.forward(baseline_model, baseline_example),
        )

    baseline = rows_by_id.get("baseline")
    if baseline is None or baseline.get("status") != "BASELINE":
        baseline_metrics = adapters.evaluate(baseline_wrapper, validation)
        baseline = _baseline_row(config, baseline_metrics)
        baseline["validation_fingerprint"] = validation_fingerprint
        rows_by_id["baseline"] = baseline
        if on_result is not None:
            on_result(dict(baseline))
    baseline_map = _float_value(baseline, "map50_95")

    manifest: dict[str, Any] = {
        "fingerprint": fingerprint,
        "versions": versions,
        "baseline": dict(baseline),
        "detect_contract": None if baseline_contract is None else baseline_contract.__dict__,
        "units": [unit.__dict__ for unit in units],
        "ratios": list(config.ratios),
        "importance": config.importance,
        "complete": False,
    }
    write_artifacts(
        config.output_dir,
        rows_by_id.values(),
        build_unit_rankings(rows_by_id.values(), config.ratios),
        manifest,
        ratios=config.ratios,
    )

    for unit in units:
        for ratio in config.ratios:
            candidate_id = _candidate_id(unit.name, ratio)
            if candidate_id in resume.reusable_candidate_ids:
                continue
            try:
                wrapper = adapters.load_model(config.checkpoint)
                dense_model = adapters.unwrap_model(wrapper)
                example = adapters.make_example_input(dense_model, config.example_image_size)
                probe = adapters.probe(
                    dense_model,
                    example,
                    unit.name,
                    ratio,
                    baseline_contract,
                    criterion=config.importance,
                )
                row = _probe_row(config, unit, probe, validation_fingerprint)
                if probe.status == "VALID":
                    if probe.model is None:
                        raise RuntimeError("A VALID structural probe did not return a pruned model.")
                    adapters.replace_model(wrapper, probe.model)
                    metrics = adapters.evaluate(wrapper, validation)
                    row.update(
                        status="COMPLETED",
                        map50_95=metrics.map50_95,
                        map50=metrics.map50,
                        precision=metrics.precision,
                        recall=metrics.recall,
                    )
                    point = calculate_point_metrics(
                        baseline_map,
                        metrics.map50_95,
                        float(probe.actual_pruning_ratio),
                    )
                    row.update(point.__dict__)
            except Exception as error:
                row = {
                    "candidate_id": candidate_id,
                    "status": "ERROR",
                    "pruning_unit": unit.name,
                    "architectural_region": unit.region,
                    "original_channels": unit.original_channels,
                    "requested_pruning_ratio": ratio,
                    "status_reason": "runtime_error",
                    "error": str(error),
                    "checkpoint": str(config.checkpoint),
                    "dataset": str(config.dataset_yaml),
                    "split": config.split,
                    "importance": config.importance,
                    "validation_fingerprint": validation_fingerprint,
                }
            rows_by_id[candidate_id] = row
            rankings = build_unit_rankings(rows_by_id.values(), config.ratios)
            write_artifacts(
                config.output_dir,
                rows_by_id.values(),
                rankings,
                manifest,
                ratios=config.ratios,
            )
            if on_result is not None:
                on_result(dict(row))

    manifest["complete"] = all(
        _candidate_id(unit.name, ratio) in rows_by_id
        and rows_by_id[_candidate_id(unit.name, ratio)].get("status") != "ERROR"
        for unit in units
        for ratio in config.ratios
    )
    rankings = build_unit_rankings(rows_by_id.values(), config.ratios)
    write_artifacts(config.output_dir, rows_by_id.values(), rankings, manifest, ratios=config.ratios)
    return list(rows_by_id.values())


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
