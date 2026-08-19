"""Config-driven two-stage structural cluster-pruning evaluation."""

from __future__ import annotations

import inspect
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrared_detection.common.experiment_config import load_experiment_config, resolve_repo_path
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv
from infrared_detection.evaluation.cluster_candidates import classify_candidate, select_cluster_candidates


Metrics = dict[str, Any]

_ORIN_TARGET = "jetson_orin_nano"
_FEASIBLE_STATUSES = frozenset({"primary_feasible", "exploratory_feasible"})
_RESUMABLE_FILTERWISE_STATUSES = frozenset({"completed", "screened_in", "screened_out", "rejected_accuracy"})
_TERMINAL_FILTERWISE_STATUSES = frozenset({"skipped"})
_FILTERWISE_CHANNEL_MISMATCH_RE = re.compile(
    r"Given groups=\d+, weight of size \[\d+, \d+, \d+, \d+\], "
    r"expected input\[\d+, \d+, \d+, \d+\] to have \d+ channels, "
    r"but got \d+ channels instead"
)
_JETSON_HARDWARE_FIELDS = (
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "fps",
    "peak_memory_mb",
    "power_w",
    "energy_mj_per_inference",
    "temperature_c",
)


def run_structural_probe(model: Any, layer: str, cluster_size: int, ratio: float) -> Any:
    """Compatibility hook for injected probe implementations.

    Production adapters close over the full configuration so they can create a
    model-shaped example input.  Keeping this narrow hook makes failure
    isolation straightforward to test without importing PyTorch at module load.
    """

    del model, layer, cluster_size, ratio
    raise RuntimeError("Structural probe adapter is not configured.")


@dataclass
class ClusterEvaluationAdapters:
    """Side-effect boundary for the cluster-pruning workflow."""

    load_model: Callable[[Path], Any]
    evaluate: Callable[[Any, Mapping[str, Any], str], Metrics]
    safe_layers: Callable[[Any, Mapping[str, Any]], list[str]]
    make_probe: Callable[[Any, str, int, float], Any]
    make_global: Callable[[Any, int, float], Any]
    fine_tune: Callable[[Any, Mapping[str, Any], Path], Path]
    reload_model: Callable[[Path], Any]
    export: Callable[[Any, Mapping[str, Any], Path], Path]
    profile: Callable[[Path, str], Metrics]
    stats: Callable[[Any], Metrics]
    artifact_stats: Callable[[Path], Metrics] | None = None
    validate_reduction: Callable[[Mapping[str, Any], Mapping[str, Any]], Metrics] | None = None
    make_filterwise_step: Callable[[Any, str, Mapping[str, Any]], Any] | None = None
    save_checkpoint: Callable[[Any, Path], Path] | None = None

    @classmethod
    def defaults(cls, config: Mapping[str, Any]) -> "ClusterEvaluationAdapters":
        """Build real adapters only after a non-dry workflow is requested."""

        return cls(
            load_model=_load_yolo,
            evaluate=_evaluate_yolo,
            safe_layers=_safe_layers,
            make_probe=lambda model, layer, size, ratio: _structural_probe(model, layer, size, ratio, config),
            make_global=lambda model, size, ratio: _structural_global(model, size, ratio, config),
            fine_tune=_fine_tune_yolo,
            reload_model=_load_yolo,
            export=_export_yolo,
            profile=_profile_export,
            stats=_collect_stats,
            artifact_stats=_artifact_stats,
            validate_reduction=_validate_structural_reduction,
            make_filterwise_step=lambda model, layer, config: _filterwise_step(model, layer, config),
            save_checkpoint=_save_filterwise_checkpoint,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_id(stage: str, *, layer: str | None = None, cluster_size: int | None = None, ratio: float | None = None) -> str:
    parts = [stage]
    if layer:
        parts.append(re.sub(r"[^A-Za-z0-9]+", "-", layer).strip("-"))
    if cluster_size is not None:
        parts.append(f"cluster-{cluster_size}")
    if ratio is not None:
        parts.append(f"ratio-{ratio:g}")
    return "-".join(parts)


def _filterwise_candidate_id(layer: str, filters_removed: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", layer).strip("-")
    return f"filterwise-{slug}-filters-{filters_removed}"


def _base_row(
    candidate_id: str,
    stage: str,
    cluster_size: int | None = None,
    ratio: float | None = None,
    layer: str | None = None,
) -> Metrics:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "layer": layer,
        "cluster_size": cluster_size,
        "prune_ratio": ratio,
        "status": "planned",
        "error": None,
        "checkpoint_path": None,
        "exported_path": None,
        "screening_device": None,
        "target_device": None,
        "evaluation_device": None,
        "profile_device": None,
        "hardware_benchmarked": False,
        "benchmark_path": None,
        "benchmark_provenance": None,
        "export_validation_status": "not_run",
        "reason": None,
        "filters_before": None,
        "filters_removed": None,
        "filters_after": None,
        "filter_reduction": None,
    }


def _append_row(rows: list[Metrics], row: Metrics) -> None:
    """Append a row while preserving a unique identifier in every manifest."""

    candidate_id = str(row["candidate_id"])
    existing_ids = {str(existing["candidate_id"]) for existing in rows}
    if candidate_id in existing_ids:
        suffix = 2
        while f"{candidate_id}-{suffix}" in existing_ids:
            suffix += 1
        row["candidate_id"] = f"{candidate_id}-{suffix}"
    rows.append(row)


def _write_artifacts(output_dir: Path, config_path: Path, rows: list[Metrics]) -> None:
    winners = select_cluster_candidates(
        row
        for row in rows
        if row.get("stage") == "global"
        and row.get("hardware_benchmarked") is True
        and row.get("export_validation_status") == "passed"
        and row.get("status") in _FEASIBLE_STATUSES
    )
    write_metrics_csv(output_dir / "candidates.csv", rows)
    write_experiment_manifest(
        output_dir / "manifest.json",
        {
            "config_path": str(config_path),
            "candidate_count": len(rows),
            "candidate_ids": [row["candidate_id"] for row in rows],
            "selected_candidate_ids": {
                name: winner["candidate_id"] if winner is not None else None
                for name, winner in winners.items()
            },
            "benchmarks": [
                {
                    "candidate_id": row["candidate_id"],
                    "benchmark_path": row["benchmark_path"],
                    "benchmark_provenance": row["benchmark_provenance"],
                }
                for row in rows
                if row.get("benchmark_path") is not None
            ],
            "rows": rows,
        },
    )


def merge_jetson_metrics(rows: list[Metrics], benchmark_path: Path) -> list[Metrics]:
    """Merge authoritative native Jetson metrics and refresh experiment artifacts."""

    path = Path(benchmark_path)
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    candidate_id = benchmark.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Native Jetson benchmark JSON must include a non-empty candidate_id.")
    supplied_provenance = {
        key: benchmark[key]
        for key in ("device", "target")
        if key in benchmark
    }
    if not supplied_provenance or any(value != _ORIN_TARGET for value in supplied_provenance.values()):
        raise ValueError("Native Jetson benchmark JSON must identify device or target as jetson_orin_nano.")
    required_latencies = ("latency_p50_ms", "latency_p95_ms")
    if any(
        isinstance(benchmark.get(field), bool)
        or not isinstance(benchmark.get(field), (int, float))
        or not math.isfinite(float(benchmark[field]))
        or float(benchmark[field]) <= 0.0
        for field in required_latencies
    ):
        raise ValueError(
            "Native Jetson benchmark JSON must include positive finite numeric latency_p50_ms and latency_p95_ms "
            "measurements."
        )

    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one candidate row for {candidate_id!r}, found {len(matches)}.")

    row = matches[0]
    for field in _JETSON_HARDWARE_FIELDS:
        if field in benchmark:
            row[field] = benchmark[field]
    row["benchmark_path"] = str(path.resolve())
    row["benchmark_provenance"] = {"candidate_id": candidate_id, **supplied_provenance}
    row["hardware_benchmarked"] = True
    row["profile_device"] = _ORIN_TARGET

    if row.get("stage") == "global" and row.get("map50_95") is not None:
        baseline = next(
            (
                candidate
                for candidate in rows
                if candidate.get("stage") == "baseline" and candidate.get("map50_95") is not None
            ),
            None,
        )
        if baseline is not None:
            row["status"] = classify_candidate(float(baseline["map50_95"]), float(row["map50_95"]))

    output_dir = path.parent
    config_path = output_dir / "cluster-workflow.yaml"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_path = Path(manifest.get("config_path", config_path))
    _write_artifacts(output_dir, config_path, rows)
    return rows


def _resolve_config(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = load_experiment_config(config_path)
    repo_root = _repo_root()
    checkpoint = resolve_repo_path(config["model"]["checkpoint"], repo_root)
    output_dir = resolve_repo_path(config["experiment"]["output_dir"], repo_root)
    return config, checkpoint, output_dir, Path(config["_config_path"])


def _planned_rows(config: Mapping[str, Any]) -> list[Metrics]:
    pruning = config["pruning"]
    rows: list[Metrics] = []
    _append_row(rows, _base_row("baseline", "baseline", ratio=0.0))
    for layer in pruning["safe_layers"]:
        for size in pruning["cluster_sizes"]:
            for ratio in pruning["probe_ratios"]:
                _append_row(
                    rows,
                    _base_row(
                        _candidate_id("probe", layer=layer, cluster_size=int(size), ratio=float(ratio)),
                        "probe",
                        int(size),
                        float(ratio),
                        layer,
                    ),
                )
    for size in pruning["cluster_sizes"]:
        for ratio in pruning["global_ratios"]:
            _append_row(rows, _base_row(_candidate_id("global", cluster_size=int(size), ratio=float(ratio)), "global", int(size), float(ratio)))
    return rows


def _planned_filterwise_rows(
    config: Mapping[str, Any], layer_widths: Mapping[str, int] | None = None
) -> list[Metrics]:
    pruning = config["pruning"]
    rows: list[Metrics] = [_base_row("baseline", "baseline", ratio=0.0)]
    configured_layers = pruning["filter_sweep_layers"]
    if configured_layers == "auto":
        if layer_widths is None:
            return rows
        layers = list(layer_widths)
        widths = {str(name): int(width) for name, width in layer_widths.items()}
    else:
        layers = [str(layer) for layer in configured_layers]
        widths = {str(name): int(width) for name, width in pruning.get("filter_sweep_widths", {}).items()}
    for layer in layers:
        if layer not in widths:
            raise ValueError(f"filter_sweep_widths must define the output width for {layer!r}.")
        for filters_removed in range(1, widths[layer]):
            row = _base_row(
                _filterwise_candidate_id(layer, filters_removed),
                "filterwise",
                layer=layer,
            )
            row["filters_before"] = widths[layer]
            row["filters_removed"] = filters_removed
            row["filters_after"] = widths[layer] - filters_removed
            row["filter_reduction"] = filters_removed / widths[layer]
            _append_row(rows, row)
    return rows


def _filterwise_layer_widths(model: Any, layers: Sequence[str]) -> dict[str, int]:
    from torch import nn

    modules = dict(_unwrap_model(model).named_modules())
    widths: dict[str, int] = {}
    for layer in layers:
        module = modules.get(layer)
        if not isinstance(module, nn.Conv2d) or int(module.weight.shape[0]) <= 1:
            raise ValueError(f"Automatic filter-wise layer {layer!r} is not a prunable Conv2d module.")
        widths[layer] = int(module.weight.shape[0])
    return widths


def _screen_probe(row: Mapping[str, Any], baseline_map50_95: float, config: Mapping[str, Any]) -> bool:
    screening = config.get("screening", {})
    max_drop = float(screening.get("max_map50_95_drop", 0.02))
    minimum_reduction = float(screening.get("minimum_serialized_reduction", 0.0))
    candidate_map = float(row["map50_95"])
    reduction = float(row.get("serialized_reduction", 0.0))
    return baseline_map50_95 - candidate_map <= max_drop and reduction >= minimum_reduction


def _metrics_row(row: Metrics, metrics: Mapping[str, Any], stats: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> None:
    row.update(dict(metrics))
    row.update(dict(stats))
    if baseline is not None:
        row["map50_95_drop"] = float(baseline["map50_95"]) - float(row["map50_95"])
        baseline_bytes = baseline.get("serialized_bytes")
        candidate_bytes = row.get("serialized_bytes")
        if baseline_bytes and candidate_bytes is not None:
            row["serialized_reduction"] = 1.0 - float(candidate_bytes) / float(baseline_bytes)


def _failure(row: Metrics, exc: Exception) -> None:
    row["status"] = "failed"
    row["error"] = f"{type(exc).__name__}: {exc}"
    if row.get("stage") == "filterwise" and _is_filterwise_channel_mismatch(row["error"]):
        row["status"] = "skipped"
        row["reason"] = "Skipped because structural channel mismatch: " + row["error"]


def _skip(row: Metrics, reason: str) -> None:
    if row["status"] == "planned":
        row["status"] = "skipped"
        row["reason"] = reason
        row["error"] = reason


def _rows_for_stage(rows: list[Metrics], stage: str) -> list[Metrics]:
    return [row for row in rows if row["stage"] == stage]


def _record_export(row: Metrics, exported: Path, baseline: Mapping[str, Any] | None = None) -> None:
    row["exported_path"] = str(exported)
    artifact_format = exported.suffix.removeprefix(".").lower()
    if not artifact_format:
        raise ValueError("Exported artifact has no file format suffix.")
    row["artifact_format"] = artifact_format
    if exported.exists():
        row["serialized_bytes"] = exported.stat().st_size
    if baseline is not None:
        baseline_bytes = baseline.get("serialized_bytes")
        candidate_bytes = row.get("serialized_bytes")
        if baseline_bytes and candidate_bytes is not None:
            row["serialized_reduction"] = 1.0 - float(candidate_bytes) / float(baseline_bytes)


def _artifact_stats(path: Path) -> Metrics:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"serialized_bytes": path.stat().st_size}


def _validate_structural_reduction(before: Mapping[str, Any], after: Mapping[str, Any]) -> Metrics:
    from infrared_detection.compression.pruning import validate_structural_reduction

    reduction = validate_structural_reduction(before, after)
    if float(after["serialized_bytes"]) >= float(before["serialized_bytes"]):
        raise ValueError("Serialized artifact did not reduce in the comparable export format.")
    return reduction


def _validate_filterwise_reduction(before: Mapping[str, Any], after: Mapping[str, Any]) -> Metrics:
    """Validate parameter reduction without requiring the engine file to shrink."""

    before_params = float(before["parameter_count"])
    after_params = float(after["parameter_count"])
    if after_params >= before_params:
        raise ValueError("Structured pruning did not reduce parameter count.")
    result = {"parameter_reduction": 1.0 - after_params / before_params}
    if "serialized_bytes" in before and "serialized_bytes" in after:
        before_bytes = float(before["serialized_bytes"])
        after_bytes = float(after["serialized_bytes"])
        result["serialized_reduction"] = 1.0 - after_bytes / before_bytes
    return result


def _validate_candidate_reduction(
    row: Metrics,
    model_stats: Mapping[str, Any],
    exported: Path,
    baseline: Mapping[str, Any],
    active: ClusterEvaluationAdapters,
) -> None:
    row["export_validation_status"] = "failed"
    if row["artifact_format"] != baseline["artifact_format"]:
        raise ValueError(
            "Candidate artifact format does not match the baseline export format "
            f"({row['artifact_format']} != {baseline['artifact_format']})."
        )
    artifact_stats = active.artifact_stats or _artifact_stats
    validator = active.validate_reduction or _validate_structural_reduction
    after = {**dict(model_stats), **dict(artifact_stats(exported))}
    before = {
        "parameter_count": baseline["parameter_count"],
        "serialized_bytes": baseline["serialized_bytes"],
    }
    row.update(validator(before, after))
    row["serialized_bytes"] = after["serialized_bytes"]
    row["export_validation_status"] = "passed"


def run_cluster_evaluation(
    config_path: Path,
    dry_run: bool = False,
    adapters: ClusterEvaluationAdapters | None = None,
    screen_only: bool = False,
) -> list[Metrics]:
    """Run (or plan) baseline, probe, and global cluster-pruning evaluation.

    Dry runs use only the YAML-declared safe layers and candidate sizes, so no
    external inference, checkpoint, CUDA, TensorRT, or data access occurs.
    """

    config, checkpoint, output_dir, resolved_config_path = _resolve_config(Path(config_path))
    if dry_run:
        rows = _planned_rows(config)
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    targets = config["targets"]
    orin_target = str(targets["orin_target"])
    if orin_target != _ORIN_TARGET:
        raise ValueError(f"targets.orin_target must be {_ORIN_TARGET!r}, got {orin_target!r}.")
    if adapters is None and not screen_only and not _is_jetson_orin_runtime():
        raise RuntimeError(
            "Refusing to claim Jetson Orin Nano results on an unverified host. "
            "Run the default workflow on the Jetson Orin Nano or inject a remote Orin adapter."
        )

    active = adapters or ClusterEvaluationAdapters.defaults(config)
    pruning = config["pruning"]
    rtx_device = str(targets["rtx_screening_device"])
    orin_execution_device = _orin_execution_device(config)
    baseline_execution_device = rtx_device if screen_only else orin_execution_device
    rows = _planned_rows(config)
    baseline = next(row for row in rows if row["stage"] == "baseline")
    try:
        baseline_model = active.load_model(checkpoint)
        baseline_metrics = active.evaluate(baseline_model, config, baseline_execution_device)
        baseline_model_stats = active.stats(baseline_model)
        baseline_export = Path(active.export(checkpoint, config, output_dir / "baseline"))
        _record_export(baseline, baseline_export)
        baseline_artifact_stats = (active.artifact_stats or _artifact_stats)(baseline_export)
        _metrics_row(
            baseline,
            baseline_metrics,
            {**dict(baseline_model_stats), **dict(baseline_artifact_stats)},
        )
        baseline["status"] = "completed"
        baseline["target_device"] = None if screen_only else orin_target
        baseline["evaluation_device"] = baseline_execution_device
        if screen_only:
            baseline["screening_device"] = rtx_device
    except Exception as exc:
        _failure(baseline, exc)
        for row in rows[1:]:
            _skip(row, f"Skipped because baseline failed: {baseline['error']}")
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    try:
        configured_layers = list(pruning["safe_layers"])
        layers = [layer for layer in active.safe_layers(baseline_model, config) if layer in configured_layers]
    except Exception as exc:
        reason = f"Skipped because safe-layer screening failed: {type(exc).__name__}: {exc}"
        for row in rows[1:]:
            _skip(row, reason)
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    surviving_sizes: set[int] = set()
    for row in _rows_for_stage(rows, "probe"):
        layer = str(row["layer"])
        if layer not in layers:
            _skip(row, f"Skipped because safe-layer screening excluded {layer}.")
            continue
        size = int(row["cluster_size"])
        probe_ratio = float(row["prune_ratio"])
        try:
            probe_model = active.load_model(checkpoint)
            pruned_model = active.make_probe(probe_model, layer, size, probe_ratio)
            probe_metrics = active.evaluate(pruned_model, config, rtx_device)
            probe_model_stats = active.stats(pruned_model)
            exported = Path(active.export(pruned_model, config, output_dir / row["candidate_id"]))
            _record_export(row, exported, baseline)
            _validate_candidate_reduction(row, probe_model_stats, exported, baseline, active)
            _metrics_row(row, probe_metrics, {**dict(probe_model_stats), "serialized_bytes": row["serialized_bytes"]}, baseline)
            row.update(active.profile(exported, rtx_device))
            row["screening_device"] = rtx_device
            row["evaluation_device"] = rtx_device
            row["profile_device"] = rtx_device
            row["status"] = "screened_in" if _screen_probe(row, float(baseline["map50_95"]), config) else "screened_out"
            if row["status"] == "screened_in":
                surviving_sizes.add(size)
        except Exception as exc:
            _failure(row, exc)

    if screen_only:
        for row in _rows_for_stage(rows, "global"):
            _skip(row, "screen-only mode does not run global candidates")
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    for row in _rows_for_stage(rows, "global"):
        if int(row["cluster_size"]) not in surviving_sizes:
            _skip(row, f"Skipped because no probe survived RTX screening for cluster size {row['cluster_size']}.")

    for row in _rows_for_stage(rows, "global"):
        if row["status"] != "planned":
            continue
        size = int(row["cluster_size"])
        ratio = float(row["prune_ratio"])
        try:
            candidate = active.make_global(active.load_model(checkpoint), size, ratio)
            candidate_dir = output_dir / row["candidate_id"]
            best_checkpoint = active.fine_tune(candidate, config, candidate_dir)
            reloaded = active.reload_model(Path(best_checkpoint))
            global_metrics = active.evaluate(reloaded, config, orin_execution_device)
            global_model_stats = active.stats(reloaded)
            exported = Path(active.export(Path(best_checkpoint), config, candidate_dir))
            row["checkpoint_path"] = str(Path(best_checkpoint))
            _record_export(row, exported, baseline)
            _validate_candidate_reduction(row, global_model_stats, exported, baseline, active)
            _metrics_row(row, global_metrics, {**dict(global_model_stats), "serialized_bytes": row["serialized_bytes"]}, baseline)
            row["target_device"] = orin_target
            row.update(active.profile(exported, orin_target))
            row["evaluation_device"] = orin_execution_device
            row["profile_device"] = orin_target
            row["status"] = classify_candidate(float(baseline["map50_95"]), float(row["map50_95"]))
        except Exception as exc:
            _failure(row, exc)

    _write_artifacts(output_dir, resolved_config_path, rows)
    return rows


def run_filterwise_evaluation(
    config_path: Path,
    dry_run: bool = False,
    adapters: ClusterEvaluationAdapters | None = None,
    full_curve: bool = False,
) -> list[Metrics]:
    """Sequentially measure accuracy and selected latency points per layer."""

    config, checkpoint, output_dir, resolved_config_path = _resolve_config(Path(config_path))
    if dry_run:
        rows = _planned_filterwise_rows(config)
        rows = _load_filterwise_checkpoint(output_dir, resolved_config_path, rows)
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    active = adapters or ClusterEvaluationAdapters.defaults(config)
    targets = config["targets"]
    rtx_device = str(targets["rtx_screening_device"])
    automatic_layers = config["pruning"]["filter_sweep_layers"] == "auto"
    rows = _planned_filterwise_rows(config)
    if not automatic_layers:
        rows = _load_filterwise_checkpoint(output_dir, resolved_config_path, rows)
    baseline = rows[0]

    if not automatic_layers and _row_is_resumable(baseline) and all(
        _row_is_resumable(row) for row in _rows_for_stage(rows, "filterwise")
    ):
        return rows

    try:
        baseline_model = active.load_model(checkpoint)
        available_layer_order = [str(layer) for layer in active.safe_layers(baseline_model, config)]
        if automatic_layers:
            rows = _planned_filterwise_rows(config, _filterwise_layer_widths(baseline_model, available_layer_order))
            rows = _load_filterwise_checkpoint(output_dir, resolved_config_path, rows)
            baseline = rows[0]
            if _row_is_resumable(baseline) and all(
                _row_is_resumable(row) for row in _rows_for_stage(rows, "filterwise")
            ):
                return rows
        if not _row_is_resumable(baseline):
            baseline_metrics = active.evaluate(baseline_model, config, rtx_device)
            baseline_model_stats = active.stats(baseline_model)
            baseline_export = Path(active.export(checkpoint, config, output_dir / "baseline"))
            _record_export(baseline, baseline_export)
            baseline_artifact_stats = (active.artifact_stats or _artifact_stats)(baseline_export)
            _metrics_row(baseline, baseline_metrics, {**dict(baseline_model_stats), **dict(baseline_artifact_stats)})
            baseline["status"] = "completed"
            baseline["screening_device"] = rtx_device
            baseline["evaluation_device"] = rtx_device
    except Exception as exc:
        _failure(baseline, exc)
        for row in rows[1:]:
            _skip(row, f"Skipped because baseline failed: {baseline['error']}")
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows
    _write_artifacts(output_dir, resolved_config_path, rows)

    configured_layer_order = (
        available_layer_order if automatic_layers else [str(layer) for layer in config["pruning"]["filter_sweep_layers"]]
    )
    configured_layers = set(configured_layer_order)
    available_layers = set(available_layer_order)

    screening = config.get("screening", {})
    step_factory = active.make_filterwise_step or (lambda model, layer, cfg: _filterwise_step(model, layer, cfg))
    checkpoint_writer = active.save_checkpoint or _save_filterwise_checkpoint
    for layer in configured_layer_order:
        layer_rows = sorted(
            (row for row in _rows_for_stage(rows, "filterwise") if row.get("layer") == layer),
            key=lambda row: int(row["filters_removed"]),
        )
        if layer not in configured_layers or layer not in available_layers:
            for row in layer_rows:
                _skip(row, f"Skipped because filter-wise layer screening excluded {layer}.")
            _write_artifacts(output_dir, resolved_config_path, rows)
            continue
        recorded = [_row_is_recorded(row) for row in layer_rows]
        if all(recorded):
            continue
        checkpointed = [
            (index, row)
            for index, row in enumerate(layer_rows)
            if _row_is_resumable(row)
        ]
        if checkpointed:
            checkpoint_index, checkpoint_row = checkpointed[-1]
            current_model = active.load_model(Path(checkpoint_row["checkpoint_path"]))
            pending_rows = [
                row for row in layer_rows[checkpoint_index + 1:]
                if not _row_is_recorded(row)
            ]
        else:
            current_model = active.load_model(checkpoint)
            pending_rows = [row for row in layer_rows if not _row_is_recorded(row)]
        if not pending_rows:
            continue
        consecutive_near_zero = 0
        for row in pending_rows:
            try:
                current_model = step_factory(current_model, layer, config)
                checkpoint_path = Path(checkpoint_writer(current_model, output_dir / row["candidate_id"]))
                row["checkpoint_path"] = str(checkpoint_path)
                if checkpoint_path.exists():
                    row["checkpoint_bytes"] = checkpoint_path.stat().st_size
                # Ultralytics validation may fuse modules and create inference-mode
                # tensors in-place. Keep the sequentially prunable model separate
                # so the next minimum-weight step still has a trainable model.
                evaluation_model = active.load_model(checkpoint_path)
                metrics = active.evaluate(evaluation_model, config, rtx_device)
                model_stats = dict(active.stats(current_model))
                _metrics_row(row, metrics, model_stats, baseline)
                row["screening_device"] = rtx_device
                row["evaluation_device"] = rtx_device
                row["status"] = (
                    "screened_in"
                    if float(row["map50_95"]) >= float(baseline["map50_95"])
                    - float(screening.get("max_map50_95_drop", 0.02))
                    else "screened_out"
                )
                if _should_profile_filterwise(row, config):
                    exported = Path(active.export(checkpoint_path, config, output_dir / row["candidate_id"]))
                    _record_export(row, exported, baseline)
                    artifact_stats = active.artifact_stats or _artifact_stats
                    after = {**model_stats, **dict(artifact_stats(exported))}
                    row.update(
                        _validate_filterwise_reduction(
                            {"parameter_count": baseline["parameter_count"]},
                            after,
                        )
                    )
                    row["export_validation_status"] = "passed"
                    row.update(active.profile(exported, rtx_device))
                    row["profile_device"] = rtx_device
                    row["hardware_benchmarked"] = True
                else:
                    row["export_validation_status"] = "not_required"
                if float(row["map50_95"]) < float(screening.get("early_stop_map50_95", 0.0)):
                    consecutive_near_zero += 1
                else:
                    consecutive_near_zero = 0
            except Exception as exc:
                _failure(row, exc)
                consecutive_near_zero = 0
            finally:
                _write_artifacts(output_dir, resolved_config_path, rows)
            if _filterwise_stop_reached(consecutive_near_zero, config, full_curve):
                for remaining in pending_rows[pending_rows.index(row) + 1:]:
                    _skip(remaining, "Skipped because early stopping detected near-zero accuracy.")
                _write_artifacts(output_dir, resolved_config_path, rows)
                break

    return rows


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _load_yolo(checkpoint: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(checkpoint))


def _evaluate_yolo(model: Any, config: Mapping[str, Any], device: str) -> Metrics:
    from infrared_detection.evaluation.detection_metrics import evaluate_yolo

    runtime = config["runtime"]
    return evaluate_yolo(
        model,
        str(resolve_repo_path(config["data"]["dataset_yaml"], _repo_root())),
        "val",
        int(config["experiment"]["image_size"]),
        device,
        float(runtime.get("conf", 0.25)),
        float(runtime.get("iou", 0.6)),
    )


def _safe_layers(model: Any, config: Mapping[str, Any]) -> list[str]:
    from torch import nn

    unwrapped = _unwrap_model(model)
    modules = dict(unwrapped.named_modules())
    configured = config["pruning"].get("safe_layers")
    if configured:
        protected = set(config["pruning"]["protected_layers"])
        safe = []
        for layer in configured:
            if any(layer == name or layer.startswith(f"{name}.") for name in protected) or layer.startswith("model.22"):
                raise ValueError(f"Configured safe layer {layer!r} is protected and cannot be pruned.")
            module = modules.get(layer)
            if module is None:
                raise ValueError(f"Configured safe layer {layer!r} does not exist in the model.")
            if not isinstance(module, nn.Conv2d):
                raise ValueError(
                    f"Configured safe layer {layer!r} is {type(module).__name__}, not a prunable Conv2d module."
                )
            safe.append(layer)
        return safe
    from infrared_detection.compression.pruning import compute_channel_importance

    protected = set(config["pruning"]["protected_layers"])
    return [
        name
        for name in sorted(compute_channel_importance(unwrapped))
        if isinstance(modules[name], nn.Conv2d)
        and not any(name == protected_name or name.startswith(f"{protected_name}.") for protected_name in protected)
        and not name.startswith("model.22")
    ]


def _structural_probe(model: Any, layer: str, cluster_size: int, ratio: float, config: Mapping[str, Any]) -> Any:
    import torch
    from torch import nn

    from infrared_detection.compression.pruning import compute_channel_importance, plan_low_importance_clusters
    from infrared_detection.compression.pruning.cluster_probe import run_structural_probe as prune

    if cluster_size <= 0:
        raise ValueError("Cluster size must be positive.")
    unwrapped = _unwrap_model(model)
    module = dict(unwrapped.named_modules()).get(layer)
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f"Structural probe target {layer!r} is not a prunable Conv2d module.")
    width = int(module.weight.shape[0])
    requested_clusters = max(1, int(width * ratio) // cluster_size)
    available_clusters = (width - 1) // cluster_size
    if available_clusters < 1:
        raise ValueError(
            f"Layer {layer!r} has no complete cluster of size {cluster_size} that can be pruned safely."
        )
    requested_clusters = min(requested_clusters, available_clusters)
    protected = set(config["pruning"]["protected_layers"])
    image_size = int(config["experiment"]["image_size"])
    aligned_image_size = ((image_size + 31) // 32) * 32
    example = torch.zeros(1, 3, aligned_image_size, aligned_image_size)

    for _ in range(requested_clusters):
        scores = compute_channel_importance(unwrapped)
        if layer not in scores:
            raise ValueError(f"No channel-importance scores are available for prunable layer {layer!r}.")
        specs = plan_low_importance_clusters(
            {layer: scores[layer]},
            cluster_size,
            protected,
            max_clusters_per_layer=1,
        )
        if len(specs) != 1 or len(specs[0].prune_indices) != cluster_size:
            raise ValueError(f"Could not plan one complete low-importance cluster for layer {layer!r}.")
        unwrapped = prune(unwrapped, example, specs[0])

    if hasattr(model, "model"):
        model.model = unwrapped
        return model
    return unwrapped


def _filterwise_probe(model: Any, layer: str, filters_removed: int, config: Mapping[str, Any]) -> Any:
    import torch
    from torch import nn

    from infrared_detection.compression.pruning import compute_channel_importance
    from infrared_detection.compression.pruning.cluster_probe import run_filterwise_probe as prune

    unwrapped = _unwrap_model(model)
    module = dict(unwrapped.named_modules()).get(layer)
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f"Filter-wise probe target {layer!r} is not a prunable Conv2d module.")
    if filters_removed <= 0 or filters_removed >= int(module.weight.shape[0]):
        raise ValueError(f"Filter-wise probe for {layer!r} must remove between 1 and output_channels - 1 filters.")
    scores = compute_channel_importance(unwrapped)
    if layer not in scores:
        raise ValueError(f"No channel-importance scores are available for prunable layer {layer!r}.")
    ranked = torch.argsort(scores[layer]).tolist()
    indices = tuple(int(index) for index in ranked[:filters_removed])
    image_size = int(config["experiment"]["image_size"])
    aligned_image_size = ((image_size + 31) // 32) * 32
    pruned = prune(
        unwrapped,
        torch.zeros(1, 3, aligned_image_size, aligned_image_size),
        layer,
        indices,
    )
    if hasattr(model, "model"):
        model.model = pruned
        return model
    return pruned


def _row_is_resumable(row: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint row contains enough data to skip rerunning it."""

    common = (
        row.get("status") in _RESUMABLE_FILTERWISE_STATUSES
        and row.get("map50_95") is not None
        and row.get("parameter_count") is not None
    )
    if row.get("stage") == "baseline":
        exported_path = row.get("exported_path")
        return bool(
            common
            and row.get("serialized_bytes") is not None
            and exported_path
            and Path(str(exported_path)).is_file()
        )
    checkpoint_path = row.get("checkpoint_path")
    return bool(common and checkpoint_path and Path(str(checkpoint_path)).is_file())


def _row_is_recorded(row: Mapping[str, Any]) -> bool:
    """Return whether a candidate result is final even if its checkpoint was cleaned up."""

    if row.get("stage") != "filterwise":
        return _row_is_resumable(row)
    return bool(
        row.get("status") in _RESUMABLE_FILTERWISE_STATUSES
        and row.get("map50_95") is not None
        and row.get("parameter_count") is not None
    ) or _row_is_terminal_exclusion(row)


def _should_profile_filterwise(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    points = config.get("screening", {}).get("latency_profile_removals")
    if points is None:
        return True
    return int(row["filters_removed"]) in {int(point) for point in points}


def _filterwise_stop_reached(consecutive_near_zero: int, config: Mapping[str, Any], full_curve: bool) -> bool:
    screening = config.get("screening", {})
    if full_curve or not bool(screening.get("early_stop", False)):
        return False
    required = max(1, int(screening.get("early_stop_consecutive", 1)))
    return consecutive_near_zero >= required


def _load_filterwise_checkpoint(output_dir: Path, config_path: Path, rows: list[Metrics]) -> list[Metrics]:
    """Merge completed rows from a prior filter-wise run into the current plan."""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return rows
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return rows

    planned_ids = [str(row["candidate_id"]) for row in rows]
    if manifest.get("candidate_ids") != planned_ids:
        return rows
    saved_config = manifest.get("config_path")
    if saved_config and Path(saved_config).resolve() != Path(config_path).resolve():
        return rows

    saved_rows = {
        str(row.get("candidate_id")): row
        for row in manifest.get("rows", [])
        if isinstance(row, dict)
    }
    for row in rows:
        saved = saved_rows.get(str(row["candidate_id"]))
        if saved is not None:
            saved = dict(saved)
            if saved.get("stage") == "filterwise" and saved.get("status") == "failed" and _is_filterwise_channel_mismatch(saved.get("error")):
                saved["status"] = "skipped"
                saved["reason"] = "Skipped because structural channel mismatch: " + str(saved.get("error"))
            if _row_is_recorded(saved) or saved.get("status") == "failed":
                row.clear()
                row.update(saved)
    return rows


def _row_is_terminal_exclusion(row: Mapping[str, Any]) -> bool:
    """Return whether a saved row was intentionally excluded from evaluation."""

    return row.get("status") in _TERMINAL_FILTERWISE_STATUSES


def _is_filterwise_channel_mismatch(error: str | None) -> bool:
    """Return whether an error describes an invalid propagated channel topology."""

    return bool(error and _FILTERWISE_CHANNEL_MISMATCH_RE.search(str(error)))


def _filterwise_step(model: Any, layer: str, config: Mapping[str, Any]) -> Any:
    """Remove exactly one lowest-Minimum-Weight filter from the current model."""

    import torch
    from torch import nn

    from infrared_detection.compression.pruning import compute_channel_importance
    from infrared_detection.compression.pruning.cluster_probe import run_filterwise_probe as prune

    unwrapped = _unwrap_model(model)
    module = dict(unwrapped.named_modules()).get(layer)
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f"Filter-wise step target {layer!r} is not a prunable Conv2d module.")
    if int(module.weight.shape[0]) <= 1:
        raise ValueError(f"Filter-wise step for {layer!r} cannot remove the final output filter.")
    scores = compute_channel_importance(unwrapped)
    if layer not in scores:
        raise ValueError(f"No channel-importance scores are available for prunable layer {layer!r}.")
    prune_index = (int(torch.argsort(scores[layer])[0]),)
    image_size = int(config["experiment"]["image_size"])
    aligned_image_size = ((image_size + 31) // 32) * 32
    pruned = prune(
        unwrapped,
        torch.zeros(1, 3, aligned_image_size, aligned_image_size),
        layer,
        prune_index,
    )
    if hasattr(model, "model"):
        model.model = pruned
        return model
    return pruned


def _save_filterwise_checkpoint(model: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "candidate.pt"
    save = getattr(model, "save", None)
    if not callable(save):
        raise RuntimeError("Sequential filter-wise pruning requires a save-capable model wrapper.")
    save(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pruned checkpoint was not written: {checkpoint}")
    return checkpoint


def _structural_global(model: Any, cluster_size: int, ratio: float, config: Mapping[str, Any]) -> Any:
    for layer in _safe_layers(model, config):
        model = _structural_probe(model, layer, cluster_size, ratio, config)
    return model


def _topology_preserving_trainer(pruned_model: Any, trainer_cls: type | None = None) -> type:
    """Return a guarded DetectionTrainer that reuses the physically pruned module."""

    if trainer_cls is None:
        import ultralytics
        from ultralytics.models.yolo.detect import DetectionTrainer

        if ultralytics.__version__ != "8.4.7":
            raise RuntimeError(
                "Topology-preserving training is validated only for pinned Ultralytics 8.4.7; "
                f"found {ultralytics.__version__}."
            )
        trainer_cls = DetectionTrainer

    parameters = tuple(inspect.signature(trainer_cls.get_model).parameters)
    if parameters[:4] != ("self", "cfg", "weights", "verbose"):
        raise RuntimeError(
            "Ultralytics DetectionTrainer.get_model is incompatible with the topology-preserving training path."
        )

    class TopologyPreservingDetectionTrainer(trainer_cls):
        def get_model(self, cfg=None, weights=None, verbose=True):
            del cfg, verbose
            if weights is not pruned_model:
                raise RuntimeError(
                    "Ultralytics did not pass the physically pruned module to the topology-preserving trainer."
                )
            return pruned_model

    TopologyPreservingDetectionTrainer.__name__ = "TopologyPreservingDetectionTrainer"
    return TopologyPreservingDetectionTrainer


def _orin_execution_device(config: Mapping[str, Any]) -> str:
    device = str(config["targets"].get("orin_execution_device", "0"))
    if not device or "orin" in device.lower() or "jetson" in device.lower():
        raise ValueError(
            "targets.orin_execution_device must be a valid Ultralytics execution device such as '0' or 'cpu', "
            f"not {device!r}."
        )
    return device


def _fine_tune_yolo(model: Any, config: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "trainer" not in inspect.signature(model.train).parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(model.train).parameters.values()
    ):
        raise RuntimeError("Ultralytics Model.train cannot accept the topology-preserving trainer override.")
    trainer_type = _topology_preserving_trainer(_unwrap_model(model))
    result = model.train(
        trainer=trainer_type,
        data=str(resolve_repo_path(config["data"]["dataset_yaml"], _repo_root())),
        epochs=int(config["pruning"]["fine_tune_epochs"]),
        imgsz=int(config["experiment"]["image_size"]),
        device=_orin_execution_device(config),
        seed=int(config["experiment"]["seed"]),
        project=str(output_dir),
        name="fine_tune",
    )
    best = getattr(getattr(model, "trainer", None), "best", None)
    if best is None:
        raise RuntimeError("Fine-tuning did not produce a best checkpoint.")
    del result
    return Path(best)


def _export_yolo(checkpoint: Any, config: Mapping[str, Any], output_dir: Path) -> Path:
    from infrared_detection.export import export_yolo

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": config.get("export", {}).get("format", "onnx"), "imgsz": int(config["experiment"]["image_size"])}
    staged_checkpoint = output_dir / "export-source.pt"
    if isinstance(checkpoint, (str, Path)):
        shutil.copy2(Path(checkpoint), staged_checkpoint)
    else:
        save = getattr(checkpoint, "save", None)
        if not callable(save):
            raise RuntimeError(
                "Isolated export requires a checkpoint path or a save-capable pruned YOLO model."
            )
        save(staged_checkpoint)

    exported = Path(export_yolo(staged_checkpoint, **kwargs))
    if not exported.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce the reported artifact: {exported}")
    isolated = output_dir / f"candidate{exported.suffix.lower()}"
    if exported.resolve() != isolated.resolve():
        shutil.copy2(exported, isolated)
    return isolated


def _is_jetson_orin_runtime() -> bool:
    """Return whether this process can verify it is running on a Jetson Orin."""

    try:
        model = Path("/sys/firmware/devicetree/base/model").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "jetson" in model and "orin" in model


def _profile_export(exported: Path, device: str) -> Metrics:
    if "orin" in device.lower() and not _is_jetson_orin_runtime():
        raise RuntimeError(
            "Refusing to label a local TensorRT profile as Orin without a verified Jetson Orin runtime. "
            "Inject a remote Orin profiler or run this workflow on the target."
        )
    from infrared_detection.benchmarking.jetson import benchmark_tensorrt_engine

    return benchmark_tensorrt_engine(exported)


def _collect_stats(model: Any) -> Metrics:
    from infrared_detection.evaluation.model_stats import collect_model_stats

    return collect_model_stats(_unwrap_model(model))
