"""Planning and artifact writing for RTX compression experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from infrared_detection.compression.pruning import compute_channel_importance
from infrared_detection.compression.pruning.cluster_selection import plan_low_importance_clusters
from infrared_detection.compression.pruning.cluster_probe import run_filterwise_probe
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


OFFICIAL_PRECISIONS = ("fp32", "fp16", "int8")
Metrics = dict[str, Any]


def _validate_prune_ratios(value: Any, context: str) -> list[float]:
    if not isinstance(value, list) or not value or not all(
        isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and 0 < ratio < 1
        for ratio in value
    ):
        raise ValueError(f"{context} requires non-empty pruning.prune_ratios between 0 and 1")
    return [float(ratio) for ratio in value]


def _validate_allowed_map_drop(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} requires non-negative pruning.allowed_map50_95_drop")
    return float(value)


def select_filterwise_candidate(
    manifest_path: Path, layer: str, filters_removed: int
) -> Mapping[str, Any]:
    """Return the explicit screened-in filterwise candidate for a prune build."""

    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Filterwise manifest is not readable: {manifest_path}") from exc

    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("Filterwise manifest has no rows.")

    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("layer") == layer
        and row.get("filters_removed") == filters_removed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Filterwise candidate for {layer!r} removing {filters_removed} filters is missing or ambiguous."
        )

    candidate = matches[0]
    if candidate.get("status") != "screened_in" or not isinstance(
        candidate.get("checkpoint_path"), str
    ) or not candidate["checkpoint_path"]:
        raise ValueError("Selected filterwise candidate is not usable.")
    return candidate


def _load_yolo_checkpoint(checkpoint_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(checkpoint_path))


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _resolve_candidate_checkpoint(manifest_path: Path, checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    return path if path.is_absolute() else manifest_path.parent / path


def build_pruned_checkpoint(
    config: Mapping[str, Any], output_dir: Path, requested_ratio: float | None = None
) -> Path:
    """Build, reload, and record a dense-source structured-pruned checkpoint."""

    import torch
    from torch import nn

    pruning = config.get("pruning")
    if not isinstance(pruning, Mapping):
        raise ValueError("pruning must be a mapping")
    manifest_value = pruning.get("filterwise_manifest")
    evidence_layer = pruning.get("evidence_layer")
    filters_removed = pruning.get("evidence_filters_removed")
    candidate_layers = pruning.get("candidate_layers")
    cluster_size = pruning.get("cluster_size")
    prune_ratios = _validate_prune_ratios(pruning.get("prune_ratios"), "Pruned checkpoint build")
    allowed_map50_95_drop = _validate_allowed_map_drop(
        pruning.get("allowed_map50_95_drop"), "Pruned checkpoint build"
    )
    importance = pruning.get("importance")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("Pruned checkpoint build requires pruning.filterwise_manifest")
    if not isinstance(evidence_layer, str) or not evidence_layer:
        raise ValueError("Pruned checkpoint build requires pruning.evidence_layer")
    if not isinstance(filters_removed, int) or isinstance(filters_removed, bool) or filters_removed <= 0:
        raise ValueError("Pruned checkpoint build requires a positive pruning.evidence_filters_removed")
    if not isinstance(candidate_layers, list) or not candidate_layers or not all(
        isinstance(layer, str) and layer for layer in candidate_layers
    ):
        raise ValueError("Pruned checkpoint build requires non-empty pruning.candidate_layers")
    if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
        raise ValueError("Pruned checkpoint build requires a positive pruning.cluster_size")
    if importance != "minimum_weight":
        raise ValueError("Pruned checkpoint build requires pruning.importance='minimum_weight'")
    if filters_removed != cluster_size:
        raise ValueError("Filterwise evidence_filters_removed must equal pruning.cluster_size")
    if requested_ratio is None:
        if len(prune_ratios) != 1:
            raise ValueError("Pruned checkpoint build requires an explicit requested_ratio from pruning.prune_ratios")
        selected_ratio = prune_ratios[0]
    elif (
        not isinstance(requested_ratio, (int, float))
        or isinstance(requested_ratio, bool)
        or float(requested_ratio) not in prune_ratios
    ):
        raise ValueError("requested_ratio must be one of pruning.prune_ratios")
    else:
        selected_ratio = float(requested_ratio)

    manifest_path = _resolve_repo_path(manifest_value)
    candidate = select_filterwise_candidate(manifest_path, evidence_layer, filters_removed)
    evidence_checkpoint = _resolve_candidate_checkpoint(manifest_path, candidate["checkpoint_path"])
    if not evidence_checkpoint.is_file():
        raise FileNotFoundError(f"Selected filterwise evidence checkpoint does not exist: {evidence_checkpoint}")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping) or not isinstance(model_config.get("checkpoint"), str):
        raise ValueError("Pruned checkpoint build requires model.checkpoint")
    source_checkpoint = _resolve_repo_path(model_config["checkpoint"])
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Dense model checkpoint does not exist: {source_checkpoint}")

    model_wrapper = _load_yolo_checkpoint(source_checkpoint)
    model = _unwrap_model(model_wrapper)
    image_size = int(config.get("experiment", {}).get("image_size", 640))
    if image_size <= 0:
        raise ValueError("experiment.image_size must be positive")
    aligned_image_size = ((image_size + 31) // 32) * 32
    before = {
        "parameter_count": _parameter_count(model),
        "serialized_bytes": source_checkpoint.stat().st_size,
    }
    pruned_model = model
    planned_clusters: dict[str, list[int]] = {}
    skipped_layers: list[str] = []
    for layer in candidate_layers:
        module = dict(pruned_model.named_modules()).get(layer)
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"Configured prune target {layer!r} is not a prunable Conv2d module.")
        clusters_to_remove = int(module.out_channels * selected_ratio) // cluster_size
        if clusters_to_remove <= 0:
            planned_clusters[layer] = []
            skipped_layers.append(layer)
            continue
        scores = compute_channel_importance(pruned_model, criterion="l1")
        if layer not in scores:
            raise ValueError(f"No channel-importance scores are available for prunable layer {layer!r}.")
        specs = plan_low_importance_clusters(
            {layer: scores[layer]}, cluster_size, set(), max_clusters_per_layer=clusters_to_remove
        )
        if len(specs) != clusters_to_remove:
            raise ValueError(f"Could not plan complete low-importance clusters for {layer!r}.")
        prune_indices = tuple(index for spec in specs for index in spec.prune_indices)
        planned_clusters[layer] = list(prune_indices)
        pruned_model = run_filterwise_probe(
            pruned_model,
            torch.zeros(1, 3, aligned_image_size, aligned_image_size),
            layer,
            prune_indices,
        )
    if not any(planned_clusters.values()):
        raise ValueError("Configured prune ratio produces no complete clusters for any candidate layer.")
    if hasattr(model_wrapper, "model"):
        model_wrapper.model = pruned_model
    else:
        model_wrapper = pruned_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "pruned.pt"
    save = getattr(model_wrapper, "save", None)
    if not callable(save):
        raise RuntimeError("Structured-pruned checkpoint requires a save-capable model wrapper.")
    save(output_checkpoint)
    if not output_checkpoint.is_file():
        raise FileNotFoundError(f"Pruned checkpoint was not written: {output_checkpoint}")

    reloaded_model = _unwrap_model(_load_yolo_checkpoint(output_checkpoint))
    after = {
        "parameter_count": _parameter_count(reloaded_model),
        "serialized_bytes": output_checkpoint.stat().st_size,
    }
    if after["parameter_count"] >= before["parameter_count"]:
        raise ValueError("Structured pruning did not reduce parameter count.")
    (output_dir / "pruning_summary.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(source_checkpoint),
                "filterwise_evidence_checkpoint": str(evidence_checkpoint),
                "evidence_layer": evidence_layer,
                "evidence_filters_removed": filters_removed,
                "candidate_layers": candidate_layers,
                "cluster_size": cluster_size,
                "selected_ratio": selected_ratio,
                "prune_ratios": prune_ratios,
                "allowed_map50_95_drop": allowed_map50_95_drop,
                "importance": importance,
                "prune_indices": planned_clusters,
                "skipped_layers": skipped_layers,
                "before": before,
                "after": after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_checkpoint


def _validate_precisions(precisions: Any) -> list[str]:
    if precisions not in (list(OFFICIAL_PRECISIONS[:2]), list(OFFICIAL_PRECISIONS)):
        raise ValueError(
            "Precision matrix must be the ordered list [fp32, fp16] or "
            "[fp32, fp16, int8]"
        )
    return list(precisions)


def load_compression_config(path: str | Path) -> dict[str, Any]:
    """Load a compression-matrix YAML config and validate its precisions."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Compression config must be a mapping.")
    _validate_precisions(config.get("precisions"))
    config["_config_path"] = str(config_path.resolve())
    return config


def planned_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return dense and optionally structured-pruned precision variants."""

    precisions = _validate_precisions(config.get("precisions"))
    pruning = config.get("pruning", {})
    if not isinstance(pruning, Mapping):
        raise ValueError("pruning must be a mapping")
    pruning_enabled = bool(pruning.get("enabled", False))
    if pruning_enabled:
        filterwise_manifest = pruning.get("filterwise_manifest")
        evidence_layer = pruning.get("evidence_layer")
        candidate_layers = pruning.get("candidate_layers")
        cluster_size = pruning.get("cluster_size")
        prune_ratios = _validate_prune_ratios(pruning.get("prune_ratios"), "Enabled pruning")
        evidence_filters_removed = pruning.get("evidence_filters_removed")
        allowed_map50_95_drop = _validate_allowed_map_drop(
            pruning.get("allowed_map50_95_drop"), "Enabled pruning"
        )
        importance = pruning.get("importance")
        if not isinstance(filterwise_manifest, str) or not filterwise_manifest:
            raise ValueError("Enabled pruning requires pruning.filterwise_manifest")
        if not isinstance(evidence_layer, str) or not evidence_layer:
            raise ValueError("Enabled pruning requires pruning.evidence_layer")
        if not isinstance(candidate_layers, list) or not candidate_layers or not all(
            isinstance(layer, str) and layer for layer in candidate_layers
        ):
            raise ValueError("Enabled pruning requires non-empty pruning.candidate_layers")
        if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
            raise ValueError("Enabled pruning requires a positive integer pruning.cluster_size")
        if not isinstance(evidence_filters_removed, int) or isinstance(evidence_filters_removed, bool) or evidence_filters_removed <= 0:
            raise ValueError("Enabled pruning requires a positive pruning.evidence_filters_removed")
        if evidence_filters_removed != cluster_size:
            raise ValueError("Enabled pruning requires evidence removal count equal to pruning.cluster_size")
        if importance != "minimum_weight":
            raise ValueError("Enabled pruning requires pruning.importance='minimum_weight'")

    variant_specs: list[tuple[str, float | None]] = [("dense", None)]
    if pruning_enabled:
        variant_specs.extend((f"cluster-{cluster_size}-ratio-{ratio:g}", ratio) for ratio in prune_ratios)

    rows: list[dict[str, Any]] = []
    for variant_name, prune_ratio in variant_specs:
        for precision in precisions:
            row = {
                    "variant_id": f"{variant_name}-{precision}",
                    "compression": "dense" if prune_ratio is None else "structured_pruning",
                    "precision": precision,
                    "prune_ratio": prune_ratio,
                    "status": "planned",
                    "error": None,
                }
            if prune_ratio is None:
                row["provenance"] = {"planner": "compression_matrix", "stage": "planning"}
            else:
                provenance = {
                    "filterwise_manifest": filterwise_manifest,
                    "evidence_layer": evidence_layer,
                    "candidate_layers": candidate_layers,
                    "cluster_size": cluster_size,
                    "evidence_filters_removed": evidence_filters_removed,
                    "prune_ratio": prune_ratio,
                    "allowed_map50_95_drop": allowed_map50_95_drop,
                    "importance": importance,
                }
                row.update(
                    {
                        "filterwise_manifest": filterwise_manifest,
                        "evidence_layer": evidence_layer,
                        "candidate_layers": candidate_layers,
                        "cluster_size": cluster_size,
                        "prune_ratio": prune_ratio,
                        "allowed_map50_95_drop": allowed_map50_95_drop,
                        "evidence_filters_removed": evidence_filters_removed,
                        "importance": importance,
                        "provenance": provenance,
                    }
                )
            rows.append(row)
    return rows


def write_compression_manifest(
    output_dir: Path,
    rows: list[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    preserve_existing: bool = True,
) -> None:
    """Write manifest and CSV checkpoints while retaining prior partial rows."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    existing_rows: dict[str, dict[str, Any]] = {}
    existing_metadata: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                existing_metadata = {key: value for key, value in payload.items() if key != "rows"}
            existing_rows = {
                str(row["variant_id"]): dict(row)
                for row in payload.get("rows", [])
                if isinstance(row, Mapping) and "variant_id" in row
            }
        except (OSError, TypeError, ValueError, AttributeError):
            existing_rows = {}

    if not preserve_existing:
        existing_rows = {}
    for row in rows:
        if "variant_id" not in row:
            raise ValueError("Every compression manifest row requires variant_id")
        existing_rows[str(row["variant_id"])] = dict(row)

    merged_rows = list(existing_rows.values())
    payload = {**existing_metadata, **dict(metadata or {}), "rows": merged_rows}
    write_experiment_manifest(manifest_path, payload)
    write_metrics_csv(output_dir / "results.csv", merged_rows)


@dataclass
class CompressionMatrixAdapters:
    """Injectable side-effect boundary for local RTX matrix evaluations."""

    build_pruned_checkpoint: Callable[[Mapping[str, Any], Path, float], Path]
    export_checkpoint: Callable[[Path, Mapping[str, Any], Path], Path]
    build_engine: Callable[[Path, Path, str, Path | None, int], Mapping[str, Any]]
    evaluate_engine: Callable[[Path, Mapping[str, Any], str], Mapping[str, Any]]
    parameter_count: Callable[[Path], int]
    benchmark_engine: Callable[[Path, str], Mapping[str, Any]]

    @classmethod
    def defaults(cls, config: Mapping[str, Any]) -> "CompressionMatrixAdapters":
        """Create real adapters lazily, so dry runs require no ML runtime."""

        return cls(
            build_pruned_checkpoint=lambda cfg, output_dir, ratio: build_pruned_checkpoint(
                cfg, output_dir, requested_ratio=ratio
            ),
            export_checkpoint=_export_checkpoint_to_onnx,
            build_engine=lambda onnx, engine, precision, calibration_cache, workspace_mb: _build_rtx_engine(
                onnx, engine, precision, calibration_cache, workspace_mb, config
            ),
            evaluate_engine=_evaluate_engine_accuracy,
            parameter_count=_checkpoint_parameter_count,
            benchmark_engine=_benchmark_rtx_engine,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _matrix_output_dir(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("output_dir"), str):
        raise ValueError("Compression matrix requires experiment.output_dir")
    return _resolve_repo_path(experiment["output_dir"])


def _dense_checkpoint(config: Mapping[str, Any]) -> Path:
    model = config.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("checkpoint"), str):
        raise ValueError("Compression matrix requires model.checkpoint")
    return _resolve_repo_path(model["checkpoint"])


def _calibration_cache(config: Mapping[str, Any]) -> Path | None:
    runtime = config.get("runtime", {})
    value = runtime.get("calibration_cache") if isinstance(runtime, Mapping) else None
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("runtime.calibration_cache must be a non-empty path when configured")
    return _resolve_repo_path(value)


def _workspace_mb(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {})
    value = runtime.get("workspace_mb", 1024) if isinstance(runtime, Mapping) else 1024
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("runtime.workspace_mb must be a positive integer")
    return value


def _evaluation_device(config: Mapping[str, Any]) -> str:
    runtime = config.get("runtime", {})
    value = runtime.get("evaluation_device", runtime.get("device", "0")) if isinstance(runtime, Mapping) else "0"
    if value == "rtx":
        value = "0"
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError("runtime.evaluation_device must be '0', a CUDA device index, or 'cpu'")
    return str(value)


def _row_provenance(config_path: Path, checkpoint: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    calibration_cache = _calibration_cache(config)
    return {
        "config_path": str(config_path.resolve()),
        "dense_checkpoint": str(checkpoint.resolve()),
        "workspace_mb": _workspace_mb(config),
        "calibration_cache": str(calibration_cache.resolve()) if calibration_cache else None,
        "device_label": "rtx3070",
    }


def _load_completed_rows(output_dir: Path, rows: list[Metrics]) -> list[Metrics]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return rows
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_rows = {
            str(row["variant_id"]): row
            for row in payload.get("rows", [])
            if isinstance(row, Mapping) and isinstance(row.get("variant_id"), str)
        }
    except (OSError, TypeError, ValueError, AttributeError):
        return rows
    for row in rows:
        saved = saved_rows.get(str(row["variant_id"]))
        if saved is not None and _is_resumable_completed_row(saved, row):
            row.clear()
            row.update(dict(saved))
    return rows


def _is_resumable_completed_row(saved: Mapping[str, Any], planned: Mapping[str, Any]) -> bool:
    required = (
        "checkpoint_path",
        "onnx_path",
        "engine_path",
        "map50_95",
        "parameter_count",
        "latency_p50_ms",
    )
    return bool(
        saved.get("status") == "completed"
        and saved.get("provenance") == planned.get("provenance")
        and all(saved.get(key) is not None for key in required)
    )


def _record_paths(row: Metrics, checkpoint: Path, onnx: Path | None = None, engine: Path | None = None) -> None:
    row["checkpoint_path"] = str(checkpoint.resolve())
    row["checkpoint_size_bytes"] = checkpoint.stat().st_size
    if onnx is not None:
        row["onnx_path"] = str(onnx.resolve())
        row["onnx_size_bytes"] = onnx.stat().st_size
    if engine is not None:
        row["engine_path"] = str(engine.resolve())
        row["engine_size_bytes"] = engine.stat().st_size


def _failure(row: Metrics, exc: Exception) -> None:
    row["status"] = "failed"
    row["error"] = f"{type(exc).__name__}: {exc}"


def _selected_candidate_id(rows: list[Metrics], allowed_drop: float) -> str | None:
    baseline = next(
        (
            row
            for row in rows
            if row.get("compression") == "dense"
            and row.get("precision") == "fp32"
            and row.get("status") == "completed"
            and row.get("map50_95") is not None
        ),
        None,
    )
    if baseline is None:
        return None
    candidates = [
        row
        for row in rows
        if row.get("compression") == "structured_pruning"
        and row.get("precision") == "fp32"
        and row.get("status") == "completed"
        and row.get("prune_ratio") is not None
        and row.get("map50_95") is not None
        and float(baseline["map50_95"]) - float(row["map50_95"]) <= allowed_drop
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda row: float(row["prune_ratio"]))["variant_id"])


def _write_matrix_artifacts(
    output_dir: Path, config_path: Path, config: Mapping[str, Any], rows: list[Metrics]
) -> None:
    pruning = config.get("pruning", {})
    allowed_drop = float(pruning.get("allowed_map50_95_drop", 0.0)) if isinstance(pruning, Mapping) else 0.0
    write_compression_manifest(
        output_dir,
        rows,
        metadata={
            "config_path": str(config_path.resolve()),
            "selected_candidate_id": _selected_candidate_id(rows, allowed_drop),
        },
        preserve_existing=False,
    )


def _pruned_checkpoint_for_row(
    row: Mapping[str, Any], config: Mapping[str, Any], output_dir: Path,
    active: CompressionMatrixAdapters, cached: dict[float, Path], failures: dict[float, Exception]
) -> Path:
    ratio = float(row["prune_ratio"])
    if ratio in failures:
        raise failures[ratio]
    if ratio in cached:
        return cached[ratio]
    candidate_dir = output_dir / "checkpoints" / f"cluster-{int(row['cluster_size'])}-ratio-{ratio:g}"
    try:
        checkpoint = Path(active.build_pruned_checkpoint(config, candidate_dir, ratio))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pruned checkpoint was not written: {checkpoint}")
    except Exception as exc:
        failures[ratio] = exc
        raise
    cached[ratio] = checkpoint
    return checkpoint


def run_compression_matrix(
    config_path: str | Path,
    dry_run: bool = False,
    adapters: CompressionMatrixAdapters | None = None,
) -> list[Metrics]:
    """Run or plan the dense and structured-pruned RTX precision matrix."""

    config_file = Path(config_path).resolve()
    config = load_compression_config(config_file)
    output_dir = _matrix_output_dir(config)
    rows = planned_variants(config)
    checkpoint = _dense_checkpoint(config)
    common_provenance = _row_provenance(config_file, checkpoint, config)
    for row in rows:
        row["provenance"] = {**dict(row["provenance"]), **common_provenance}
    if dry_run:
        _write_matrix_artifacts(output_dir, config_file, config, rows)
        return rows

    active = adapters or CompressionMatrixAdapters.defaults(config)
    rows = _load_completed_rows(output_dir, rows)
    cached_checkpoints: dict[float, Path] = {}
    ratio_failures: dict[float, Exception] = {}
    total_rows = len(rows)
    for row_index, row in enumerate(rows, start=1):
        if row.get("status") == "completed":
            if row.get("compression") == "structured_pruning":
                saved = Path(str(row["checkpoint_path"]))
                if saved.is_file():
                    cached_checkpoints.setdefault(float(row["prune_ratio"]), saved)
            continue
        variant_id = str(row["variant_id"])
        precision = str(row["precision"]).upper()
        print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: preparing checkpoint", flush=True)
        try:
            if row["compression"] == "dense":
                variant_checkpoint = checkpoint
            else:
                variant_checkpoint = _pruned_checkpoint_for_row(
                    row, config, output_dir, active, cached_checkpoints, ratio_failures
                )
            variant_dir = output_dir / "variants" / str(row["variant_id"])
            print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: exporting ONNX", flush=True)
            onnx = Path(active.export_checkpoint(variant_checkpoint, config, variant_dir))
            if not onnx.is_file():
                raise FileNotFoundError(f"ONNX export was not written: {onnx}")
            engine = variant_dir / "model.engine"
            print(
                f"[compression-matrix] {row_index}/{total_rows} {variant_id}: "
                f"building {precision} TensorRT engine (this may be quiet for a while)",
                flush=True,
            )
            build = dict(
                active.build_engine(onnx, engine, str(row["precision"]), _calibration_cache(config), _workspace_mb(config))
            )
            if not engine.is_file():
                raise FileNotFoundError(f"TensorRT engine was not written: {engine}")
            evaluation_engine = Path(str(build.get("evaluation_engine_path", engine)))
            if not evaluation_engine.is_file():
                raise FileNotFoundError(f"TensorRT evaluation engine was not written: {evaluation_engine}")
            print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: evaluating detection accuracy", flush=True)
            metrics = dict(active.evaluate_engine(evaluation_engine, config, _evaluation_device(config)))
            print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: benchmarking RTX 3070 latency", flush=True)
            benchmark = dict(active.benchmark_engine(engine, "rtx3070"))
            _record_paths(row, variant_checkpoint, onnx, engine)
            row["evaluation_engine_path"] = str(evaluation_engine.resolve())
            row.update({key: value for key, value in metrics.items() if key != "precision"})
            if "precision" in metrics:
                row["detection_precision"] = metrics["precision"]
            row.update(benchmark)
            row["parameter_count"] = int(active.parameter_count(variant_checkpoint))
            row["commands"] = {
                "build": build.get("command"),
                "benchmark": benchmark.get("command"),
            }
            row["cache_provenance"] = {
                "calibration_cache": build.get("calibration_cache_provenance", common_provenance["calibration_cache"]),
                "checkpoint": str(variant_checkpoint.resolve()),
            }
            row["status"] = "completed"
            row["error"] = None
            print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: completed", flush=True)
        except Exception as exc:
            _failure(row, exc)
            print(f"[compression-matrix] {row_index}/{total_rows} {variant_id}: failed - {exc}", flush=True)
        finally:
            _write_matrix_artifacts(output_dir, config_file, config, rows)
    return rows


def _export_checkpoint_to_onnx(checkpoint: Path, config: Mapping[str, Any], output_dir: Path) -> Path:
    from infrared_detection.export import export_yolo

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = output_dir / "export-source.pt"
    shutil.copy2(checkpoint, staged_checkpoint)
    exported = Path(export_yolo(staged_checkpoint, format="onnx", imgsz=int(config.get("experiment", {}).get("image_size", 640))))
    if not exported.is_file():
        raise FileNotFoundError(f"ONNX export was not written: {exported}")
    isolated = output_dir / "model.onnx"
    if exported.resolve() != isolated.resolve():
        shutil.copy2(exported, isolated)
    return isolated


def _build_rtx_engine(
    onnx: Path,
    engine: Path,
    precision: str,
    calibration_cache: Path | None,
    workspace_mb: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    from infrared_detection.benchmarking.rtx import (
        build_tensorrt_engine,
        prepare_tensorrt_precision_onnx,
        write_ultralytics_engine_metadata,
    )

    engine.parent.mkdir(parents=True, exist_ok=True)
    prepared_onnx = onnx if precision == "fp32" else engine.parent / f"{onnx.stem}.{precision}.onnx"
    preparation = prepare_tensorrt_precision_onnx(onnx, prepared_onnx, precision)
    build = build_tensorrt_engine(prepared_onnx, engine, precision, calibration_cache, workspace_mb)
    build["precision_preparation"] = preparation
    data = config.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("dataset_yaml"), str):
        raise ValueError("Compression matrix requires data.dataset_yaml for engine metadata")
    dataset_path = _resolve_repo_path(data["dataset_yaml"])
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    names = dataset.get("names") if isinstance(dataset, Mapping) else None
    if not isinstance(names, (list, dict)):
        raise ValueError(f"Dataset YAML has no names list or mapping: {dataset_path}")
    image_size = int(config.get("experiment", {}).get("image_size", 640))
    metadata = {
        "stride": 32,
        "task": "detect",
        "batch": 1,
        "imgsz": [image_size, image_size],
        "names": names,
    }
    evaluation_engine = engine.parent / "model.evaluation.engine"
    build["evaluation_engine_path"] = write_ultralytics_engine_metadata(engine, evaluation_engine, metadata)
    return build


def _evaluate_engine_accuracy(
    engine: Path, config: Mapping[str, Any], device: str
) -> Mapping[str, Any]:
    from infrared_detection.evaluation.detection_metrics import evaluate_yolo

    data = config.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("dataset_yaml"), str):
        raise ValueError("Compression matrix requires data.dataset_yaml")
    runtime = config.get("runtime", {})
    if device != "cpu":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                f"Evaluation requested on CUDA device {device!r}, but PyTorch cannot see a CUDA device. "
                "Check CUDA_VISIBLE_DEVICES and the installed PyTorch CUDA runtime."
            )
    wrapper = _load_yolo_checkpoint(engine)
    return evaluate_yolo(
        wrapper,
        str(_resolve_repo_path(data["dataset_yaml"])),
        "val",
        int(config.get("experiment", {}).get("image_size", 640)),
        device,
        float(runtime.get("conf", 0.25)) if isinstance(runtime, Mapping) else 0.25,
        float(runtime.get("iou", 0.6)) if isinstance(runtime, Mapping) else 0.6,
    )


def _checkpoint_parameter_count(checkpoint: Path) -> int:
    return _parameter_count(_unwrap_model(_load_yolo_checkpoint(checkpoint)))


def _benchmark_rtx_engine(engine: Path, device_label: str) -> Mapping[str, Any]:
    from infrared_detection.benchmarking.rtx import benchmark_tensorrt_engine

    return benchmark_tensorrt_engine(engine, device_label=device_label)
