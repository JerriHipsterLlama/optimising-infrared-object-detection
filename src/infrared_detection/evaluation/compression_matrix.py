"""Planning and artifact writing for RTX compression experiments."""

from __future__ import annotations

import json
import hashlib
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


def _load_yolo_checkpoint(checkpoint_path: Path) -> Any:
    from infrared_detection.evaluation.single_layer_performance_screening import (
        _install_legacy_pathlib_checkpoint_compatibility,
    )

    _install_legacy_pathlib_checkpoint_compatibility()
    from ultralytics import YOLO

    return YOLO(str(checkpoint_path))


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_pruned_checkpoint(
    config: Mapping[str, Any], output_dir: Path, requested_ratio: float | None = None
) -> Path:
    """Build, reload, and record a dense-source structured-pruned checkpoint."""

    import torch
    from torch import nn

    pruning = config.get("pruning")
    if not isinstance(pruning, Mapping):
        raise ValueError("pruning must be a mapping")
    candidate_layers = pruning.get("candidate_layers")
    cluster_size = pruning.get("cluster_size")
    prune_ratios = _validate_prune_ratios(pruning.get("prune_ratios"), "Pruned checkpoint build")
    allowed_map50_95_drop = _validate_allowed_map_drop(
        pruning.get("allowed_map50_95_drop"), "Pruned checkpoint build"
    )
    importance = pruning.get("importance")
    if not isinstance(candidate_layers, list) or not candidate_layers or not all(
        isinstance(layer, str) and layer for layer in candidate_layers
    ):
        raise ValueError("Pruned checkpoint build requires non-empty pruning.candidate_layers")
    if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
        raise ValueError("Pruned checkpoint build requires a positive pruning.cluster_size")
    if importance != "minimum_weight":
        raise ValueError("Pruned checkpoint build requires pruning.importance='minimum_weight'")
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
        scores = compute_channel_importance(pruned_model, criterion=str(importance))
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
        candidate_layers = pruning.get("candidate_layers")
        cluster_size = pruning.get("cluster_size")
        prune_ratios = _validate_prune_ratios(pruning.get("prune_ratios"), "Enabled pruning")
        allowed_map50_95_drop = _validate_allowed_map_drop(
            pruning.get("allowed_map50_95_drop"), "Enabled pruning"
        )
        importance = pruning.get("importance")
        if not isinstance(candidate_layers, list) or not candidate_layers or not all(
            isinstance(layer, str) and layer for layer in candidate_layers
        ):
            raise ValueError("Enabled pruning requires non-empty pruning.candidate_layers")
        if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
            raise ValueError("Enabled pruning requires a positive integer pruning.cluster_size")
        if importance != "minimum_weight":
            raise ValueError("Enabled pruning requires pruning.importance='minimum_weight'")

    variant_specs: list[tuple[str, float | None]] = [("dense", None)]
    if pruning_enabled:
        variant_specs.extend((f"cluster-{cluster_size}-ratio-{ratio:g}", ratio) for ratio in prune_ratios)

    rows: list[dict[str, Any]] = []
    for variant_name, prune_ratio in variant_specs:
        candidate_id = variant_name
        for precision in precisions:
            row = {
                    "variant_id": f"{variant_name}-{precision}",
                    "candidate_id": candidate_id,
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
                    "candidate_layers": candidate_layers,
                    "cluster_size": cluster_size,
                    "prune_ratio": prune_ratio,
                    "allowed_map50_95_drop": allowed_map50_95_drop,
                    "importance": importance,
                }
                row.update(
                    {
                        "candidate_layers": candidate_layers,
                        "cluster_size": cluster_size,
                        "prune_ratio": prune_ratio,
                        "allowed_map50_95_drop": allowed_map50_95_drop,
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
    """Injectable side-effect boundary for ONNX artifact generation."""

    build_pruned_checkpoint: Callable[[Mapping[str, Any], Path, float], Path]
    export_checkpoint: Callable[[Path, Mapping[str, Any], Path], Path]
    convert_precision: Callable[[Path, str, Path, Mapping[str, Any]], Mapping[str, Any]]

    @classmethod
    def defaults(cls, config: Mapping[str, Any]) -> "CompressionMatrixAdapters":
        """Create real adapters lazily, so dry runs require no ML runtime."""

        return cls(
            build_pruned_checkpoint=lambda cfg, output_dir, ratio: build_pruned_checkpoint(
                cfg, output_dir, requested_ratio=ratio
            ),
            export_checkpoint=_export_checkpoint_to_onnx,
            convert_precision=_convert_precision_onnx,
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


def _row_provenance(config_path: Path, checkpoint: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(config_path.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "config_fingerprint": hashlib.sha256(
            json.dumps(
                {key: value for key, value in config.items() if key != "_config_path"},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
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
    if saved.get("status") != "completed" or saved.get("provenance") != planned.get("provenance"):
        return False
    paths = (saved.get("base_onnx_path"), saved.get("onnx_path"))
    if any(not isinstance(path, str) or not Path(path).is_file() for path in paths):
        return False
    checkpoint = saved.get("checkpoint_path")
    return isinstance(checkpoint, str) and Path(checkpoint).is_file()


def _failure(row: Metrics, exc: Exception) -> None:
    row["status"] = "failed"
    row["error"] = f"{type(exc).__name__}: {exc}"


def _write_matrix_artifacts(
    output_dir: Path, config_path: Path, config: Mapping[str, Any], rows: list[Metrics]
) -> None:
    write_compression_manifest(
        output_dir,
        rows,
        metadata={
            "config_path": str(config_path.resolve()),
            "candidate_count": len({str(row["candidate_id"]) for row in rows}),
            "precision_count": len(rows),
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
    """Create one base ONNX export per candidate and convert its precisions."""

    config_file = Path(config_path).resolve()
    config = load_compression_config(config_file)
    output_dir = _matrix_output_dir(config)
    rows = planned_variants(config)
    checkpoint = _dense_checkpoint(config)
    common_provenance = _row_provenance(config_file, checkpoint, config)
    for row in rows:
        row["provenance"] = {
            **dict(row["provenance"]),
            **common_provenance,
            "candidate_id": row["candidate_id"],
            "precision": row["precision"],
        }
    if dry_run:
        _write_matrix_artifacts(output_dir, config_file, config, rows)
        return rows

    active = adapters or CompressionMatrixAdapters.defaults(config)
    rows = _load_completed_rows(output_dir, rows)
    cached_checkpoints: dict[float, Path] = {}
    ratio_failures: dict[float, Exception] = {}
    grouped: dict[str, list[Metrics]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    for candidate_id, candidate_rows in grouped.items():
        completed_rows = [row for row in candidate_rows if row.get("status") == "completed"]
        if completed_rows and candidate_rows[0]["compression"] == "structured_pruning":
            saved_checkpoint = Path(str(completed_rows[0]["checkpoint_path"]))
            cached_checkpoints.setdefault(float(candidate_rows[0]["prune_ratio"]), saved_checkpoint)
        try:
            if candidate_rows[0]["compression"] == "dense":
                variant_checkpoint = checkpoint
            else:
                variant_checkpoint = _pruned_checkpoint_for_row(
                    candidate_rows[0], config, output_dir, active, cached_checkpoints, ratio_failures
                )
            candidate_dir = output_dir / candidate_id
            reusable = next(
                (row for row in completed_rows if Path(str(row.get("base_onnx_path", ""))).is_file()),
                None,
            )
            if reusable is not None:
                source_onnx = Path(str(reusable["base_onnx_path"]))
            else:
                source_onnx = Path(active.export_checkpoint(variant_checkpoint, config, candidate_dir / "source"))
                if not source_onnx.is_file():
                    raise FileNotFoundError(f"Base ONNX export was not written: {source_onnx}")
        except Exception as exc:
            for row in candidate_rows:
                if row.get("status") != "completed":
                    _failure(row, exc)
                    _write_matrix_artifacts(output_dir, config_file, config, rows)
            continue

        for row in candidate_rows:
            if row.get("status") == "completed":
                continue
            precision = str(row["precision"])
            onnx = candidate_dir / precision / "model.onnx"
            try:
                conversion = dict(active.convert_precision(source_onnx, precision, onnx, config))
                if not onnx.is_file():
                    raise FileNotFoundError(f"Precision ONNX was not written: {onnx}")
                row.update(
                    {
                        "checkpoint_path": str(variant_checkpoint.resolve()),
                        "base_onnx_path": str(source_onnx.resolve()),
                        "base_onnx_size_bytes": source_onnx.stat().st_size,
                        "onnx_path": str(onnx.resolve()),
                        "onnx_size_bytes": onnx.stat().st_size,
                        "conversion": conversion,
                        "status": "completed",
                        "error": None,
                    }
                )
                if row["compression"] == "structured_pruning":
                    summary = variant_checkpoint.parent / "pruning_summary.json"
                    if summary.is_file():
                        row["pruning_summary_path"] = str(summary.resolve())
            except Exception as exc:
                _failure(row, exc)
            finally:
                _write_matrix_artifacts(output_dir, config_file, config, rows)
    return rows


def _export_checkpoint_to_onnx(checkpoint: Path, config: Mapping[str, Any], output_dir: Path) -> Path:
    from infrared_detection.evaluation.single_layer_performance_screening import (
        _install_legacy_pathlib_checkpoint_compatibility,
    )

    _install_legacy_pathlib_checkpoint_compatibility()
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


def _convert_precision_onnx(
    source: Path, precision: str, output: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    from infrared_detection.compression.quantization.onnx_precision import (
        calibration_reader_from_onnx,
        prepare_precision_onnx,
    )

    quantization = config.get("quantization", {})
    data = config.get("data", {})
    if not isinstance(quantization, Mapping) or not isinstance(data, Mapping):
        raise ValueError("quantization and data config sections must be mappings")
    reader = None
    calibration_method = str(quantization.get("calibration_method", "entropy")).lower()
    if precision == "int8":
        calibration_dir = data.get("calibration_image_dir")
        if not isinstance(calibration_dir, str) or not calibration_dir:
            raise ValueError("INT8 conversion requires data.calibration_image_dir")
        calibration_split = data.get("calibration_split", data.get("split"))
        reader = calibration_reader_from_onnx(
            source,
            _resolve_repo_path(calibration_dir),
            sample_limit=int(quantization.get("calibration_samples", 128)),
            batch_size=int(quantization.get("batch_size", 1)),
            calibration_split=str(calibration_split) if calibration_split is not None else None,
        )
    return prepare_precision_onnx(
        source,
        output,
        precision,
        calibration_reader=reader,
        calibration_method=calibration_method,
    )
