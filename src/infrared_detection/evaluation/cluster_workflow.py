"""Config-driven two-stage structural cluster-pruning evaluation."""

from __future__ import annotations

import inspect
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrared_detection.common.experiment_config import load_experiment_config, resolve_repo_path
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv
from infrared_detection.evaluation.cluster_candidates import classify_candidate, select_cluster_candidates


Metrics = dict[str, Any]

_ORIN_TARGET = "jetson_orin_nano"
_FEASIBLE_STATUSES = frozenset({"primary_feasible", "exploratory_feasible"})
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
    if adapters is None and not _is_jetson_orin_runtime():
        raise RuntimeError(
            "Refusing to claim Jetson Orin Nano results on an unverified host. "
            "Run the default workflow on the Jetson Orin Nano or inject a remote Orin adapter."
        )

    active = adapters or ClusterEvaluationAdapters.defaults(config)
    pruning = config["pruning"]
    rtx_device = str(targets["rtx_screening_device"])
    orin_execution_device = _orin_execution_device(config)
    rows = _planned_rows(config)
    baseline = next(row for row in rows if row["stage"] == "baseline")
    try:
        baseline_model = active.load_model(checkpoint)
        baseline_metrics = active.evaluate(baseline_model, config, orin_execution_device)
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
        baseline["target_device"] = orin_target
        baseline["evaluation_device"] = orin_execution_device
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
    width = int(module.out_channels)
    requested_clusters = max(1, int(width * ratio) // cluster_size)
    available_clusters = (width - 1) // cluster_size
    if available_clusters < 1:
        raise ValueError(
            f"Layer {layer!r} has no complete cluster of size {cluster_size} that can be pruned safely."
        )
    requested_clusters = min(requested_clusters, available_clusters)
    protected = set(config["pruning"]["protected_layers"])
    example = torch.zeros(1, 3, int(config["experiment"]["image_size"]), int(config["experiment"]["image_size"]))

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
