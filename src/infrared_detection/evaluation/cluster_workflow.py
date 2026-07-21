"""Config-driven two-stage structural cluster-pruning evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrared_detection.common.experiment_config import load_experiment_config, resolve_repo_path
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv
from infrared_detection.evaluation.cluster_candidates import classify_candidate


Metrics = dict[str, Any]


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


def _base_row(candidate_id: str, stage: str, cluster_size: int | None = None, ratio: float | None = None) -> Metrics:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "cluster_size": cluster_size,
        "prune_ratio": ratio,
        "status": "planned",
        "error": None,
        "checkpoint_path": None,
        "exported_path": None,
        "screening_device": None,
        "target_device": None,
    }


def _append_row(rows: list[Metrics], row: Metrics) -> None:
    """Append a row while preserving a unique identifier in every manifest."""

    candidate_id = str(row["candidate_id"])
    duplicates = sum(existing["candidate_id"] == candidate_id for existing in rows)
    if duplicates:
        row["candidate_id"] = f"{candidate_id}-{duplicates + 1}"
    rows.append(row)


def _write_artifacts(output_dir: Path, config_path: Path, rows: list[Metrics]) -> None:
    write_metrics_csv(output_dir / "candidates.csv", rows)
    write_experiment_manifest(
        output_dir / "manifest.json",
        {
            "config_path": str(config_path),
            "candidate_count": len(rows),
            "candidate_ids": [row["candidate_id"] for row in rows],
            "rows": rows,
        },
    )


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
            _append_row(rows, _base_row(_candidate_id("probe", layer=layer, cluster_size=int(size)), "probe", int(size), float(pruning["probe_ratios"][0])))
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


def _record_export(row: Metrics, exported: Path, baseline: Mapping[str, Any] | None = None) -> None:
    row["exported_path"] = str(exported)
    if exported.exists():
        row["serialized_bytes"] = exported.stat().st_size
    if baseline is not None:
        baseline_bytes = baseline.get("serialized_bytes")
        candidate_bytes = row.get("serialized_bytes")
        if baseline_bytes and candidate_bytes is not None:
            row["serialized_reduction"] = 1.0 - float(candidate_bytes) / float(baseline_bytes)


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

    active = adapters or ClusterEvaluationAdapters.defaults(config)
    pruning = config["pruning"]
    rtx_device = str(config["targets"]["rtx_screening_device"])
    orin_device = str(config["targets"]["orin_target"])
    rows: list[Metrics] = []

    baseline = _base_row("baseline", "baseline", ratio=0.0)
    _append_row(rows, baseline)
    try:
        baseline_model = active.load_model(checkpoint)
        _metrics_row(baseline, active.evaluate(baseline_model, config, rtx_device), active.stats(checkpoint))
        baseline["status"] = "completed"
        baseline["screening_device"] = rtx_device
        layers = active.safe_layers(baseline_model, config)
    except Exception as exc:
        _failure(baseline, exc)
        _write_artifacts(output_dir, resolved_config_path, rows)
        return rows

    surviving_sizes: set[int] = set()
    for layer in layers:
        for size in pruning["cluster_sizes"]:
            row = _base_row(_candidate_id("probe", layer=layer, cluster_size=int(size)), "probe", int(size), float(pruning["probe_ratios"][0]))
            _append_row(rows, row)
            try:
                probe_model = active.load_model(checkpoint)
                pruned_model = active.make_probe(probe_model, layer, int(size), float(pruning["probe_ratios"][0]))
                _metrics_row(row, active.evaluate(pruned_model, config, rtx_device), active.stats(pruned_model), baseline)
                exported = Path(active.export(pruned_model, config, output_dir / row["candidate_id"]))
                _record_export(row, exported, baseline)
                row.update(active.profile(exported, rtx_device))
                row["screening_device"] = rtx_device
                row["status"] = "screened_in" if _screen_probe(row, float(baseline["map50_95"]), config) else "screened_out"
                if row["status"] == "screened_in":
                    surviving_sizes.add(int(size))
            except Exception as exc:
                _failure(row, exc)

    for size in sorted(surviving_sizes):
        for ratio in pruning["global_ratios"]:
            row = _base_row(_candidate_id("global", cluster_size=size, ratio=float(ratio)), "global", size, float(ratio))
            _append_row(rows, row)
            try:
                candidate = active.make_global(active.load_model(checkpoint), size, float(ratio))
                candidate_dir = output_dir / row["candidate_id"]
                best_checkpoint = active.fine_tune(candidate, config, candidate_dir)
                reloaded = active.reload_model(Path(best_checkpoint))
                _metrics_row(row, active.evaluate(reloaded, config, orin_device), active.stats(reloaded), baseline)
                exported = Path(active.export(Path(best_checkpoint), config, candidate_dir))
                row["checkpoint_path"] = str(Path(best_checkpoint))
                _record_export(row, exported, baseline)
                row["target_device"] = orin_device
                row.update(active.profile(exported, orin_device))
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
    configured = config["pruning"].get("safe_layers")
    if configured:
        protected = set(config["pruning"]["protected_layers"])
        return [layer for layer in configured if layer not in protected and not layer.startswith("model.22")]
    from infrared_detection.compression.pruning import compute_channel_importance

    protected = set(config["pruning"]["protected_layers"])
    return [name for name in sorted(compute_channel_importance(_unwrap_model(model))) if name not in protected and not name.startswith("model.22")]


def _structural_probe(model: Any, layer: str, cluster_size: int, ratio: float, config: Mapping[str, Any]) -> Any:
    import torch

    from infrared_detection.compression.pruning import ClusterSpec
    from infrared_detection.compression.pruning.cluster_probe import run_structural_probe as prune

    module = dict(_unwrap_model(model).named_modules())[layer]
    width = int(getattr(module, "out_channels", None) or getattr(module, "out_features"))
    count = max(cluster_size, (int(width * ratio) // cluster_size) * cluster_size)
    count = min(count, width - 1)
    indices = tuple(range(count))
    example = torch.zeros(1, 3, int(config["experiment"]["image_size"]), int(config["experiment"]["image_size"]))
    pruned = prune(_unwrap_model(model), example, ClusterSpec(layer, cluster_size, indices))
    if hasattr(model, "model"):
        model.model = pruned
        return model
    return pruned


def _structural_global(model: Any, cluster_size: int, ratio: float, config: Mapping[str, Any]) -> Any:
    for layer in _safe_layers(model, config):
        model = _structural_probe(model, layer, cluster_size, ratio, config)
    return model


def _fine_tune_yolo(model: Any, config: Mapping[str, Any], output_dir: Path) -> Path:
    runtime = config["runtime"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = model.train(
        data=str(resolve_repo_path(config["data"]["dataset_yaml"], _repo_root())),
        epochs=int(config["pruning"]["fine_tune_epochs"]),
        imgsz=int(config["experiment"]["image_size"]),
        device=runtime["device"],
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
    if hasattr(checkpoint, "export"):
        exported = checkpoint.export(**kwargs)
    else:
        exported = export_yolo(checkpoint, **kwargs)
    return Path(exported)


def _profile_export(exported: Path, device: str) -> Metrics:
    del device
    from infrared_detection.benchmarking.jetson import benchmark_tensorrt_engine

    return benchmark_tensorrt_engine(exported)


def _collect_stats(model: Any) -> Metrics:
    from infrared_detection.evaluation.model_stats import collect_model_stats

    return collect_model_stats(_unwrap_model(model))
