"""Build structured-pruned PyTorch checkpoints independently of LPQ export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from infrared_detection.compression.pruning.cluster_probe import run_filterwise_probe
from infrared_detection.compression.pruning.cluster_selection import plan_low_importance_clusters
from infrared_detection.compression.pruning.importance import minimum_weight_scores
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


def _validate_prune_ratios(value: Any) -> list[float]:
    if not isinstance(value, list) or not value or not all(
        isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 < ratio < 1
        for ratio in value
    ):
        raise ValueError("pruning.prune_ratios must be a non-empty list of values between 0 and 1")
    return [float(ratio) for ratio in value]


def _validate_pruning(config: Mapping[str, Any]) -> tuple[list[str], int, list[float]]:
    pruning = config.get("pruning")
    if not isinstance(pruning, Mapping):
        raise ValueError("pruning must be a mapping")
    layers = pruning.get("candidate_layers")
    cluster_size = pruning.get("cluster_size")
    if not isinstance(layers, list) or not layers or not all(isinstance(layer, str) and layer for layer in layers):
        raise ValueError("pruning.candidate_layers must be a non-empty list of layer names")
    if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
        raise ValueError("pruning.cluster_size must be a positive integer")
    if pruning.get("importance") != "minimum_weight":
        raise ValueError("pruning.importance must be 'minimum_weight'")
    drop = pruning.get("allowed_map50_95_drop")
    if not isinstance(drop, (int, float)) or isinstance(drop, bool) or drop < 0:
        raise ValueError("pruning.allowed_map50_95_drop must be non-negative")
    return layers, cluster_size, _validate_prune_ratios(pruning.get("prune_ratios"))


def load_pruning_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Cluster-pruning config must be a mapping")
    model = config.get("model")
    experiment = config.get("experiment")
    if not isinstance(model, Mapping) or not isinstance(model.get("checkpoint"), str):
        raise ValueError("Cluster pruning requires model.checkpoint")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("output_dir"), str):
        raise ValueError("Cluster pruning requires experiment.output_dir")
    _validate_pruning(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def planned_pruning_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    layers, cluster_size, ratios = _validate_pruning(config)
    pruning = config["pruning"]
    return [
        {
            "variant_id": f"cluster-{cluster_size}-ratio-{ratio:g}",
            "compression": "structured_pruning",
            "candidate_layers": layers,
            "cluster_size": cluster_size,
            "prune_ratio": ratio,
            "allowed_map50_95_drop": float(pruning["allowed_map50_95_drop"]),
            "importance": pruning["importance"],
            "status": "planned",
            "error": None,
        }
        for ratio in ratios
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _output_dir(config: Mapping[str, Any]) -> Path:
    return _resolve_repo_path(config["experiment"]["output_dir"])


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
    """Build one minimum-weight structured-pruned checkpoint from the configured source."""
    import torch
    from torch import nn

    layers, cluster_size, ratios = _validate_pruning(config)
    pruning = config["pruning"]
    if requested_ratio is None:
        if len(ratios) != 1:
            raise ValueError("Build requires requested_ratio when multiple pruning.prune_ratios are configured")
        selected_ratio = ratios[0]
    elif not isinstance(requested_ratio, (int, float)) or isinstance(requested_ratio, bool) or float(requested_ratio) not in ratios:
        raise ValueError("requested_ratio must be one of pruning.prune_ratios")
    else:
        selected_ratio = float(requested_ratio)

    source_value = config["model"]["checkpoint"]
    source_checkpoint = _resolve_repo_path(source_value)
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
    for layer in layers:
        module = dict(pruned_model.named_modules()).get(layer)
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"Configured prune target {layer!r} is not a prunable Conv2d module.")
        clusters_to_remove = int(module.out_channels * selected_ratio) // cluster_size
        if clusters_to_remove <= 0:
            planned_clusters[layer] = []
            skipped_layers.append(layer)
            continue
        scores = minimum_weight_scores(module)
        specs = plan_low_importance_clusters(
            {layer: scores}, cluster_size, set(), max_clusters_per_layer=clusters_to_remove
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
                "candidate_layers": layers,
                "cluster_size": cluster_size,
                "selected_ratio": selected_ratio,
                "prune_ratios": ratios,
                "allowed_map50_95_drop": float(pruning["allowed_map50_95_drop"]),
                "importance": pruning["importance"],
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


def _write_pruning_artifacts(output_dir: Path, rows: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    write_experiment_manifest(
        output_dir / "manifest.json",
        {
            "config_path": config["_config_path"],
            "source_checkpoint": str(_resolve_repo_path(config["model"]["checkpoint"]).resolve()),
            "candidate_count": len(rows),
            "rows": rows,
        },
    )
    write_metrics_csv(output_dir / "results.csv", rows)


def run_cluster_pruning(
    config_path: str | Path,
    dry_run: bool = False,
    build_checkpoint: Callable[[Mapping[str, Any], Path, float], Path] | None = None,
) -> list[dict[str, Any]]:
    """Create configured `.pt` variants and record each result independently."""
    config = load_pruning_config(config_path)
    output_dir = _output_dir(config)
    rows = planned_pruning_variants(config)
    for row in rows:
        row["candidate_id"] = row["variant_id"]
        row["checkpoint_path"] = None
        row["pruning_summary_path"] = None
    _write_pruning_artifacts(output_dir, rows, config)
    if dry_run:
        return rows

    source_checkpoint = _resolve_repo_path(config["model"]["checkpoint"])
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Dense model checkpoint does not exist: {source_checkpoint}")
    builder = build_checkpoint or build_pruned_checkpoint
    for row in rows:
        candidate_dir = output_dir / str(row["candidate_id"])
        try:
            checkpoint = Path(builder(config, candidate_dir, float(row["prune_ratio"])))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Pruned checkpoint was not written: {checkpoint}")
            row["checkpoint_path"] = str(checkpoint.resolve())
            summary_path = checkpoint.parent / "pruning_summary.json"
            row["pruning_summary_path"] = str(summary_path.resolve()) if summary_path.is_file() else None
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["parameter_count_before"] = summary.get("before", {}).get("parameter_count")
                row["parameter_count_after"] = summary.get("after", {}).get("parameter_count")
                row["checkpoint_size_bytes"] = summary.get("after", {}).get("serialized_bytes")
            row["status"] = "completed"
            row["error"] = None
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _write_pruning_artifacts(output_dir, rows, config)
    return rows
