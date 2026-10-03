"""Config-driven structural cluster-size screening."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrared_detection.common.experiment_config import load_experiment_config, resolve_repo_path
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


Metrics = dict[str, Any]

@dataclass
class ClusterScreeningAdapters:
    """Model operations needed for the cluster-size accuracy screen."""

    load_model: Callable[[Path], Any]
    evaluate: Callable[[Any, Mapping[str, Any], str], Metrics]
    safe_layers: Callable[[Any, Mapping[str, Any]], list[str]]
    make_probe: Callable[[Any, str, int, float], Any]
    stats: Callable[[Any], Metrics]

    @classmethod
    def defaults(cls, config: Mapping[str, Any]) -> "ClusterScreeningAdapters":
        return cls(
            load_model=_load_yolo,
            evaluate=_evaluate_yolo,
            safe_layers=_safe_layers,
            make_probe=lambda model, layer, size, ratio: _structural_probe(
                model, layer, size, ratio, config
            ),
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


def _resolve_config(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = load_experiment_config(config_path)
    repo_root = _repo_root()
    checkpoint = resolve_repo_path(config["model"]["checkpoint"], repo_root)
    output_dir = resolve_repo_path(config["experiment"]["output_dir"], repo_root)
    return config, checkpoint, output_dir, Path(config["_config_path"])


def _validate_screening_config(config: Mapping[str, Any]) -> None:
    pruning = config.get("pruning")
    if not isinstance(pruning, Mapping):
        raise ValueError("pruning must be a configuration mapping.")
    layers = pruning.get("candidate_layers", pruning.get("safe_layers"))
    if not isinstance(layers, list) or not layers or any(not isinstance(layer, str) or not layer.strip() for layer in layers):
        raise ValueError("pruning.candidate_layers must be a non-empty list of layer paths.")
    if len(set(layers)) != len(layers):
        raise ValueError("pruning.candidate_layers must not contain duplicates.")
    sizes = pruning.get("cluster_sizes")
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("pruning.cluster_sizes must be a non-empty list of positive integers.")
    ratios = pruning.get("probe_ratios")
    if not isinstance(ratios, list) or not ratios or any(
        isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1
        for ratio in ratios
    ):
        raise ValueError("pruning.probe_ratios must be a non-empty list of values between 0 and 1.")


def _planned_rows(config: Mapping[str, Any]) -> list[Metrics]:
    _validate_screening_config(config)
    pruning = config["pruning"]
    layers = pruning.get("candidate_layers", pruning.get("safe_layers"))
    rows: list[Metrics] = []
    _append_row(rows, _base_row("baseline", "baseline", ratio=0.0))
    for layer in layers:
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
    return rows


def _metrics_row(row: Metrics, metrics: Mapping[str, Any], stats: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> None:
    row.update(dict(metrics))
    row.update(dict(stats))
    if baseline is not None:
        row["map50_95_drop"] = float(baseline["map50_95"]) - float(row["map50_95"])
        baseline_parameters = baseline.get("parameter_count")
        candidate_parameters = row.get("parameter_count")
        if baseline_parameters and candidate_parameters is not None:
            row["parameter_count_reduction"] = 1.0 - float(candidate_parameters) / float(baseline_parameters)


def _failure(row: Metrics, exc: Exception) -> None:
    row["status"] = "failed"
    row["error"] = f"{type(exc).__name__}: {exc}"


def _skip(row: Metrics, reason: str) -> None:
    if row["status"] == "planned":
        row["status"] = "skipped"
        row["reason"] = reason
        row["error"] = reason


def _write_screening_artifacts(output_dir: Path, config_path: Path, rows: list[Metrics]) -> None:
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


def run_cluster_screening(
    config_path: str | Path,
    dry_run: bool = False,
    adapters: ClusterScreeningAdapters | None = None,
) -> list[Metrics]:
    """Compare configured cluster sizes on sensitivity-approved layers."""

    config, checkpoint, output_dir, resolved_config_path = _resolve_config(Path(config_path))
    _validate_screening_config(config)
    rows = _planned_rows(config)
    if dry_run:
        _write_screening_artifacts(output_dir, resolved_config_path, rows)
        return rows

    active = adapters or ClusterScreeningAdapters.defaults(config)
    device = str(config["runtime"]["device"])
    baseline = rows[0]
    try:
        baseline_model = active.load_model(checkpoint)
        baseline_metrics = active.evaluate(baseline_model, config, device)
        baseline_stats = active.stats(baseline_model)
        _metrics_row(baseline, baseline_metrics, baseline_stats)
        baseline["status"] = "completed"
    except Exception as exc:
        _failure(baseline, exc)
        for row in rows[1:]:
            _skip(row, f"Skipped because baseline failed: {baseline['error']}")
        _write_screening_artifacts(output_dir, resolved_config_path, rows)
        return rows

    try:
        valid_layers = set(active.safe_layers(baseline_model, config))
    except Exception as exc:
        for row in rows[1:]:
            _skip(row, f"Skipped because candidate-layer screening failed: {type(exc).__name__}: {exc}")
        _write_screening_artifacts(output_dir, resolved_config_path, rows)
        return rows

    for row in rows[1:]:
        layer = str(row["layer"])
        if layer not in valid_layers:
            _skip(row, f"Candidate layer {layer!r} is not structurally prunable.")
            continue
        try:
            model = active.load_model(checkpoint)
            candidate = active.make_probe(
                model,
                layer,
                int(row["cluster_size"]),
                float(row["prune_ratio"]),
            )
            metrics = active.evaluate(candidate, config, device)
            stats = active.stats(candidate)
            _metrics_row(row, metrics, stats, baseline)
            row["status"] = "completed"
        except Exception as exc:
            _failure(row, exc)

    _write_screening_artifacts(output_dir, resolved_config_path, rows)
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
    configured = config["pruning"].get("candidate_layers", config["pruning"].get("safe_layers"))
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


def _collect_stats(model: Any) -> Metrics:
    from infrared_detection.evaluation.model_stats import collect_model_stats

    return collect_model_stats(_unwrap_model(model))
