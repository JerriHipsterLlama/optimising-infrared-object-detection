"""Planning and artifact writing for RTX compression experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from infrared_detection.compression.pruning import compute_channel_importance
from infrared_detection.compression.pruning.cluster_selection import plan_low_importance_clusters
from infrared_detection.compression.pruning.cluster_probe import run_filterwise_probe
from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


OFFICIAL_PRECISIONS = ("fp32", "fp16", "int8")


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
    evidence_layer = pruning.get("candidate_layer")
    filters_removed = pruning.get("filter_removal_count")
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
        raise ValueError("Pruned checkpoint build requires pruning.candidate_layer")
    if not isinstance(filters_removed, int) or isinstance(filters_removed, bool) or filters_removed <= 0:
        raise ValueError("Pruned checkpoint build requires a positive pruning.filter_removal_count")
    if not isinstance(candidate_layers, list) or not candidate_layers or not all(
        isinstance(layer, str) and layer for layer in candidate_layers
    ):
        raise ValueError("Pruned checkpoint build requires non-empty pruning.candidate_layers")
    if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
        raise ValueError("Pruned checkpoint build requires a positive pruning.cluster_size")
    if importance != "minimum_weight":
        raise ValueError("Pruned checkpoint build requires pruning.importance='minimum_weight'")
    if filters_removed != cluster_size:
        raise ValueError("Filterwise evidence removal count must equal pruning.cluster_size")
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

    manifest_path = Path(manifest_value)
    candidate = select_filterwise_candidate(manifest_path, evidence_layer, filters_removed)
    evidence_checkpoint = _resolve_candidate_checkpoint(manifest_path, candidate["checkpoint_path"])
    if not evidence_checkpoint.is_file():
        raise FileNotFoundError(f"Selected filterwise evidence checkpoint does not exist: {evidence_checkpoint}")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping) or not isinstance(model_config.get("checkpoint"), str):
        raise ValueError("Pruned checkpoint build requires model.checkpoint")
    source_checkpoint = Path(model_config["checkpoint"])
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
    if precisions != list(OFFICIAL_PRECISIONS):
        raise ValueError(
            "Official precisions matrix must be exactly the ordered list "
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
    candidate_id = pruning.get("candidate_id")
    if pruning_enabled and not isinstance(candidate_id, str):
        raise ValueError("Enabled pruning requires pruning.candidate_id")
    if pruning_enabled:
        filterwise_manifest = pruning.get("filterwise_manifest")
        candidate_layer = pruning.get("candidate_layer")
        candidate_layers = pruning.get("candidate_layers")
        cluster_size = pruning.get("cluster_size")
        prune_ratios = _validate_prune_ratios(pruning.get("prune_ratios"), "Enabled pruning")
        filter_removal_count = pruning.get("filter_removal_count")
        allowed_map50_95_drop = _validate_allowed_map_drop(
            pruning.get("allowed_map50_95_drop"), "Enabled pruning"
        )
        importance = pruning.get("importance")
        if not isinstance(filterwise_manifest, str) or not filterwise_manifest:
            raise ValueError("Enabled pruning requires pruning.filterwise_manifest")
        if not isinstance(candidate_layer, str) or not candidate_layer:
            raise ValueError("Enabled pruning requires pruning.candidate_layer")
        if not isinstance(candidate_layers, list) or not candidate_layers or not all(
            isinstance(layer, str) and layer for layer in candidate_layers
        ):
            raise ValueError("Enabled pruning requires non-empty pruning.candidate_layers")
        if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
            raise ValueError("Enabled pruning requires a positive integer pruning.cluster_size")
        if not isinstance(filter_removal_count, int) or isinstance(filter_removal_count, bool) or filter_removal_count <= 0:
            raise ValueError("Enabled pruning requires a positive pruning.filter_removal_count")
        if filter_removal_count != cluster_size:
            raise ValueError("Enabled pruning requires evidence removal count equal to pruning.cluster_size")
        if importance != "minimum_weight":
            raise ValueError("Enabled pruning requires pruning.importance='minimum_weight'")

    variant_specs = [("dense", None)]
    if pruning_enabled:
        variant_specs.append((candidate_id, candidate_id))

    rows: list[dict[str, Any]] = []
    for variant_name, pruning_candidate_id in variant_specs:
        for precision in precisions:
            row = {
                    "variant_id": f"{variant_name}-{precision}",
                    "compression": "dense" if pruning_candidate_id is None else "structured_pruning",
                    "precision": precision,
                    "pruning_candidate_id": pruning_candidate_id,
                    "status": "planned",
                    "error": None,
                }
            if pruning_candidate_id is None:
                row["provenance"] = {"planner": "compression_matrix", "stage": "planning"}
            else:
                provenance = {
                    "filterwise_manifest": filterwise_manifest,
                    "candidate_layer": candidate_layer,
                    "candidate_layers": candidate_layers,
                    "cluster_size": cluster_size,
                    "filter_removal_count": filter_removal_count,
                    "prune_ratios": prune_ratios,
                    "allowed_map50_95_drop": allowed_map50_95_drop,
                    "importance": importance,
                }
                row.update(
                    {
                        "filterwise_manifest": filterwise_manifest,
                        "candidate_layer": candidate_layer,
                        "candidate_layers": candidate_layers,
                        "cluster_size": cluster_size,
                        "prune_ratios": prune_ratios,
                        "allowed_map50_95_drop": allowed_map50_95_drop,
                        "filter_removal_count": filter_removal_count,
                        "importance": importance,
                        "provenance": provenance,
                    }
                )
            rows.append(row)
    return rows


def write_compression_manifest(output_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write manifest and CSV checkpoints while retaining prior partial rows."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    existing_rows: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_rows = {
                str(row["variant_id"]): dict(row)
                for row in payload.get("rows", [])
                if isinstance(row, Mapping) and "variant_id" in row
            }
        except (OSError, TypeError, ValueError, AttributeError):
            existing_rows = {}

    for row in rows:
        if "variant_id" not in row:
            raise ValueError("Every compression manifest row requires variant_id")
        existing_rows[str(row["variant_id"])] = dict(row)

    merged_rows = list(existing_rows.values())
    write_experiment_manifest(manifest_path, {"rows": merged_rows})
    write_metrics_csv(output_dir / "results.csv", merged_rows)
