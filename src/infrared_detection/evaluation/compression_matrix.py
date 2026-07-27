"""Planning and artifact writing for RTX compression experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from infrared_detection.evaluation.artifacts import write_experiment_manifest, write_metrics_csv


OFFICIAL_PRECISIONS = ("fp32", "fp16", "int8")


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
        cluster_size = pruning.get("cluster_size")
        pruning_ratio = pruning.get("pruning_ratio")
        filter_removal_count = pruning.get("filter_removal_count")
        if not isinstance(filterwise_manifest, str) or not filterwise_manifest:
            raise ValueError("Enabled pruning requires pruning.filterwise_manifest")
        if not isinstance(candidate_layer, str) or not candidate_layer:
            raise ValueError("Enabled pruning requires pruning.candidate_layer")
        if not isinstance(cluster_size, int) or isinstance(cluster_size, bool) or cluster_size <= 0:
            raise ValueError("Enabled pruning requires a positive integer pruning.cluster_size")
        has_ratio = isinstance(pruning_ratio, (int, float)) and not isinstance(pruning_ratio, bool)
        has_count = isinstance(filter_removal_count, int) and not isinstance(filter_removal_count, bool)
        if has_ratio == has_count or (has_ratio and not 0 < pruning_ratio < 1) or (has_count and filter_removal_count <= 0):
            raise ValueError(
                "Enabled pruning requires exactly one valid pruning_ratio or filter_removal_count"
            )

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
                    "cluster_size": cluster_size,
                }
                if pruning_ratio is not None:
                    provenance["pruning_ratio"] = pruning_ratio
                else:
                    provenance["filter_removal_count"] = filter_removal_count
                row.update(
                    {
                        "filterwise_manifest": filterwise_manifest,
                        "candidate_layer": candidate_layer,
                        "cluster_size": cluster_size,
                        "pruning_ratio": pruning_ratio,
                        "filter_removal_count": filter_removal_count,
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
