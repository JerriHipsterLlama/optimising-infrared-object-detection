"""Compression experiment-matrix planning and artifact generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from infrared_detection.common.experiment_config import load_experiment_config, resolve_repo_path

VARIANTS = {
    "dense_fp32",
    "dense_fp16",
    "lpq_int8",
    "structured_pruning",
    "structured_pruning_kd",
    "lpq_int8_kd",
    "structured_pruning_lpq_int8_kd",
}


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_variant(name: str, base_checkpoint: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Create reproducible artifacts for one configured compression variant."""

    if name not in VARIANTS:
        raise ValueError(f"Unknown compression variant: {name}")
    output_dir = Path(config.get("experiment", {}).get("output_dir", "artifacts/experiments")) / name
    result = {
        "variant": name,
        "base_checkpoint": str(Path(base_checkpoint)),
        "config_hash": _config_hash(config),
        "status": "dry_run" if config.get("dry_run") else "planned",
        "artifact_dir": str(output_dir.resolve()),
    }
    _write_json(output_dir / "manifest.json", result)
    _write_json(output_dir / "metrics.json", {"variant": name, "status": result["status"]})
    return result


def run_matrix(config_path: str | Path, dry_run: bool = False) -> list[dict[str, Any]]:
    """Plan every configured compression variant without modifying the model."""

    config = load_experiment_config(config_path)
    config["dry_run"] = dry_run
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = resolve_repo_path(config["model"]["checkpoint"], repo_root)
    return [run_variant(name, checkpoint, config) for name in config.get("variants", [])]
