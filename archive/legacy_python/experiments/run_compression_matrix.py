"""Run comparable compression variants and emit experiment manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.evaluation.artifacts import write_experiment_manifest, write_metrics
from src.python.experiments.config import load_experiment_config, resolve_repo_path


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
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_variant(name: str, base_checkpoint: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one variant; dry-run mode is always available without a checkpoint."""

    if name not in VARIANTS:
        raise ValueError(f"Unknown compression variant: {name}")
    output_dir = Path(config.get("experiment", {}).get("output_dir", "runs/experiments")) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": name,
        "base_checkpoint": str(Path(base_checkpoint)),
        "config_hash": _config_hash(config),
        "status": "dry_run" if config.get("dry_run") else "planned",
        "artifact_dir": str(output_dir.resolve()),
    }
    write_experiment_manifest(output_dir / "manifest.json", result)
    write_metrics(output_dir / "metrics.json", {"variant": name, "status": result["status"]})
    return result


def run_matrix(config_path: str | Path, dry_run: bool = False) -> list[dict[str, Any]]:
    config = load_experiment_config(config_path)
    config["dry_run"] = dry_run
    checkpoint = resolve_repo_path(config["model"]["checkpoint"], REPO_ROOT)
    return [run_variant(name, checkpoint, config) for name in config.get("variants", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compression experiment matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = run_matrix(args.config, dry_run=args.dry_run)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
