"""Create structured-pruned PyTorch checkpoints from a YAML experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster-pruned .pt checkpoints; does not export ONNX.")
    parser.add_argument("--config", type=Path, required=True, help="Cluster-pruning YAML configuration.")
    parser.add_argument("--dry-run", action="store_true", help="List configured pruning ratios without loading a model.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.cluster_pruning_workflow import run_cluster_pruning

    print(json.dumps(run_cluster_pruning(args.config, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
