"""Run the config-driven two-stage cluster-pruning evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structurally pruned YOLO cluster candidates.")
    parser.add_argument("--config", type=Path, required=True, help="Cluster-pruning evaluation YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Plan candidates without accessing models or hardware.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.cluster_workflow import run_cluster_evaluation

    print(json.dumps(run_cluster_evaluation(args.config, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
