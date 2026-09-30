"""Run config-driven pruning sensitivity evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen cluster sizes on sensitivity-approved YOLO layers.")
    parser.add_argument("--config", type=Path, required=True, help="Cluster-screening YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Plan candidates without accessing models or hardware.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.cluster_workflow import run_cluster_screening

    result = run_cluster_screening(args.config, dry_run=args.dry_run)

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
