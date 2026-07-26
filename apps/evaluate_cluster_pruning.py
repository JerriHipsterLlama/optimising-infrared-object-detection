"""Run config-driven pruning sensitivity evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structurally pruned YOLO candidates.")
    parser.add_argument("--config", type=Path, required=True, help="Pruning evaluation YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Plan candidates without accessing models or hardware.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--screen-only",
        action="store_true",
        help="Run baseline and RTX probes only; skip global candidates and Orin winner selection.",
    )
    mode.add_argument(
        "--filterwise",
        action="store_true",
        help="Run the 1..N single-layer filter-removal sensitivity sweep on the screening device.",
    )
    parser.add_argument(
        "--full-curve",
        action="store_true",
        help="For --filterwise, continue to one remaining filter instead of early stopping.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.cluster_workflow import run_cluster_evaluation, run_filterwise_evaluation

    if args.filterwise:
        result = run_filterwise_evaluation(args.config, dry_run=args.dry_run, full_curve=args.full_curve)
    else:
        result = run_cluster_evaluation(args.config, dry_run=args.dry_run, screen_only=args.screen_only)

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
