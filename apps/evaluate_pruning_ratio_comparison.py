"""Compare existing cluster-pruned checkpoints against a baseline ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export existing pruning checkpoints and compare ONNX accuracy, size, and latency."
    )
    parser.add_argument("--config", type=Path, required=True, help="Pruning-ratio comparison YAML.")
    parser.add_argument("--dry-run", action="store_true", help="List checkpoint and baseline inputs without running inference.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.pruning_ratio_comparison import run_pruning_ratio_comparison

    rows = run_pruning_ratio_comparison(args.config, dry_run=args.dry_run)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
