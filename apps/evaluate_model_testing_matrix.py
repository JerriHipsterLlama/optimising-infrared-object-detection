"""Evaluate precision-specific ONNX artifacts with configured backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test discovered precision-specific ONNX artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Model testing matrix YAML configuration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover ONNX files and write planned rows without inference or engine building.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.model_testing_matrix import run_model_testing_matrix

    print(json.dumps(run_model_testing_matrix(args.config, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
