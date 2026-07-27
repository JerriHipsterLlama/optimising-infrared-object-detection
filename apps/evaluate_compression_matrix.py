"""Run the local RTX compression precision matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and structured-pruned RTX precision variants.")
    parser.add_argument("--config", type=Path, required=True, help="Compression-matrix YAML configuration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan matrix rows without loading models or requiring TensorRT.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from infrared_detection.evaluation.compression_matrix import run_compression_matrix

    print(json.dumps(run_compression_matrix(args.config, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
