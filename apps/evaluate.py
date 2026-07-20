"""Evaluate and normalize detection metrics into a JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infrared_detection.evaluation.artifacts import write_metrics
from infrared_detection.evaluation.detection_metrics import summarize_detection_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize detection metrics into a project JSON schema.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing evaluator output.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions/metrics.json"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    write_metrics(args.output, summarize_detection_metrics(payload))


if __name__ == "__main__":
    main()
