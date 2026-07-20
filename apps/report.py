"""Generate a Pareto-optimal subset from experiment rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infrared_detection.evaluation.artifacts import write_metrics
from infrared_detection.evaluation.frontier import pareto_optimal_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Find Pareto-optimal compression variants.")
    parser.add_argument("--input", type=Path, required=True, help="JSON array of experiment rows.")
    parser.add_argument("--output", type=Path, default=Path("reports/pareto.json"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Report input must be a JSON array of experiment rows.")
    write_metrics(args.output, {"pareto_optimal": pareto_optimal_rows(rows)})


if __name__ == "__main__":
    main()
