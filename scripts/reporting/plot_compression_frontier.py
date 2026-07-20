"""Plot compression experiment trade-offs from a CSV report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from infrared_detection.evaluation.frontier import pareto_optimal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    rows = frame.to_dict("records")
    frontier = pareto_optimal_rows(rows)
    frontier_names = {row["variant"] for row in frontier}

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        is_frontier = row["variant"] in frontier_names
        axis.scatter(row["latency_p50_ms"], row["map50_95"], s=80 if is_frontier else 45, marker="*" if is_frontier else "o")
        axis.annotate(row["variant"], (row["latency_p50_ms"], row["map50_95"]), fontsize=8)
    axis.set_xlabel("Median latency (ms)")
    axis.set_ylabel("mAP50-95")
    axis.set_title("Accuracy-latency compression frontier")
    axis.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
