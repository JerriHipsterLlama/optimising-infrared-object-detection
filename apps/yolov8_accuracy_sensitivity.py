"""Run standalone YOLOv8n structural-pruning accuracy sensitivity screening."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from infrared_detection.evaluation.accuracy_sensitivity import run_accuracy_sensitivity


def _number(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "-" if value in (None, "") else f"{float(value):.4f}"


def print_progress(row: dict[str, Any]) -> None:
    """Print one concise, blank-safe screening update."""

    if row.get("status") == "BASELINE":
        print(f"[BASELINE] mAP50-95={_number(row, 'map50_95')}")
        return
    print(
        f"[{row.get('status', '-')}] {row.get('architectural_region', '-')} "
        f"{row.get('pruning_unit', '-')} requested={_number(row, 'requested_pruning_ratio')} "
        f"actual={_number(row, 'actual_pruning_ratio')} "
        f"mAP50-95={_number(row, 'map50_95')} "
        f"sensitivity={_number(row, 'point_sensitivity')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen YOLOv8n structural-pruning accuracy sensitivity."
    )
    parser.add_argument(
        "--config", required=True, help="Sensitivity experiment YAML configuration."
    )
    args = parser.parse_args()
    run_accuracy_sensitivity(args.config, on_result=print_progress)


if __name__ == "__main__":
    main()
