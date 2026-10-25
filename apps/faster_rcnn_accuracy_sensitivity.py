"""Run Faster R-CNN structural-pruning accuracy sensitivity screening."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from infrared_detection.evaluation.accuracy_sensitivity import run_accuracy_sensitivity


def _number(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "-" if value in (None, "") else f"{float(value):.4f}"


def print_progress(row: dict[str, Any]) -> None:
    if row.get("status") == "BASELINE":
        print(
            f"[BASELINE] mAP50={_number(row, 'map50')} "
            f"mAP50-95={_number(row, 'map50_95')} "
            f"recall={_number(row, 'recall')} "
            f"inference={_number(row, 'inference_ms_per_image')} ms/image"
        )
        return
    print(
        f"[{row.get('status', '-')}] {row.get('architectural_region', '-')} "
        f"{row.get('pruning_unit', '-')} requested={_number(row, 'requested_pruning_ratio')} "
        f"actual={_number(row, 'actual_pruning_ratio')} "
        f"mAP50={_number(row, 'map50')} "
        f"mAP50-95={_number(row, 'map50_95')} "
        f"recall={_number(row, 'recall')} "
        f"inference={_number(row, 'inference_ms_per_image')} ms/image "
        f"mAP-sensitivity={_number(row, 'point_sensitivity')} "
        f"recall-sensitivity={_number(row, 'recall_sensitivity')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen Faster R-CNN structural-pruning accuracy sensitivity."
    )
    parser.add_argument("--config", required=True, help="Screening experiment YAML.")
    args = parser.parse_args()
    run_accuracy_sensitivity(args.config, on_result=print_progress)


if __name__ == "__main__":
    main()
