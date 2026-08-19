from __future__ import annotations

import argparse

import yaml

from infrared_detection.evaluation.single_layer_performance_screening import (
    ScreeningAdapters,
    run_single_layer_performance_screening,
)


def _progress(row: dict) -> None:
    if row.get("status") != "completed" or row.get("stage") != "single_layer":
        return
    print(
        f"[{row['model_variant']}/{row['hardware']}] {row['layer']} rank "
        f"{row['filter_rank']}/{row['filters_before'] - 1} | remaining={row['filters_remaining']} | "
        f"mAP50-95={row['map50_95']:.4f} | p50={row['latency_p50_ms']:.3f} ms | completed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-layer performance screening.")
    parser.add_argument("--config", required=True, help="Experiment YAML configuration.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_single_layer_performance_screening(
        args.config,
        adapters=ScreeningAdapters.defaults(config),
        on_result=_progress,
    )


if __name__ == "__main__":
    main()
