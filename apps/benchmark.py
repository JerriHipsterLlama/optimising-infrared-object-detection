"""Run the Python-side TensorRT benchmark adapter on a Jetson."""

from __future__ import annotations

import argparse
from pathlib import Path

from infrared_detection.benchmarking.jetson import benchmark_tensorrt_engine
from infrared_detection.evaluation.artifacts import write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a prebuilt TensorRT engine with trtexec.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions/benchmark.json"))
    args = parser.parse_args()
    result = benchmark_tensorrt_engine(args.engine, iterations=args.iterations, warmup=args.warmup)
    write_metrics(args.output, result)


if __name__ == "__main__":
    main()
