"""Dispatch model-compression workflows through one explicit contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from infrared_detection.compression.distillation import train_student_with_kd
from infrared_detection.compression.pruning import prune_yolo_channels
from infrared_detection.compression.quantization import quantize_checkpoint
from infrared_detection.training.compression_matrix import run_matrix


def _prune(model: nn.Module, config: Mapping[str, Any]) -> nn.Module:
    backend = config.get("backend", "structured")
    if backend != "structured":
        raise ValueError("Only the structured pruning backend is supported by compress(); use fcpts APIs directly.")
    try:
        return prune_yolo_channels(config["graph"], config["channel_masks"])
    except KeyError as exc:
        raise ValueError("Structured pruning requires 'graph' and 'channel_masks' configuration entries.") from exc


def compress(model: nn.Module, config: Mapping[str, Any]) -> nn.Module:
    """Apply the requested compression method and return the resulting model.

    ``method`` must be one of ``prune``, ``quantize``, ``distill``, or
    ``matrix``. Quantization stores its serialisable payload on the returned
    model as ``_compression_quantization`` because checkpoint quantization
    does not alter the in-memory floating-point module.
    """

    method = config.get("method")
    if method == "prune":
        return _prune(model, config)
    if method == "quantize":
        model._compression_quantization = quantize_checkpoint(model, int(config.get("bit_width", 8)))
        return model
    if method == "distill":
        try:
            return train_student_with_kd(config["teacher"], model, config["dataloader"], config)
        except KeyError as exc:
            raise ValueError("Distillation requires 'teacher' and 'dataloader' configuration entries.") from exc
    if method == "matrix":
        try:
            run_matrix(config["config_path"], dry_run=bool(config.get("dry_run", False)))
        except KeyError as exc:
            raise ValueError("Matrix dispatch requires a 'config_path' configuration entry.") from exc
        return model
    raise ValueError("Compression method must be one of: prune, quantize, distill, matrix.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-compression workflow.")
    parser.add_argument("method", choices=("prune", "quantize", "distill", "matrix"))
    parser.add_argument("--config", type=Path, help="Compression-matrix YAML path (required for matrix).")
    parser.add_argument("--dry-run", action="store_true", help="Plan the compression matrix without running experiments.")
    args = parser.parse_args()
    if args.method != "matrix":
        parser.error("CLI execution currently supports 'matrix'; use compress(model, config) for model workflows.")
    if args.config is None:
        parser.error("--config is required for matrix execution.")
    run_matrix(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
