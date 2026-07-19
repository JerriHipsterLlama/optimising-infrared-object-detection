"""Dispatch model-compression workflows through one explicit contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


def build_yolo_dependency_graph(model: "nn.Module", example_input: Any) -> Any:
    """Load the structured-pruning graph builder only when pruning is requested."""

    from infrared_detection.compression.pruning import build_yolo_dependency_graph as build_graph

    return build_graph(model, example_input)


def prune_yolo_channels(graph: Any, channel_masks: Mapping[str, Any]) -> "nn.Module":
    """Load structured-pruning operations only when pruning is requested."""

    from infrared_detection.compression.pruning import prune_yolo_channels as prune_channels

    return prune_channels(graph, channel_masks)


def quantize_checkpoint(model: "nn.Module", bit_width: int) -> dict[str, Any]:
    """Load quantization only when quantization is requested."""

    from infrared_detection.compression.quantization import quantize_checkpoint as quantize

    return quantize(model, bit_width)


def train_student_with_kd(teacher: "nn.Module", student: "nn.Module", dataloader: Any, config: Mapping[str, Any]) -> "nn.Module":
    """Load distillation only when distillation is requested."""

    from infrared_detection.compression.distillation import train_student_with_kd as train

    return train(teacher, student, dataloader, config)


def run_matrix(config_path: str | Path, dry_run: bool = False) -> list[dict[str, Any]]:
    """Load experiment-matrix planning only when matrix dispatch is requested."""

    from infrared_detection.training.compression_matrix import run_matrix as run

    return run(config_path, dry_run=dry_run)


def _load_serialized_object(path: Path) -> Any:
    """Load a trusted PyTorch-serialised input only for a requested CLI workflow."""

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_serialized_module(path: Path) -> "nn.Module":
    """Load a model module supplied to a CLI model-compression workflow."""

    from torch import nn

    model = _load_serialized_object(path)
    if not isinstance(model, nn.Module):
        raise ValueError(f"Expected --model/--teacher path to contain a torch.nn.Module: {path}")
    return model


def _load_channel_masks(path: Path) -> dict[str, Any]:
    """Read JSON channel masks into boolean tensors for structured pruning."""

    import torch

    with path.open("r", encoding="utf-8") as handle:
        raw_masks = json.load(handle)
    if not isinstance(raw_masks, Mapping):
        raise ValueError("--channel-masks must contain a JSON object mapping layer names to boolean masks.")
    return {str(name): torch.as_tensor(mask, dtype=torch.bool) for name, mask in raw_masks.items()}


def _prune(model: "nn.Module", config: Mapping[str, Any]) -> "nn.Module":
    backend = config.get("backend", "structured")
    if backend != "structured":
        raise ValueError("Only the structured pruning backend is supported by compress(); use fcpts APIs directly.")

    try:
        channel_masks = config["channel_masks"]
    except KeyError as exc:
        raise ValueError("Structured pruning requires a 'channel_masks' configuration entry.") from exc

    graph = config.get("graph")
    if graph is None:
        try:
            graph = build_yolo_dependency_graph(model, config["example_input"])
        except KeyError as exc:
            raise ValueError(
                "Structured pruning requires either a model-bound 'graph' or an 'example_input' configuration entry."
            ) from exc
    elif getattr(graph, "model", None) is not model:
        raise ValueError("The supplied pruning graph does not belong to the supplied model.")

    return prune_yolo_channels(graph, channel_masks)


def compress(model: "nn.Module", config: Mapping[str, Any]) -> "nn.Module":
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
    parser.add_argument("--model", type=Path, help="Trusted PyTorch-serialised student model (required except for matrix).")
    parser.add_argument("--bit-width", type=int, default=8, help="Quantization bit width.")
    parser.add_argument("--example-input", type=Path, help="Trusted serialised example tensor for structured pruning.")
    parser.add_argument("--channel-masks", type=Path, help="JSON layer-to-boolean-mask mapping for structured pruning.")
    parser.add_argument("--teacher", type=Path, help="Trusted PyTorch-serialised teacher model for distillation.")
    parser.add_argument("--dataloader", type=Path, help="Trusted serialised iterable of distillation batches.")
    args = parser.parse_args()

    if args.method == "matrix":
        if args.config is None:
            parser.error("--config is required for matrix execution.")
        compress(None, {"method": "matrix", "config_path": args.config, "dry_run": args.dry_run})
        return

    if args.model is None:
        parser.error("--model is required for prune, quantize, and distill execution.")

    model = _load_serialized_module(args.model)
    config: dict[str, Any] = {"method": args.method}
    if args.method == "quantize":
        config["bit_width"] = args.bit_width
    elif args.method == "prune":
        if args.example_input is None or args.channel_masks is None:
            parser.error("prune requires --example-input and --channel-masks.")
        config["example_input"] = _load_serialized_object(args.example_input)
        config["channel_masks"] = _load_channel_masks(args.channel_masks)
    elif args.method == "distill":
        if args.teacher is None or args.dataloader is None:
            parser.error("distill requires --teacher and --dataloader.")
        config["teacher"] = _load_serialized_module(args.teacher)
        config["dataloader"] = _load_serialized_object(args.dataloader)

    compress(model, config)


if __name__ == "__main__":
    main()
