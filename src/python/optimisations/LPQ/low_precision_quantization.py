"""Low precision post-training quantization utilities.

This module quantizes floating point tensors in a PyTorch checkpoint or state
dictionary into signed integer tensors with per-tensor symmetric scaling.

Supported bit widths are 32, 16, 8, 6, 4, and 2. For 6/4/2 bit quantization,
the values are stored in the smallest practical signed integer container
(int8) while the bit width is preserved in metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import torch


SUPPORTED_BIT_WIDTHS = (32, 16, 8, 4, 2)


@dataclass(frozen=True)
class TensorQuantizationStats:
    """Summary statistics for one quantized tensor."""

    name: str
    bit_width: int
    original_dtype: str
    storage_dtype: str
    shape: Tuple[int, ...]
    scale: float
    zero_point: int
    original_numel: int
    original_bytes: int
    quantized_bytes: int
    mse: float
    max_abs_error: float


def _storage_dtype(bit_width: int) -> torch.dtype:
    """Return the integer storage dtype used for a requested bit width."""

    if bit_width <= 8:
        return torch.int8
    if bit_width <= 16:
        return torch.int16
    return torch.int32


def _quantization_range(bit_width: int) -> Tuple[int, int]:
    """Return signed integer quantization bounds for a bit width."""

    qmin = -(1 << (bit_width - 1))
    qmax = (1 << (bit_width - 1)) - 1
    return qmin, qmax


def _is_tensor_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping) and any(torch.is_tensor(value) for value in obj.values())


def load_checkpoint(checkpoint_path: str | Path) -> Any:
    """Load a checkpoint with CPU mapping.

    The helper tolerates both modern and older torch.load signatures.
    """

    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, Any], str | None]:
    """Extract a state dict from a checkpoint or model object.

    Returns the extracted mapping and the key it came from if applicable.
    """

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict(), None

    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if _is_tensor_mapping(candidate):
                return dict(candidate), key

    if _is_tensor_mapping(checkpoint):
        return dict(checkpoint), None

    raise TypeError(
        "Unsupported checkpoint format. Expected a torch.nn.Module, a raw state dict, "
        "or a checkpoint containing a tensor mapping under 'state_dict'/'model_state_dict'."
    )


def quantize_tensor(tensor: torch.Tensor, bit_width: int) -> Tuple[torch.Tensor, float, int, Dict[str, float]]:
    """Quantize a floating point tensor into a signed integer tensor.

    Uses symmetric per-tensor quantization with zero-point fixed at 0.
    """

    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"Unsupported bit width: {bit_width}. Expected one of {SUPPORTED_BIT_WIDTHS}.")

    if not torch.is_floating_point(tensor):
        raise TypeError("quantize_tensor expects a floating point tensor.")

    qmin, qmax = _quantization_range(bit_width)
    storage_dtype = _storage_dtype(bit_width)
    source = tensor.detach().to(torch.float64)
    max_abs = float(source.abs().max().item())

    if math.isclose(max_abs, 0.0):
        scale = 1.0
        quantized = torch.zeros_like(source, dtype=storage_dtype)
    else:
        scale = max_abs / qmax
        quantized = torch.clamp(torch.round(source / scale), qmin, qmax).to(storage_dtype)

    reconstructed = quantized.to(torch.float64) * scale
    error = reconstructed - source
    mse = float(torch.mean(error * error).item())
    max_abs_error = float(error.abs().max().item())

    metrics = {
        "mse": mse,
        "max_abs_error": max_abs_error,
    }
    return quantized, scale, 0, metrics


def dequantize_tensor(quantized: torch.Tensor, scale: float, zero_point: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize an integer tensor back to a floating point tensor."""

    if dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"Unsupported floating point dtype: {dtype}")

    restored = (quantized.to(torch.float64) - zero_point) * scale
    return restored.to(dtype)


def quantize_state_dict(state_dict: Mapping[str, Any], bit_width: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, TensorQuantizationStats]]:
    """Quantize floating point tensors in a state dict.

    Non-floating tensors are preserved unchanged.
    """

    quantized_state_dict: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {
        "bit_width": bit_width,
        "quantization_scheme": "symmetric_per_tensor",
        "storage_dtype": str(_storage_dtype(bit_width)).replace("torch.", ""),
    }
    tensor_stats: Dict[str, TensorQuantizationStats] = {}

    total_original_bytes = 0
    total_quantized_bytes = 0

    for name, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            quantized, scale, zero_point, metrics = quantize_tensor(value, bit_width)
            quantized_state_dict[name] = quantized

            original_bytes = value.numel() * value.element_size()
            quantized_bytes = quantized.numel() * quantized.element_size()
            total_original_bytes += original_bytes
            total_quantized_bytes += quantized_bytes

            tensor_stats[name] = TensorQuantizationStats(
                name=name,
                bit_width=bit_width,
                original_dtype=str(value.dtype).replace("torch.", ""),
                storage_dtype=str(quantized.dtype).replace("torch.", ""),
                shape=tuple(value.shape),
                scale=scale,
                zero_point=zero_point,
                original_numel=value.numel(),
                original_bytes=original_bytes,
                quantized_bytes=quantized_bytes,
                mse=metrics["mse"],
                max_abs_error=metrics["max_abs_error"],
            )
            metadata[name] = {
                "original_dtype": str(value.dtype).replace("torch.", ""),
                "storage_dtype": str(quantized.dtype).replace("torch.", ""),
                "shape": list(value.shape),
                "scale": scale,
                "zero_point": zero_point,
                "is_quantized": True,
            }
        else:
            quantized_state_dict[name] = value.clone() if torch.is_tensor(value) else value
            if torch.is_tensor(value):
                original_bytes = value.numel() * value.element_size()
                total_original_bytes += original_bytes
                total_quantized_bytes += original_bytes
                metadata[name] = {
                    "original_dtype": str(value.dtype).replace("torch.", ""),
                    "storage_dtype": str(value.dtype).replace("torch.", ""),
                    "shape": list(value.shape),
                    "scale": None,
                    "zero_point": None,
                    "is_quantized": False,
                }
            else:
                metadata[name] = {
                    "is_quantized": False,
                }

    summary = {
        "bit_width": bit_width,
        "quantized_tensors": len(tensor_stats),
        "original_bytes": total_original_bytes,
        "quantized_bytes": total_quantized_bytes,
        "compression_ratio": (total_original_bytes / total_quantized_bytes) if total_quantized_bytes else None,
        "average_mse": (
            sum(stats.mse for stats in tensor_stats.values()) / len(tensor_stats)
            if tensor_stats
            else None
        ),
        "max_abs_error": max((stats.max_abs_error for stats in tensor_stats.values()), default=None),
    }
    metadata["summary"] = summary

    return quantized_state_dict, metadata, tensor_stats


def dequantize_state_dict(
    quantized_state_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reconstruct a floating point state dict from quantized tensors."""

    restored: Dict[str, Any] = {}

    for name, value in quantized_state_dict.items():
        entry = metadata.get(name, {})
        if torch.is_tensor(value) and entry.get("is_quantized"):
            target_dtype_name = entry.get("original_dtype", "float32")
            target_dtype = getattr(torch, target_dtype_name, torch.float32)
            restored[name] = dequantize_tensor(
                value,
                float(entry["scale"]),
                int(entry.get("zero_point", 0)),
                target_dtype,
            )
        else:
            restored[name] = value.clone() if torch.is_tensor(value) else value

    return restored


def quantize_checkpoint(
    checkpoint: Any,
    bit_width: int,
) -> Dict[str, Any]:
    """Quantize a checkpoint and return a serialisable payload."""

    state_dict, source_key = extract_state_dict(checkpoint)
    quantized_state_dict, metadata, tensor_stats = quantize_state_dict(state_dict, bit_width)

    payload = {
        "format": "low_precision_quantized_checkpoint",
        "source_key": source_key,
        "bit_width": bit_width,
        "quantized_state_dict": quantized_state_dict,
        "metadata": metadata,
        "tensor_stats": {name: stats.__dict__ for name, stats in tensor_stats.items()},
    }
    return payload


def save_quantized_checkpoint(payload: Dict[str, Any], output_path: str | Path) -> Path:
    """Save a quantized checkpoint payload to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def quantize_file(input_path: str | Path, output_dir: str | Path, bit_widths: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """Quantize one checkpoint into several bit-width variants."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(input_path)
    results: Dict[int, Dict[str, Any]] = {}

    for bit_width in bit_widths:
        payload = quantize_checkpoint(checkpoint, bit_width)
        output_path = output_dir / f"{input_path.stem}_int{bit_width}.pt"
        save_quantized_checkpoint(payload, output_path)

        summary = dict(payload["metadata"]["summary"])
        summary["output_path"] = str(output_path)
        results[int(bit_width)] = summary

    return results


def _parse_bit_widths(raw_values: list[str] | None) -> list[int]:
    if not raw_values:
        return list(SUPPORTED_BIT_WIDTHS)

    bit_widths = [int(value) for value in raw_values]
    invalid = [value for value in bit_widths if value not in SUPPORTED_BIT_WIDTHS]
    if invalid:
        raise ValueError(f"Unsupported bit widths: {invalid}. Expected values from {SUPPORTED_BIT_WIDTHS}.")
    return bit_widths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a floating point PyTorch checkpoint to INT_X variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to a .pt/.pth checkpoint or state dict.")
    parser.add_argument("--output-dir", required=True, help="Directory for quantized checkpoints and summary.")
    parser.add_argument(
        "--bits",
        nargs="*",
        default=None,
        help="Bit widths to generate. Supported values: 32 16 8 6 4 2.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional path for the JSON summary file. Defaults to <output-dir>/quantization_summary.json.",
    )
    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="Also save a dequantized float checkpoint next to each quantized checkpoint.",
    )

    args = parser.parse_args()
    bit_widths = _parse_bit_widths(args.bits)

    results = quantize_file(args.input, args.output_dir, bit_widths)
    output_dir = Path(args.output_dir)

    if args.dequantize:
        checkpoint = load_checkpoint(args.input)
        state_dict, _ = extract_state_dict(checkpoint)

        for bit_width in bit_widths:
            payload = quantize_checkpoint(checkpoint, bit_width)
            restored_state_dict = dequantize_state_dict(payload["quantized_state_dict"], payload["metadata"])
            restored_path = output_dir / f"{Path(args.input).stem}_int{bit_width}_dequantized.pt"
            torch.save(restored_state_dict, restored_path)
            results[bit_width]["dequantized_path"] = str(restored_path)

    summary_path = Path(args.summary_file) if args.summary_file else output_dir / "quantization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Quantization complete. Summary written to {summary_path}")
    for bit_width in bit_widths:
        entry = results[bit_width]
        print(
            f"INT{bit_width}: tensors={entry['quantized_tensors']}, "
            f"compression_ratio={entry['compression_ratio']}, output={entry['output_path']}"
        )


if __name__ == "__main__":
    main()
