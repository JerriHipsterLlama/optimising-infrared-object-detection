# Low Precision Quantization: Step-Through Guide

This guide explains what each function in this module does, in the order you would encounter them while running the script.

## 1. What The Script Is For

The script converts floating-point PyTorch model weights (typically `float32` or `float64`) into low-precision integer formats (`INT32`, `INT16`, `INT8`, `INT6`, `INT4`, `INT2`) using symmetric per-tensor quantization.

Main goals:
- Reduce model storage size.
- Provide a baseline comparison across multiple integer precisions.
- Keep enough metadata to dequantize back to float for error analysis.

## 2. High-Level Runtime Flow

When run from CLI, the call path is:
1. `main()` parses CLI arguments.
2. `_parse_bit_widths()` validates requested integer bit widths.
3. `quantize_file()` loads checkpoint and loops over bit widths.
4. `quantize_checkpoint()` extracts model weights and quantizes them.
5. `quantize_state_dict()` walks tensor-by-tensor and quantizes floating tensors.
6. `quantize_tensor()` applies core numeric quantization math.
7. `save_quantized_checkpoint()` writes each quantized artifact.
8. Optional: `dequantize_state_dict()` + `dequantize_tensor()` restore float tensors for analysis.

---

## 3. Constants And Data Structures

### `SUPPORTED_BIT_WIDTHS = (32, 16, 8, 6, 4, 2)`
Defines the only allowed target precisions.

### `TensorQuantizationStats` (dataclass)
Holds per-tensor statistics after quantization:
- Tensor name, shape, original dtype, storage dtype.
- Scale and zero-point.
- Size before/after quantization.
- Quantization error metrics (`mse`, `max_abs_error`).

---

## 4. Helper Functions (Low-Level)

### `_storage_dtype(bit_width)`
Maps logical bit width to actual PyTorch storage dtype:
- `<= 8` -> `torch.int8`
- `<= 16` -> `torch.int16`
- otherwise -> `torch.int32`

Why this matters:
- PyTorch does not have native `int6`, `int4`, or `int2` tensor dtypes.
- So `INT6/4/2` values are stored in `int8`, but clipped to the proper numeric range.

### `_quantization_range(bit_width)`
Computes signed integer bounds:
- `qmin = -(2^(bit_width - 1))`
- `qmax = 2^(bit_width - 1) - 1`

Example:
- `INT8` -> `[-128, 127]`
- `INT4` -> `[-8, 7]`

### `_is_tensor_mapping(obj)`
Checks whether an object is dict-like and contains tensors among its values.

Used to detect whether a loaded checkpoint looks like a state dict.

---

## 5. Checkpoint Loading And Extraction

### `load_checkpoint(checkpoint_path)`
Loads checkpoint from disk with CPU mapping.

Behavior:
- Tries `torch.load(..., weights_only=False)` first.
- Falls back to older signature if needed.

Goal:
- Compatible with different PyTorch versions/checkpoint styles.

### `extract_state_dict(checkpoint)`
Extracts model weights regardless of checkpoint format.

Supports:
- A `torch.nn.Module` object.
- A raw state dict.
- A wrapped training checkpoint with keys like:
  - `state_dict`
  - `model_state_dict`
  - `model`
  - `net`

Returns:
- The resolved state dict.
- The source key (or `None` when not nested).

---

## 6. Core Tensor Quantization

### `quantize_tensor(tensor, bit_width)`
Core numeric conversion from float tensor to integer tensor.

Steps:
1. Validate `bit_width` and ensure input is floating-point.
2. Get quantization range (`qmin`, `qmax`) and storage dtype.
3. Convert source tensor to `float64` for stable intermediate math.
4. Compute `max_abs = max(abs(tensor))`.
5. Compute scale:
   - If tensor is all zeros: `scale = 1.0`, output all zeros.
   - Else: `scale = max_abs / qmax`.
6. Quantize:
   - `round(source / scale)`
   - clamp to `[qmin, qmax]`
   - cast to storage dtype.
7. Reconstruct float approximation (`quantized * scale`) and compute errors:
   - Mean squared error (`mse`)
   - Maximum absolute error (`max_abs_error`)

Returns:
- Quantized integer tensor
- Scale
- Zero-point (always `0` for symmetric scheme)
- Metrics dict

### `dequantize_tensor(quantized, scale, zero_point, dtype)`
Converts quantized integer tensor back to float:
- `(quantized - zero_point) * scale`
- Cast to target floating dtype (`float16/32/64/bfloat16`).

Used for analysis and optional dequantized artifact generation.

---

## 7. State Dict Quantization

### `quantize_state_dict(state_dict, bit_width)`
Applies quantization to every entry in a state dict.

Per key behavior:
- Floating tensor: quantize using `quantize_tensor()`.
- Non-floating tensor: copy unchanged.
- Non-tensor value: preserve as-is.

It also builds:
- `metadata` per tensor (`scale`, dtype info, quantized flag, shape).
- `tensor_stats` with detailed error and size metrics.
- `summary` totals:
  - number of quantized tensors
  - bytes before/after
  - compression ratio
  - average MSE
  - max absolute error across tensors

Returns:
- Quantized state dict
- Metadata dict
- Tensor stats dict

### `dequantize_state_dict(quantized_state_dict, metadata)`
Reconstructs a float state dict from quantized tensors using saved metadata.

Per key behavior:
- If marked quantized: use stored `scale`, `zero_point`, and original dtype.
- Else: copy unchanged.

---

## 8. Checkpoint-Level Packaging

### `quantize_checkpoint(checkpoint, bit_width)`
Wraps the full quantization process for one precision.

Steps:
1. Resolve state dict with `extract_state_dict()`.
2. Quantize all tensors with `quantize_state_dict()`.
3. Build serializable payload including:
   - format marker
   - source key
   - bit width
   - quantized state dict
   - metadata
   - per-tensor stats

### `save_quantized_checkpoint(payload, output_path)`
Creates output directory if needed, then saves payload with `torch.save()`.

---

## 9. File-Level Batch Quantization

### `quantize_file(input_path, output_dir, bit_widths)`
Quantizes one input checkpoint into multiple INT variants in one run.

Loop behavior for each bit width:
1. Build payload using `quantize_checkpoint()`.
2. Save to `<input_stem>_int{bit}.pt`.
3. Collect run summary into `results`.

Returns:
- A dictionary keyed by bit width with summary metrics and output paths.

### `_parse_bit_widths(raw_values)`
Converts CLI `--bits` values to integers and validates against supported list.

Behavior:
- No value given -> default to all supported precisions.
- Invalid value -> raises `ValueError`.

---

## 10. CLI Entrypoint

### `main()`
The script entrypoint.

CLI args:
- `--input`: path to `.pt/.pth` checkpoint.
- `--output-dir`: where quantized files are written.
- `--bits`: optional subset (for example: `32 16 8 4`).
- `--summary-file`: optional summary JSON path.
- `--dequantize`: optional flag to save dequantized float checkpoints.

Runtime actions:
1. Parse args.
2. Validate bit widths.
3. Run quantization batch.
4. Optionally dequantize and save reconstructed state dicts.
5. Write JSON summary.
6. Print concise per-bit result lines.

### `if __name__ == "__main__": main()`
Standard Python script guard that starts CLI flow only when executed directly.

---

## 11. Practical Notes

- `INT6/INT4/INT2` are logical precisions stored inside `int8` containers.
- Quantization is symmetric and per-tensor (single scale per tensor).
- Zero-point is fixed at `0`.
- This implementation is ideal for storage/analysis baselines.
- Hardware-optimized inference may still need export steps (for example ONNX/TensorRT with compatible kernels).
