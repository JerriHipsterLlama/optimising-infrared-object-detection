# RTX 3070 preliminary compression results

This workflow provides preliminary local measurements before final validation on the Jetson Orin Nano. The RTX 3070 is useful for comparing relative accuracy, TensorRT latency, serialized engine size, and the effect of the cluster-pruning ratio. It is not a substitute for the target-board result.

## What is evaluated

The configured experiment contains:

- dense FP32 and FP16 baselines;
- structured pruning with cluster size 8;
- candidate layers `model.4.cv2.conv`, `model.6.cv2.conv`, and `model.8.cv2.conv`;
- prune ratios 0.05, 0.10, 0.15, 0.20, 0.25, and 0.30;
- FP32 and FP16 exports for every pruning ratio;
- input size 352 × 352.

This produces 14 planned rows. Each pruned checkpoint is created from the dense checkpoint and reused across the two precision exports. The filterwise experiment supplies cluster-size evidence; it does not become the source checkpoint for the final structured-pruning candidates.

## Where to read the results

After the run, inspect:

- `runs/experiments/rtx_compression_matrix/manifest.json`: complete provenance, status, errors, selected candidate, commands, and benchmark metadata;
- `runs/experiments/rtx_compression_matrix/results.csv`: compact table for spreadsheets and plots;
- `runs/experiments/rtx_compression_matrix/checkpoints/`: generated structured-pruned `.pt` checkpoints and pruning summaries;
- `runs/experiments/rtx_compression_matrix/variants/<variant-id>/`: ONNX exports and TensorRT engines for each row.

For the main comparison, use the dense FP32 row as the accuracy and latency reference. Plot `prune_ratio` against `map50_95`, `latency_p50_ms`/`latency_p95_ms`, and engine size. The selected candidate is the highest completed structured-pruning FP32 ratio whose mAP50-95 remains within `pruning.allowed_map50_95_drop` of dense FP32. Precision rows then show the deployment trade-off for that same checkpoint.

Rows with `status=failed` should be retained in the report with their error. INT8 is not part of this snapshot; it will be added only after the TensorRT 11.1 ModelOpt calibration workflow is validated. Do not replace a failed measurement with an estimate.

## Reproducibility

```powershell
$env:PYTHONPATH="$PWD\\src"
python apps/evaluate_compression_matrix.py --config configs/experiments/rtx_compression_matrix.yaml --dry-run
python apps/evaluate_compression_matrix.py --config configs/experiments/rtx_compression_matrix.yaml
```

Record the GPU, CUDA, TensorRT, Ultralytics, PyTorch, input size, warm-up count, and iteration count alongside any reported table. The RTX table should be labelled “preliminary”; repeat the selected candidates with native TensorRT benchmarking and power/memory telemetry on the Jetson Orin Nano for the final research result.
