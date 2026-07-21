# Cluster Pruning Evaluation Design

## Goal

Evaluate hardware-aware, structured cluster pruning for the infrared YOLO detector. The primary result is the smallest exported model whose mAP50-95 is within 1.0 absolute point of the unpruned baseline. Jetson Orin Nano latency is a selection tie-breaker and a required reported deployment metric.

## Scope

The prior `focus-on-yolo` implementation is a source of cluster-selection and ONNX inspection logic, not a drop-in pruning executor. Its default operation zeroes ONNX filters while retaining tensor shapes, so it cannot reduce the dense exported model size. The new implementation must structurally remove safe channel groups from the trainable detector before export and fine-tune the resulting model.

## Two-Stage Cluster-Size Selection

### Stage 1: Single-Layer Sensitivity Profiling

1. Measure the unpruned baseline on the RTX 3070 and, when available, the Jetson Orin Nano.
2. For each safe backbone/neck dependency group, remove one small structural cluster at each candidate cluster size that fits the group width.
3. Export and validate each probe candidate. Record model bytes, mAP50-95 change, and p50/p95 latency through the deployment path.
4. Use RTX 3070 measurements only to eliminate candidates that provide no measurable latency or size benefit, or create disproportionate accuracy loss.
5. Re-run the surviving probes on the Jetson Orin Nano. Retain the non-dominated cluster sizes: those with a real Orin Nano benefit and no disproportionate local mAP loss.

Single-layer profiling narrows the cluster-size candidates; it does not select the final model.

### Stage 2: Global Prune-Fine-Tune Evaluation

1. Apply each surviving cluster size at configured global prune ratios.
2. Fine-tune every structurally pruned candidate from its trainable model, then export and validate it.
3. Benchmark every valid candidate on the Jetson Orin Nano.
4. Select and report the primary and exploratory candidates using the evaluation contract below.

## Candidate Generation

1. Load a trained YOLO checkpoint and its dataset/training configuration.
2. Identify prunable convolution groups while protecting the detection head and coupled graph paths.
3. Score filter clusters using the retained cluster-ranking policy.
4. Generate candidates across configured cluster sizes and prune ratios.
5. Structurally prune a candidate only when the dependency group is safe to update. Reject an unsafe candidate with an explicit reason; do not silently fall back to zero masking.
6. Fine-tune each structurally pruned candidate from the pruned trainable model, then export it to ONNX or the deployment format.

## Evaluation Contract

Every baseline and candidate writes one machine-readable result row containing:

- candidate identifier, cluster size, prune ratio, and pruning/fine-tuning status;
- exported model bytes, parameter count, and GFLOPs;
- mAP50-95 and absolute mAP50-95 change from the baseline;
- Jetson benchmark p50 and p95 end-to-end latency, FPS, peak memory, power, and energy per inference when available;
- paths to the checkpoint, exported model, validation result, and Jetson benchmark artifact.

Candidates are classified as:

- `primary_feasible`: mAP50-95 drop is at most 1.0 absolute point;
- `exploratory_feasible`: drop is more than 1.0 and at most 2.0 points;
- `rejected_accuracy`: drop exceeds 2.0 points;
- `failed`: pruning, fine-tuning, exporting, validating, or benchmarking failed.

The selected primary candidate is the smallest `primary_feasible` exported model. Equal-size candidates are ordered by lower Jetson p50 latency, then higher mAP50-95. The best exploratory candidate is reported separately and is never substituted for the primary result.

## Hardware-Aware Cluster Sizing

The evaluation sweeps configured cluster sizes rather than treating an LCM-derived value as an unverified optimum. RTX 3070 measurements are an interim screening signal only; the Jetson Orin Nano is the final target and the sole hardware environment used for final candidate ranking. A cluster size is therefore selected by measured model-size/accuracy/Orin-Nano-latency trade-offs, matching the empirical principle of Gamanayake et al.

## Error Handling

- Preserve the baseline result even when every candidate fails.
- Fail a candidate independently; continue evaluating remaining candidates.
- Record unavailable hardware metrics as null with a reason rather than fabricating CPU results as Jetson measurements.
- Validate each exported model before benchmarking and reject invalid output shapes or unsupported graph changes.

## Tests

Tests are written before implementation and cover:

- candidate classification at the 1.0 and 2.0 point accuracy boundaries;
- deterministic selection of the primary and exploratory winners;
- result-row completeness and stable serialization;
- structural-pruning safety guards for protected/coupled YOLO components;
- continuation after an individual candidate failure.

Integration testing uses a small YOLO-compatible graph/checkpoint fixture. End-to-end infrared training and Jetson benchmarks are documented manual runs because they require project data and physical target hardware.
