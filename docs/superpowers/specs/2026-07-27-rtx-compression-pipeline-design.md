# RTX Compression Pipeline Design

## Goal

Create a reproducible local RTX 3070 pipeline that takes a selected structured-pruned YOLO checkpoint, exports dense and pruned variants as FP32, FP16, and INT8 TensorRT engines, evaluates detection accuracy, benchmarks latency, and writes one comparable results manifest.

## Scope

This phase covers structured channel/cluster-pruning artifact preparation and low-precision deployment evaluation. Knowledge distillation is explicitly out of scope until its teacher-student training protocol is designed and validated.

The official deployment precision variants are:

- FP32: dense or pruned TensorRT engine reference;
- FP16: practical reduced-precision TensorRT deployment;
- INT8: TensorRT calibrated integer deployment.

The existing PyTorch `int16` checkpoint utility remains available as a diagnostic LPQ utility, but it is not treated as a TensorRT INT16 deployment variant.

## Inputs

- A trained YOLO checkpoint from the repository configuration.
- The completed filterwise screening manifest, used to select a layer and candidate cluster-size pattern.
- A selected pruning ratio and cluster size, supplied explicitly rather than inferred silently.
- The existing CAMEL dataset configuration and calibration image directory.
- RTX 3070 CUDA/TensorRT runtime with `trtexec` available on `PATH`.

## Outputs

Each experiment directory will contain:

- the selected pruned `.pt` checkpoint;
- one ONNX export per precision variant where required;
- one TensorRT `.engine` per precision variant;
- accuracy metrics for every variant;
- latency p50, p95 when available, FPS, iteration count, and warm-up count;
- parameter count, serialized checkpoint size, ONNX size, and engine size;
- a `manifest.json` and `results.csv` with candidate identity, precision, device, provenance, and failure reasons.

The pipeline must preserve partial results. A failed INT8 calibration or TensorRT build must not discard successful FP32/FP16 results.

## Architecture

### 1. Pruning adapter

Add a focused adapter around the existing physical pruning functions. It will:

1. load the dense checkpoint;
2. select the configured layer and cluster size;
3. compute Minimum-Weight channel importance;
4. remove the requested complete channel cluster through the dependency graph;
5. save and reload the checkpoint;
6. verify that the reloaded topology and parameter count changed as expected.

The adapter will not modify the active filterwise screening manifest. It will consume it read-only and write a separate compression-matrix experiment directory.

### 2. TensorRT precision builder

Add a local builder that exports a checkpoint to ONNX and invokes the repository’s existing TensorRT export path for FP32, FP16, and INT8. INT8 will use the configured calibration image directory and record calibration provenance. The builder will expose a small injectable interface so tests can validate command construction without requiring TensorRT.

### 3. Evaluation and benchmark runner

For each built engine/checkpoint pair, run the existing YOLO validation path for accuracy and the TensorRT `trtexec` adapter for latency. Results will be normalized into the project metric schema and tagged with `device=rtx3070`, `precision`, `candidate_id`, and `benchmark_provenance`.

### 4. CLI and configuration

Add one configuration-driven CLI command with explicit arguments for:

- source checkpoint;
- filterwise manifest;
- selected layer;
- cluster size;
- pruning ratio or number of filters/clusters removed;
- output directory;
- calibration data;
- benchmark iterations and warm-up count.

The default configuration will generate the six preliminary variants: dense FP32/FP16/INT8 and structured-pruned FP32/FP16/INT8.

## Accuracy and performance rules

- All variants use the same dataset split, image size, confidence threshold, IoU threshold, and batch size.
- TensorRT measurements use warm-up followed by repeated batch-1 inference.
- PyTorch validation metrics and TensorRT latency are recorded separately; neither is presented as a substitute for the other.
- A missing engine, failed export, failed calibration, or failed benchmark is represented as a failed row with its exception and command provenance.
- No candidate is selected solely from parameter count or theoretical FLOPs; the final summary retains accuracy, artifact size, and measured latency together.

## Testing strategy

- Unit-test candidate selection from a filterwise manifest.
- Unit-test cluster-pruning configuration validation and output naming.
- Unit-test FP32/FP16/INT8 TensorRT command construction, including calibration arguments.
- Unit-test result-manifest merging and partial-failure preservation.
- Run existing pruning, quantization, workflow, and application tests.
- Perform one local smoke run only after the unit and integration tests pass; this smoke run may require the user’s installed TensorRT runtime and model checkpoint.

## Alternatives considered

### Separate scripts for pruning, export, and benchmarking

This would reuse more existing commands but makes it easier to accidentally compare different checkpoints, image sizes, or precision settings. It is retained as a fallback for debugging, not as the primary research workflow.

### PyTorch-only integer LPQ

This would be easy to run but would not provide a valid TensorRT latency comparison. It remains useful for quantization-error diagnostics, not for the primary deployment result.

### Add KD now

This would increase the experimental matrix and introduce a second training variable before the baseline pipeline is validated. It is deferred so the first results isolate pruning and deployment precision.
