# Single-Layer Performance Screening Design

## Purpose

Replace the overlapping filterwise sensitivity and screening workflows with one authoritative workflow named `single_layer_performance_screening`. The workflow will reproduce the Single Layer Performance Response stage of the Cluster Pruning method for YOLOv8n and YOLOv8m on an RTX 3070 and a Jetson Orin Nano.

This workflow ends after independent single-layer response curves have been collected. It does not select a final cluster size, perform multi-layer cluster pruning, fine-tune a pruned model, export ONNX, or build TensorRT engines.

## Scope

The implementation covers:

- Exact MinimumWeight scoring from Equation 3 of the Cluster Pruning paper.
- A fixed ascending filter ranking calculated from the dense layer before pruning.
- Independent complete sweeps for selected YOLOv8n and YOLOv8m layers.
- Accuracy evaluation after every successful filter removal.
- Direct PyTorch/CUDA forward-pass latency after every successful filter removal.
- Resumable execution with bounded checkpoint storage.
- A research-oriented CSV with primary progress and performance fields first.
- RTX 3070 and Jetson Orin Nano configurations.
- Removal of the legacy filterwise screening orchestration, CLI modes, configuration, Kaggle generator, and tests.

Faster R-CNN with a MobileNet-family backbone is explicitly outside this implementation. It will require a separate layer-selection and dependency design.

## Authoritative Names

- Workflow module: `src/infrared_detection/evaluation/single_layer_performance_screening.py`
- CLI application: `apps/single_layer_performance_screening.py`
- Result filename: `results.csv`
- Manifest filename: `manifest.json`
- Ranking filename: `rankings.json`
- Resume-state filename: `state.json`
- Transient checkpoint filename: `resume.pt`

The words `filterwise_sensitivity` and `filterwise_screening` will no longer identify executable workflows. Historical design documents and existing run directories remain as research history, but current documentation and commands will point only to `single_layer_performance_screening`.

## Model and Hardware Matrix

The supplied checkpoints are:

- `yolov8n.pt`
- `yolov8m.pt`

The workflow uses four explicit experiment configurations so results from different model and hardware combinations cannot be merged accidentally:

- `configs/experiments/single_layer_performance_yolov8n_rtx.yaml`
- `configs/experiments/single_layer_performance_yolov8m_rtx.yaml`
- `configs/experiments/single_layer_performance_yolov8n_orin.yaml`
- `configs/experiments/single_layer_performance_yolov8m_orin.yaml`

Each configuration records `model_variant`, `hardware_label`, checkpoint path, dataset YAML, CUDA device, precision, image size, confidence threshold, IoU threshold, latency warm-up count, latency iteration count, selected layer patterns, and an isolated output directory.

The initial configurations use an image size of 352, batch size 1, FP16 CUDA execution, 20 warm-up forward passes, and 100 measured forward passes. The actual CUDA device name, PyTorch version, CUDA version, Ultralytics version, and resolved input shape are written to the manifest.

## Layer Selection

The workflow accepts shell-style wildcard patterns and expands them deterministically in `named_modules()` order:

```yaml
layer_patterns:
  - model.6.m.*.cv1.conv
  - model.6.m.*.cv2.conv
  - model.8.m.*.cv1.conv
  - model.8.m.*.cv2.conv
  - model.12.m.*.cv1.conv
  - model.12.m.*.cv2.conv
  - model.18.m.*.cv1.conv
  - model.18.m.*.cv2.conv
  - model.21.m.*.cv1.conv
  - model.21.m.*.cv2.conv
```

Every pattern must match at least one module. Every matched module must be a `torch.nn.Conv2d`. Duplicate matches are removed while preserving model order. A missing pattern or non-convolution match fails configuration validation before the baseline evaluation begins.

The supplied models resolve as follows:

| Model | Concrete layers | Layer widths | Pruned candidates |
|---|---:|---|---:|
| YOLOv8n | 12 | eight 64-filter and four 128-filter layers | 1,012 |
| YOLOv8m | 24 | sixteen 192-filter and eight 288-filter layers | 5,352 |
| Combined | 36 | — | 6,364 |

Candidate count is the sum of `original_filter_count - 1` for every selected layer.

## MinimumWeight Scoring and Ranking

For output filter `k` in layer `l`, the workflow implements Equation 3 as:

\[
\theta_{MW}(F_l^k) = \operatorname{mean}\left((W_l^k)^2\right)
\]

For a convolution weight tensor with shape `[out_channels, in_channels, kernel_height, kernel_width]`, the mean is taken over the input-channel and kernel dimensions. Scores are calculated in floating point without changing the stored model weights.

Filters are stable-sorted by `(minimum_weight_score, original_filter_index)` in ascending order. The complete ranking is calculated once from the dense model before that layer is modified and is persisted in `rankings.json`.

The ranking is not recalculated after each removal. This is required to match the paper's procedure: rank the original filters, then prune them in ascending rank order.

## Original and Physical Filter Indices

Every ranking entry retains its original dense-model filter index. Physical tensor indices shift after each removal, so the workflow maintains an ordered list of original indices still present in the current model.

For each pruning step:

1. Read the next original filter index from the fixed ranking.
2. Find that original index in the current `remaining_original_indices` list.
3. Use its list position as the physical channel index supplied to structural pruning.
4. Remove the original index from `remaining_original_indices` after pruning succeeds.

Each result row stores both `original_filter_index` and `physical_filter_index`. This makes the pruning sequence auditable and allows a resumed run to continue the fixed ranking correctly.

## Independent Layer Sweep

Each layer is screened independently:

1. Load the same dense checkpoint.
2. Resolve the selected convolution.
3. Calculate and persist its fixed MinimumWeight ranking.
4. Prune the first ranked filter structurally.
5. Evaluate detection accuracy.
6. Measure direct PyTorch/CUDA forward latency.
7. Persist the result and resume state.
8. Prune the next ranked filter from the already-pruned current model.
9. Continue until one output filter remains.
10. Discard the completed layer model and reload the dense checkpoint for the next layer.

No accuracy early stop is permitted. A valid layer with `K` filters produces `K - 1` measurements.

No fine-tuning occurs during this screening stage. This isolates the immediate response caused by structured filter removal.

## Structural Pruning

The workflow reuses the existing Torch-Pruning-backed dependency graph and explicit-index structural pruning primitive. It physically removes the selected output channel and propagates required changes to normalisation parameters and downstream input channels.

Reusable primitives such as `run_filterwise_probe` remain in the pruning package because the compression and future cluster-pruning workflows still require explicit structural channel removal. Only the duplicate sensitivity orchestration is removed from `cluster_workflow.py`.

## Accuracy Measurement

The dense baseline and every valid candidate are evaluated with the same dataset, split, image size, batch size, confidence threshold, IoU threshold, device, and precision.

The primary metric is `map50_95`. Supporting metrics are:

- `map50`
- `precision`
- `recall`
- per-class average precision
- `map50_95_drop` relative to the corresponding dense model and hardware baseline

Ultralytics validation is performed on a separately loaded evaluation copy when validation mutates or fuses modules. The sequentially pruned working model remains suitable for the next structural pruning operation.

## Direct PyTorch/CUDA Latency

The sensitivity workflow does not export candidate models. Latency is measured directly from the raw PyTorch model forward pass using CUDA events.

The profiler:

1. Places the model and synthetic batch-1 input on the configured CUDA device.
2. Applies the configured FP16 or FP32 precision consistently to model and input.
3. Sets evaluation mode and disables gradients.
4. Runs the configured warm-up passes.
5. Synchronises CUDA.
6. Records one CUDA-event duration for each measured forward pass.
7. Synchronises before reading durations.
8. Returns mean, P50, P95, FPS derived from mean latency, warm-up count, iteration count, input shape, device name, and precision.

The timing scope is model forward propagation only. It excludes file I/O, image decoding, preprocessing, dataset loading, and non-maximum suppression. RTX and Jetson measurements are independent because hardware-efficient filter counts may differ.

## Resume and Storage Model

The workflow stores no per-candidate checkpoint, ONNX model, or TensorRT engine.

For the active layer only, it stores:

- `resume.pt`: the latest successfully measured pruned model.
- `state.json`: current model/hardware identity, configuration fingerprint, active layer, next filter rank, remaining original indices, and last completed candidate ID.

The fixed ranking is stored separately in `rankings.json`. Completed measurements are stored in `results.csv` and `manifest.json`.

Before evaluation, the newly pruned model is saved as `resume.pending.pt` so validation can load an isolated copy. The previous `resume.pt` remains untouched until accuracy and latency both succeed. A successful candidate atomically promotes `resume.pending.pt` to `resume.pt`, then writes the completed result and advanced state. A failed measurement deletes the pending file and leaves the previous valid resume checkpoint and state unchanged. Result rows are keyed by deterministic candidate IDs, so retrying a step replaces the same logical row instead of duplicating it.

After a layer completes, its transient checkpoint is deleted. The next layer starts from the dense checkpoint. Storage therefore remains approximately the size of the dense checkpoint plus one active pruned checkpoint and small metadata files.

A resume is rejected if the checkpoint identity, dataset identity, model variant, layer patterns, precision, image size, or ranking algorithm version differs from the saved experiment fingerprint.

## Failure Semantics

Failures are divided into two classes:

### Terminal structural failure

A dependency-graph rejection or propagated channel mismatch means the requested one-filter sequence cannot produce a valid model for that layer. The failing candidate is marked `skipped`. All later ranks for that independent layer are also marked `skipped` with a reason referencing the first structural failure. The workflow then continues with the next dense layer sweep.

These rows are never interpreted as accuracy measurements and are not retried on restart.

### Retryable runtime failure

CUDA out-of-memory, interrupted execution, dataset access failure, or another environmental/runtime failure records the candidate as `failed`, preserves the last valid resume checkpoint, writes state, and stops the workflow. Restarting retries that candidate from the last valid state.

The workflow does not continue repeatedly after an OOM because doing so cannot produce valid subsequent candidates.

## Result CSV

`results.csv` contains one dense baseline row followed by deterministic candidate rows. The writer uses an explicit research-oriented leading-column order:

1. `candidate_id`
2. `model_variant`
3. `hardware`
4. `status`
5. `layer`
6. `filter_rank`
7. `original_filter_index`
8. `physical_filter_index`
9. `minimum_weight_score`
10. `filters_before`
11. `filters_removed`
12. `filters_remaining`
13. `map50_95`
14. `map50_95_drop`
15. `map50`
16. `precision`
17. `recall`
18. `latency_mean_ms`
19. `latency_p50_ms`
20. `latency_p95_ms`
21. `fps`

Remaining measurement and provenance columns follow in deterministic order. `reason` and `error` are always the final two columns.

The generic CSV writer gains optional leading and trailing column order arguments. Existing workflows retain their current behaviour unless they supply those arguments.

## Output Layout

Each model/hardware combination has an isolated directory:

```text
runs/experiments/single_layer_performance_screening/
  yolov8n/
    rtx3070/
    jetson_orin_nano/
  yolov8m/
    rtx3070/
    jetson_orin_nano/
```

Each leaf directory contains:

```text
results.csv
manifest.json
rankings.json
state.json
resume.pt  # present only while a layer is incomplete
```

## Legacy Workflow Removal

The implementation removes:

- `run_filterwise_evaluation` and its private planning, resume, step, and export helpers from `cluster_workflow.py`.
- `--filterwise` and `--full-curve` from `apps/evaluate_cluster_pruning.py`.
- `configs/experiments/filterwise_rtx_screening.yaml`.
- `tools/build_kaggle_filterwise_notebook.py`.
- `tests/unit/test_build_kaggle_filterwise_notebook.py`.
- Legacy filterwise workflow tests from `tests/unit/test_cluster_workflow.py`.
- README instructions that point to the Kaggle filterwise notebook or legacy filterwise command.

The implementation does not delete historical run data or historical design/plan documents. The supervisor report generator is updated to accept the new `results.csv` locations without rewriting historical artifacts.

## CLI Behaviour

The sole command interface is:

```powershell
.venv\Scripts\python.exe apps\single_layer_performance_screening.py --config configs\experiments\single_layer_performance_yolov8n_rtx.yaml
```

Complete curves are mandatory; there is no `--full-curve` switch and no accuracy early-stop option. The command prints the model variant, hardware label, selected layer count, total planned candidates, active layer, current filter rank, filters remaining, latest `map50_95`, and latest latency after each candidate.

## Testing Strategy

Unit tests cover:

- Exact mean-squared MinimumWeight scores.
- Stable ascending ranking and original-index tie-breaking.
- Wildcard expansion order, deduplication, and validation failures.
- Expected layer and candidate counts for lightweight YOLOv8n and YOLOv8m architecture fixtures.
- Original-to-physical index translation after sequential removals.
- Independent dense restart for every layer.
- Complete `K - 1` candidate generation without early stopping.
- Direct CUDA timing through an injectable profiler adapter so unit tests do not require a GPU.
- Baseline-relative accuracy metrics.
- One-checkpoint storage and completed-layer checkpoint deletion.
- Resume from the latest valid candidate without duplicate rows.
- Resume rejection after experiment fingerprint changes.
- Terminal structural failure propagation to remaining layer rows.
- Retryable OOM/runtime failure behaviour.
- Explicit CSV leading columns and trailing `reason`, `error` columns.
- Removal of legacy CLI flags and documentation references.

An integration smoke test runs a tiny convolutional model through ranking, two structural pruning steps, metric persistence, interruption, and resume. Hardware acceptance consists of one short RTX run and one short Jetson run with reduced iterations before starting the complete experiments.

## Success Criteria

The replacement is complete when:

- One command runs the authoritative workflow for each model/hardware configuration.
- The requested patterns resolve correctly for both supplied YOLO models.
- Filter ranking exactly implements mean squared weights and remains fixed per dense layer.
- Every valid layer runs until one filter remains.
- Every valid candidate has accuracy and direct CUDA latency measurements.
- No candidate ONNX or TensorRT artifacts are produced.
- At most one transient pruned checkpoint exists per active experiment.
- Restart does not repeat completed measurements or terminal skips.
- `results.csv` presents progress and primary research metrics first.
- The legacy filterwise screening workflows are no longer executable or documented.
- The complete automated test suite passes.
