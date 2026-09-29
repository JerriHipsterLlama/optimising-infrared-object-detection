# Compression and Model-Testing Matrix Simplification

## Goal

Keep four research capabilities while removing duplicated work and legacy paths:

1. Model sensitivity screening to justify which YOLOv8n layers are selected.
2. Filterwise/cluster-size screening to justify the selected cluster size.
3. A compression matrix that structurally prunes candidates and exports portable ONNX models.
4. A testing matrix that reads ONNX models from a configured directory and evaluates them against the configured dataset using Python, Jetson C++, or both.

Keep model configuration and graph/dependency exports. The existing user changes in the RTX backend, its tests, and unrelated files are preserved.

## Current behavior and motivation

`run_compression_matrix` currently plans dense plus five pruning ratios for each of FP32 and FP16 (12 rows). Each row exports ONNX, builds a local TensorRT engine, validates accuracy, and benchmarks latency. The structural-pruned checkpoint is already reused per ratio, but ONNX export and device-specific build/evaluation work are repeated by precision. This makes compression depend on the machine’s TensorRT stack and mixes artifact generation with deployment testing.

The Jetson C++ runner accepts a TensorRT engine, reads a configured image directory, and reports latency. It may save raw outputs, but it does not calculate detection metrics. Python validation therefore remains the source of mAP/precision/recall; the C++ runner is the Jetson performance path.

There is no standalone directory-driven model-testing matrix today. `cluster_workflow.py` also combines useful cluster-size probes with global pruning, fine-tuning, target selection, and Jetson-result merging.

## Design

### 1. Model sensitivity screening

Keep `apps/yolov8_accuracy_sensitivity.py`, its config, dependency-graph discovery, and detailed/ranking artifacts. Its independent structural probes remain the evidence for layer selection. Do not add TensorRT engine building or fine-tuning to this stage.

### 2. Filterwise and cluster-size screening

Keep one config-driven screening command whose retained purpose is to compare configured cluster sizes on the selected candidate layers using structural probes and dataset accuracy. Reuse the existing cluster selection, structural-pruning, model-loading, validation, and graph-export helpers. Preserve the baseline comparison and per-candidate metrics needed to justify the chosen size (currently 8).

Remove the unrelated global-candidate/fine-tuning/winner-selection/Jetson-merge stages from this screening path. The full one-filter-at-a-time performance sweep is not a second required workflow; its useful filter ranking and structural helpers remain available to the cluster-size screen where applicable. Historical result files are not deleted.

### 3. Compression matrix

Make compression device-neutral. For the dense model and each configured pruning ratio, load the original checkpoint, apply the configured candidate layers, cluster size, and importance rule, then export one ONNX model per candidate. For the current five ratios this is six ONNX exports, not twelve precision-specific exports.

The compression stage does not build TensorRT engines, run dataset validation, benchmark latency, or apply FP16/INT8 conversion. Precision is a testing-matrix setting because TensorRT engines must be built on their target machine. Keep the source checkpoint and config provenance, candidate identity, pruning summary, ONNX path, and export errors in a compact manifest/CSV. Intermediate checkpoints may be temporary; ONNX is the durable candidate artifact.

Use deterministic candidate directories, for example:

```text
<output_dir>/dense/model.onnx
<output_dir>/cluster-8-ratio-0.1/model.onnx
...
<output_dir>/manifest.json
<output_dir>/results.csv
```

### 4. Model-testing matrix

Add a config-driven command that reads a configured model directory and deterministic ONNX glob, with no hard-coded candidate list. It uses the configured dataset YAML, split, image size, thresholds, and device settings for every model and records one result row per model/backend/precision combination.

- Python evaluation loads the ONNX model through the existing Ultralytics validation/metric-normalization path and reports mAP50, mAP50-95, precision, recall, and per-class AP. Python timing, when enabled, is labeled separately from device-native latency.
- Jetson C++ evaluation builds a TensorRT engine locally from each ONNX model using the configured precision, then invokes the existing native runner against the dataset’s test-image directory. It reports native latency and runtime provenance. It does not claim C++-computed accuracy; raw-output decoding is outside this change.
- `both` runs Python quality evaluation and Jetson C++ performance evaluation and joins results by model identity. If only C++ is selected, quality metrics are explicitly unavailable rather than inferred.

Failed exports, engine builds, Python evaluations, and C++ runs are recorded per row; a failure for one model does not erase completed rows for others. Output records the input ONNX path, backend, precision, dataset/split, commands, and device/runtime identity.

## FCPTS removal

Remove the active FCPTS implementation and its public imports/exports from `infrared_detection.compression.pruning`. Remove the stale `apps/compress.py` reference that directs users to FCPTS. Do not remove shared packages such as PyTorch, OpenCV, NumPy, or tqdm solely because FCPTS used them; remove a declared dependency only after confirming no retained workflow imports it.

Keep the historical `archive/legacy_python` snapshot untouched: it is not imported by the active package, so retaining it does not retain a runtime dependency. The supported pruning API is physical, dependency-aware structured channel pruning.

## Configuration boundaries

- Screening configs retain the trained model checkpoint, dataset, candidate layers, cluster-size probe values, validation settings, and output directory.
- Compression config retains checkpoint, candidate layers, cluster size, prune ratios, image size, and ONNX output directory. It has no runtime device, precision matrix, TensorRT workspace, calibration, or benchmark settings.
- Testing config owns `models.directory`, an ONNX glob, dataset/split, Python and/or Jetson C++ backend selection, TensorRT precision, and benchmark warm-up/iteration settings.
- Existing YOLO model config, dataset config, and graph/dependency exports remain supported.

## Alternatives considered

- Keeping engine builds and testing inside compression preserves the current coupling and repeats ONNX export for each precision; rejected because it does not fix the requested runtime or Windows-to-Jetson handoff.
- Rewriting all pruning and evaluation code from scratch risks losing verified dependency-graph and metric behavior; rejected in favor of reusing the existing primitives and deleting orchestration that is outside the four requested stages.
- Building TensorRT engines on Windows and transferring them to Jetson is not the default path; each target builds its own engine from portable ONNX.

## Acceptance criteria

- The sensitivity workflow and its layer-ranking evidence remain runnable.
- The filterwise screen directly compares configured cluster sizes on the selected layers and records comparable validation metrics.
- Compression produces exactly one usable ONNX per dense/pruned candidate and does not call TensorRT, dataset evaluation, or benchmarking.
- The testing matrix discovers models from its configured directory, evaluates the configured dataset, and supports Python, Jetson C++, and combined modes with backend-appropriate metrics.
- The current six-candidate/two-precision setup performs six compression exports; precision-specific engine builds and tests happen only in the testing matrix.
- Core structured-pruning imports no longer load FCPTS modules; no active CLI or package API advertises FCPTS.
- Model and dataset configs plus graph/dependency exports remain available.
- Existing user edits are preserved, and historical result artifacts are not deleted.
- Unit tests cover candidate planning/export count, directory discovery, per-row failure handling, config validation, backend metric separation, and FCPTS-free imports. Hardware acceptance is a short Jetson run after unit-level verification; no model or Jetson run is part of the design/spec phase.

## Deferred scope and dependency notes

INT8/ModelOpt calibration, pruning fine-tuning, automated winner selection, historical artifact cleanup, and wholesale repository/requirements cleanup are not part of this change. Shared dependencies remain until the retained-workflow import audit proves they are unused. The RTX native-FP16 changes already present in the worktree are user changes and will not be overwritten; this design moves precision-specific builds to the testing stage without prescribing a change to those backend edits.
