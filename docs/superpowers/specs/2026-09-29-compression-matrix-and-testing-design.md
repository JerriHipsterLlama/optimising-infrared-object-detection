# Compression and Model-Testing Matrix Simplification

## Goal

Keep five research capabilities while removing duplicated work and legacy paths:

1. Model sensitivity screening to justify which YOLOv8n layers are selected.
2. Filterwise screening on sensitivity-approved layers to record per-filter ranking and accuracy/latency curves.
3. Cluster-size screening on the same candidate layers to justify the selected cluster size.
4. A compression matrix that structurally prunes candidates and produces FP32, FP16, and INT8 ONNX artifacts.
5. A testing matrix that reads ONNX models from a configured directory, builds target-local TensorRT engines, and evaluates them against the configured dataset using Python, Jetson C++, or both.

Keep the existing training scripts and their configs, model configuration, and graph/dependency exports. Training remains a distinct upstream workflow. The existing user changes in the RTX backend, its tests, and unrelated files are preserved.

## Current behavior and motivation

`run_compression_matrix` currently plans dense plus five pruning ratios for each of FP32 and FP16 (12 rows). Each row exports ONNX, builds a local TensorRT engine, validates accuracy, and benchmarks latency. The structural-pruned checkpoint is already reused per ratio, but ONNX export and device-specific build/evaluation work are repeated by precision. This makes compression depend on the machine’s TensorRT stack and mixes artifact generation with deployment testing.

The Jetson C++ runner accepts a TensorRT engine, reads a configured image directory, and reports latency. It may save raw outputs, but it does not calculate detection metrics. Python validation therefore remains the source of mAP/precision/recall; the C++ runner is the Jetson performance path.

There is no standalone directory-driven model-testing matrix today. `cluster_workflow.py` also combines useful cluster-size probes with global pruning, fine-tuning, target selection, and Jetson-result merging.

## Design

### 1. Model sensitivity screening

Keep `apps/yolov8_accuracy_sensitivity.py`, its config, dependency-graph discovery, and detailed/ranking artifacts. Its independent structural probes remain the evidence for layer selection. Do not add TensorRT engine building or fine-tuning to this stage.

### 2. Filterwise screening on sensitivity-approved layers

Preserve the implementation in `src/infrared_detection/evaluation/single_layer_performance_screening.py`, its `apps/single_layer_performance_screening.py` command, screening configs, and tests. This stage ranks filters within each selected layer and records the existing one-filter-at-a-time structural-pruning accuracy and latency curve. Its configured layer patterns must be restricted to the candidate layers accepted from the sensitivity-screening evidence; retain the rankings and per-candidate metrics as auditable outputs. Do not replace this filterwise workflow with cluster-size probes or remove it as a redundant command.

### 3. Cluster-size screening

Keep a separate config-driven screen that compares configured cluster sizes on the same sensitivity-approved candidate layers using structural probes and dataset accuracy. Reuse the existing cluster selection, structural-pruning, model-loading, validation, and graph-export helpers. Preserve the baseline comparison and per-candidate metrics needed to justify the chosen size (currently 8).

Remove unrelated global-candidate/fine-tuning/winner-selection/Jetson-merge stages from the cluster-size screening path. The filterwise curve and cluster-size comparison answer different questions and both remain supported. Historical result files are not deleted.

### 4. Compression matrix

Make compression device-neutral. For the dense model and each configured pruning ratio, load the original checkpoint, apply the configured candidate layers, cluster size, and importance rule, then export one FP32 ONNX source per candidate. From that source, use NVIDIA ModelOpt offline ONNX transforms to produce an FP16 ONNX variant and an INT8 Q/DQ ONNX variant. INT8 uses configured representative calibration data and records calibration settings/provenance. For the current five ratios this is six base exports and eighteen durable precision-specific ONNX artifacts (six candidates × three precisions).

The compression stage does not build TensorRT engines, run validation, or benchmark latency. ModelOpt belongs here solely for precision conversion/quantization; it is not an engine builder. Cache/reuse artifacts when candidate and conversion/calibration config fingerprints match. A failure in one precision conversion is recorded for that candidate/precision and must not erase other completed variants. Keep source checkpoint and config provenance, candidate identity, pruning summary, precision, ONNX path, calibration provenance where applicable, and errors in a manifest/CSV. Intermediate checkpoints and base ONNX inputs may be retained for audit/reuse according to config; precision-specific ONNX files are durable outputs.

Use deterministic candidate directories, for example:

```text
<output_dir>/dense/fp32/model.onnx
<output_dir>/dense/fp16/model.onnx
<output_dir>/dense/int8/model.onnx
<output_dir>/cluster-8-ratio-0.1/{fp32,fp16,int8}/model.onnx
...
<output_dir>/manifest.json
<output_dir>/results.csv
```

### 5. Model-testing matrix

Add a config-driven command that reads a configured model directory and deterministic ONNX glob, with no hard-coded candidate list. It uses the configured dataset YAML, split, image size, thresholds, and device settings for every model and records one result row per model/backend/precision combination.

- Python evaluation loads the ONNX model through the existing Ultralytics validation/metric-normalization path and reports mAP50, mAP50-95, precision, recall, and per-class AP. Python timing, when enabled, is labeled separately from device-native latency.
- Jetson C++ evaluation builds a TensorRT engine locally from each precision-specific ONNX model, then invokes the existing native runner against the dataset’s test-image directory. The standard `trtexec`/ONNX-parser route consumes ONNX, not a PyTorch `.pt` checkpoint; checkpoints must first be exported to ONNX. Torch-TensorRT is a separate PyTorch compilation route and is outside this design. It reports native latency and runtime provenance. It does not claim C++-computed accuracy; raw-output decoding is outside this change.
- `both` runs Python quality evaluation and Jetson C++ performance evaluation and joins results by model identity. If only C++ is selected, quality metrics are explicitly unavailable rather than inferred.

Failed exports, engine builds, Python evaluations, and C++ runs are recorded per row; a failure for one model does not erase completed rows for others. Output records the input ONNX path, backend, precision, dataset/split, commands, and device/runtime identity.

TensorRT engines are target artifacts: RTX/Windows and Jetson/Linux each build locally from the same portable precision-specific ONNX files. Do not copy a serialized engine between these targets.

### 6. Training workflows

Preserve the existing training entrypoints, model-specific training scripts, configurations, and outputs. Training remains separate from sensitivity screening and compression: it produces the trained checkpoint consumed by those workflows. Do not introduce compression or distillation into training as part of this change. Focal and Global Knowledge Distillation are explicitly deferred until FP32/FP16/INT8 compression and testing are implemented.

## FCPTS removal

Remove the active FCPTS implementation and its public imports/exports from `infrared_detection.compression.pruning`. Remove the stale `apps/compress.py` reference that directs users to FCPTS. Do not remove shared packages such as PyTorch, OpenCV, NumPy, or tqdm solely because FCPTS used them; remove a declared dependency only after confirming no retained workflow imports it.

Keep the historical `archive/legacy_python` snapshot untouched: it is not imported by the active package, so retaining it does not retain a runtime dependency. The supported pruning API is physical, dependency-aware structured channel pruning.

## Configuration boundaries

- Sensitivity config retains its trained model checkpoint, dataset, candidate-unit screening settings, validation settings, and output directory. Filterwise config retains the checkpoint, dataset, sensitivity-approved layer patterns, runtime/latency settings, and output directory. Cluster-size screening config retains the checkpoint, dataset, sensitivity-approved candidate layers, cluster-size probe values, validation settings, and output directory.
- Compression config retains checkpoint, candidate layers, cluster size, prune ratios, image size, FP32/FP16/INT8 selection, ModelOpt conversion settings, INT8 calibration dataset/settings, cache behavior, and ONNX output directory. It has no runtime device, TensorRT workspace, validation, or benchmark settings. Calibration uses configured representative data and records its source; it must not silently use the held-out test split.
- Testing config owns `models.directory`, an ONNX glob, dataset/split, Python and/or Jetson C++ backend selection, engine-build settings, and benchmark warm-up/iteration settings. Precision comes from each precision-specific ONNX artifact (validated metadata or filename/path convention); engine building must not silently convert a different precision.
- Existing training scripts/configs, YOLO model config, dataset config, and graph/dependency exports remain supported.

## Alternatives considered

- Keeping engine builds and testing inside compression preserves the current coupling and repeats ONNX export for each precision; rejected because it does not fix the requested runtime or Windows-to-Jetson handoff.
- Rewriting all pruning and evaluation code from scratch risks losing verified dependency-graph and metric behavior; rejected in favor of reusing the existing primitives and deleting orchestration that is outside the requested screening and compression/testing stages.
- Building TensorRT engines on Windows and transferring them to Jetson is not the default path; each target builds its own engine from portable ONNX.

## Acceptance criteria

- Model sensitivity screening and its layer-ranking evidence remain runnable and continue to provide evidence for valid candidate layers.
- The single-layer filterwise screen remains runnable with its per-filter rankings and one-filter-at-a-time accuracy/latency curves, restricted to the selected candidate layers.
- The cluster-size screen directly compares configured sizes on those same candidate layers and records comparable validation metrics.
- Compression produces FP32, FP16, and INT8 ONNX variants per dense/pruned candidate using ModelOpt for the offline precision transformations; it does not call TensorRT, dataset evaluation, or benchmarking.
- The testing matrix discovers models from its configured directory, evaluates the configured dataset, and supports Python, Jetson C++, and combined modes with backend-appropriate metrics.
- The current six-candidate/three-precision setup performs six base checkpoint-to-ONNX exports and produces up to eighteen precision-specific ONNX files. Precision-specific TensorRT engines are built locally from ONNX and tested only in the testing matrix.
- Core structured-pruning imports no longer load FCPTS modules; no active CLI or package API advertises FCPTS.
- Existing training scripts/configs, model and dataset configs, and graph/dependency exports remain available.
- Existing user edits are preserved, and historical result artifacts are not deleted.
- Unit tests cover candidate planning/export count, directory discovery, per-row failure handling, config validation, backend metric separation, FCPTS-free imports, and preservation of sensitivity, filterwise, and cluster-size screening. Hardware acceptance is a short Jetson run after unit-level verification; no model or Jetson run is part of the design/spec phase.

## Deferred scope and dependency notes

Focal and Global Knowledge Distillation, pruning fine-tuning, automated winner selection, historical artifact cleanup, and wholesale repository/requirements cleanup are not part of this change. ModelOpt is an intentional scoped dependency for ONNX FP16 conversion and INT8 quantization; its compatibility with the existing YOLO export graph and target TensorRT parsers must be verified during implementation. Shared dependencies remain until the retained-workflow import audit proves they are unused. The RTX native-FP16 changes already present in the worktree are user changes and will not be overwritten; precision-specific engine construction moves to the testing stage.

TensorRT's documented ONNX deployment workflow uses `trtexec --onnx=... --saveEngine=...`. TensorRT 11's strongly typed route consumes precision-encoded ONNX (including ModelOpt FP16 and INT8 Q/DQ exports) rather than relying on legacy `--fp16`/`--int8` switches. The target's installed TensorRT/JetPack version and ONNX operator/opset support remain authoritative at engine-build time.
