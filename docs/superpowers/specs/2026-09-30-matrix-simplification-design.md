# Compression and Screening Workflow Simplification

## Goal

Reduce the active research pipeline to five auditable workflows while keeping the existing YOLO training entry point available:

1. Model sensitivity screening establishes which layers are valid pruning candidates.
2. Filterwise screening records per-filter ranking and performance response on the approved layers.
3. A distinct cluster-size screen compares candidate cluster sizes on those layers and supports the selected size (currently 8).
4. The compression matrix produces portable FP32, FP16, and INT8 ONNX artifacts from a configured baseline checkpoint.
5. The testing matrix discovers ONNX models from a configured directory and evaluates the configured dataset with Python, Jetson C++, or both.

Keep these as distinct, config-driven entry points. Retain model/dataset configs, dependency-graph exports, the native Jetson runner, and the minimal YOLO training command. The compression matrix consumes the checkpoint path supplied in its config; it does not train or infer a checkpoint path. Faster R-CNN training/compression integration is future work, not part of this change.

## Current evidence and constraints

- The existing 29 September design retains ModelOpt and a broader multi-stage pipeline; this design supersedes those choices for the current simplification.
- The current matrix plans six candidates × three precisions (18 ONNX outputs). Preserve the candidate/precision coverage and one base export per candidate.
- The current compression implementation recalculates minimum-weight scores for every module 16 times per pruning ratio. On the supplied checkpoint, 16 whole-model scoring passes took 0.385 s; scoring the same 16 target layers directly took 0.008 s, with exact score parity. This is a safe, small optimization, not a claim that scoring dominates total runtime.
- Existing matrix tests currently have 38 passing and one failure: the RTX config test expects an obsolete seven-layer list rather than the configured 16 layers. Screening tests have 97 passing and 2 skipped.
- The project environment has ONNX, ONNX Runtime, and `onnxconverter-common`. A temporary synthetic-graph check successfully converted FP16 and static QDQ INT8 ONNX without ModelOpt. Full YOLO and Jetson/TensorRT compatibility still require verification.
- The current Jetson manifest/log contains prior ModelOpt failures; it is not evidence of a successful current 18-variant run.

## Design

### 1. Sensitivity screening

Keep `apps/yolov8_accuracy_sensitivity.py`, `evaluation/accuracy_sensitivity.py`, its config, candidate/dependency discovery, validation checks, and ranking/graph artifacts. The approved candidate-layer list remains config-driven and traceable to these results. Remove only adjacent plotting/reporting or legacy screening paths after confirming they have no retained consumer.

### 2. Filterwise screening

Keep `apps/single_layer_performance_screening.py` and `evaluation/single_layer_performance_screening.py`, including MinimumWeight rankings, independent one-layer-at-a-time pruning curves, accuracy/recall/latency outputs, resumable checkpoints, and CSV/manifest artifacts. Use the sensitivity-approved candidate layers from config. The CLI already wires its default adapters; retain that behavior.

### 3. Dedicated cluster-size screening

Keep a separate fifth workflow and entry point for cluster-size comparison. It evaluates configured cluster sizes (including 8) on the sensitivity-approved layers using structural pruning and dataset validation, and records per-layer/per-size metrics and parameter reduction. Reuse the current structural-probe and validation primitives. Remove fine-tuning, global-pruning winner selection, Jetson metric merging, and unrelated target-selection behavior from this screen. Do not merge it into the filterwise CLI.

### 4. Compression matrix

Keep one config-driven matrix for the configured external YOLO checkpoint, candidate layers, cluster size, pruning ratios, input size, precision set, calibration data, and output directory. Maintain six base checkpoint-to-ONNX exports and up to eighteen precision outputs for the current matrix.

- FP32 copies the base ONNX graph.
- FP16 uses `onnxconverter-common` conversion, keeping model I/O types compatible with the current runtime path.
- INT8 uses ONNX Runtime static quantization in QDQ form with the configured representative calibration split; the test split remains prohibited for calibration.
- Remove ModelOpt from the active converter path and its optional dependency declaration. ONNX Runtime is already a project dependency; declare `onnxconverter-common` directly in the compression extra rather than relying on it arriving transitively through ModelOpt.
- Score only the current candidate convolution when planning its MinimumWeight clusters. This preserves the existing score formula and ranking exactly while avoiding repeated whole-model scans.
- Keep the existing per-candidate base-ONNX reuse, deterministic output paths, row-level failure isolation, and resumable manifest/CSV. Do not reduce the precision or pruning-ratio matrix to improve timing.

Compression performs no TensorRT engine build, dataset evaluation, or latency benchmark. Precision-specific engine builds remain local to the testing target. Validate every written ONNX with the ONNX checker and record converter/calibration provenance.

### 5. Model-testing matrix

Keep configured `models.directory` and ONNX glob discovery; do not introduce a hard-coded candidate list. Keep Python, Jetson C++, and `both` modes and their backend-specific metrics. Build TensorRT engines locally from each ONNX artifact and invoke the existing C++ runner for Jetson measurements. Keep Python quality metrics distinct from native C++ latency. Persist completed rows and reuse a result only when the input ONNX and relevant test/backend settings still match, so reruns do not rebuild/re-evaluate unchanged artifacts. Preserve independent row failures.

### YOLO training and future Faster R-CNN support

Retain a minimal YOLO training command whose checkpoint output is set by the YOLO model config. The compression matrix still requires an explicit checkpoint path, whether produced by training or supplied externally. Do not run training implicitly from compression.

Do not implement or claim Faster R-CNN compression support in this change. Keep the current Faster R-CNN source/config available as future work, but treat its checkpoints as unsupported by the YOLO-specific sensitivity/pruning/export path until model-family-specific behavior is added and tested. Do not build a speculative plugin framework now.

## Simplification and removal boundary

After mapping imports, app entry points, tests, and configs, remove active workflows not needed by the five screens/matrices or retained training: legacy combined compression/distillation, FCPTS-only code and APIs, duplicate reporting/benchmark commands, and obsolete experiment configs/tests. Retain shared model loading, dataset/evaluation metrics, dependency graph/pruning primitives, ONNX export/precision tools, manifest writing, model/dataset configs, graph exports, native Jetson runner, and existing historical run artifacts. Do not delete data or run outputs. Preserve the current user-owned `.gitignore`, `rtx.py` deletion, and `.vscode/` changes.

Update the stale RTX config assertion to the approved 16 candidate layers if that config remains; otherwise remove it only if the RTX config itself is proven obsolete by the retained workflow inventory.

## Failure and compatibility handling

- Missing checkpoint, data, ONNX converter, calibration images, or TensorRT tools must identify the exact missing input/tool and the affected row.
- Invalid/empty/test-split calibration fails before conversion.
- Conversion failure in one precision does not erase sibling outputs; testing failure for one ONNX/backend does not erase other result rows.
- TensorRT/JetPack ONNX operator, opset, FP16, and INT8 QDQ compatibility is target-dependent. Host ONNX validation is necessary but not sufficient; run a small real-model Jetson engine-build and inference test before calling the matrix deployable.
- Dependency cleanup follows a retained-workflow import audit. Keep a dependency if a retained screen/training/export/test path still uses it; report any package whose removal breaks an in-scope workflow instead of deleting that path.

## Acceptance criteria

- The five screening/compression/testing workflows remain separately runnable from their configs; sensitivity evidence feeds the candidate list used by both screening workflows and compression.
- The minimal YOLO trainer trains from its model config and emits a checkpoint at the configured output. Compression uses only its explicitly configured checkpoint.
- Filterwise rankings/curves and dedicated cluster-size comparisons remain independently auditable and resumable where currently supported.
- Compression emits valid FP32/FP16/INT8 ONNX for dense and configured pruned candidates without ModelOpt, preserving calibration safeguards, provenance, row isolation, and one base export per candidate.
- Exact target-layer importance parity is tested; measured scoring time improves without changing candidate ordering or pruning output.
- The testing matrix discovers models solely from configured directory/glob, supports Python/C++/both, resumes only matching completed results, and records dataset/backend/device provenance.
- Relevant unit/integration suites pass. The current stale RTX layer-list assertion is corrected or removed based on config-retention evidence.
- A target-side Jetson smoke run builds and executes at least one FP32, FP16, and INT8 engine from ONNX; any unsupported TensorRT/operator/dependency path is reported explicitly.
- No unrelated user edits or historical experiment outputs are changed.

## Alternatives considered

- Fold cluster-size evaluation into filterwise screening: fewer entry points, but it conflates per-filter response with cluster-size comparison and contradicts the user's direction to retain this as a fifth workflow. Rejected.
- Keep ModelOpt for precision conversion: conflicts with the target's missing ModelOpt dependency and is unnecessary for the demonstrated ONNX Runtime/onnxconverter-common path. Rejected pending full-model and Jetson validation.
- Reduce the number of pruning ratios or precisions to shorten runtime: would weaken the requested compression comparison. Rejected.
- Rewrite screening/pruning algorithms: risks changing research evidence; reuse the existing primitives and remove only orchestration/features outside the retained scope.

## Deferred work

Faster R-CNN-specific sensitivity, pruning, ONNX export, and compression; knowledge distillation; fine-tuning compressed candidates; historical artifact cleanup; full training-system redesign; and any dependency removal that cannot be proven safe by the import audit.
