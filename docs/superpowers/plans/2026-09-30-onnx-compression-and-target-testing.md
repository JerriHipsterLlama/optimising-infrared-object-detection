# ONNX Compression and Target Testing Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce FP32, FP16, and INT8 ONNX artifacts for dense and structured-pruned candidates, then evaluate each artifact with Python and/or a target-local TensorRT/Jetson test path.

**Architecture:** First create one base FP32 ONNX export per candidate, then use NVIDIA ModelOpt offline ONNX conversion/calibration for FP16 and INT8 Q/DQ siblings. Keep compression independent of TensorRT, quality evaluation, and timing. Add a directory-driven testing matrix that validates ONNX artifacts with Python and builds TensorRT engines locally from those ONNX files before Jetson C++ latency runs.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics, ONNX, NVIDIA ModelOpt ONNX quantization, TensorRT `trtexec`, Jetson C++ benchmark runner, YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-compression-matrix-and-testing-design.md`

**Prerequisite:** Complete `docs/superpowers/plans/2026-09-30-screening-and-fcpts-cleanup.md` first so structured-pruning imports no longer eagerly load FCPTS.

## Global Constraints

- Preserve the approved 16 Backbone/Neck candidate layers, cluster size 8, and configured pruning ratios.
- Consume the candidate layers established by sensitivity analysis and preserve the separate per-filter screening workflow; do not replace either screening stage with compression.
- Produce FP32, FP16, and INT8 ONNX per candidate; six current candidates yield six base exports and up to eighteen precision ONNX artifacts.
- Use ModelOpt for FP16 ONNX conversion and calibrated INT8 Q/DQ ONNX conversion; do not use TensorRT during compression.
- Never use held-out test images for INT8 calibration; record calibration source and settings.
- Testing discovers ONNX artifacts from a configured directory; Python supplies quality metrics and Jetson C++ supplies native timing.
- TensorRT engines are built on the device that will execute them; `.pt` input is first exported to ONNX; do not transfer engines between Windows/RTX and Linux/Jetson.
- Preserve the existing RTX backend/test edits; make additive, scoped changes and do not overwrite unrelated user worktree changes.
- Focal and Global Knowledge Distillation remain out of scope.

## Review Focus

- Candidate artifacts with an absent or ambiguous precision label must be rejected rather than built under a guessed precision; pin in Task 3 discovery/config tests.
- Empty calibration data or calibration pointed at the test split must fail before ModelOpt quantization; pin in Task 2 config/data-reader tests.
- A ModelOpt failure for one precision must leave sibling precision outputs and manifest rows intact; pin in Task 1 matrix failure-isolation tests.
- Missing TensorRT or Jetson runner must fail only the affected backend row, preserving Python metrics and other candidates; pin in Task 3 adapter tests.
- ONNX external-weight sidecars and unsupported opset/parser cases must be represented accurately and fail with artifact-specific diagnostics; pin in Task 3 model-discovery/build tests.

---

### Task 1: Refactor compression into one source export plus per-precision artifacts

**Files:**
- Modify: `src/infrared_detection/evaluation/compression_matrix.py`
- Modify: `apps/evaluate_compression_matrix.py`
- Test: `tests/unit/test_compression_matrix_workflow.py`
- Test: `tests/unit/test_compression_matrix.py`

**Interfaces:**
- Consumes: `build_pruned_checkpoint(config, output_dir, requested_ratio)` and `_export_checkpoint_to_onnx(checkpoint, config, output_dir)`.
- Produces: `CompressionMatrixAdapters.convert_precision(source_onnx: Path, precision: str, output_onnx: Path, config: Mapping[str, Any]) -> Mapping[str, Any]`; `run_compression_matrix(config_path, dry_run=False, adapters=None) -> list[Metrics]` returns a manifest row per candidate/precision.

- [ ] **Step 1: Write failing tests** named `test_compression_exports_base_onnx_once_per_candidate`, `test_precision_outputs_use_deterministic_candidate_directories`, `test_compression_does_not_build_engines_or_evaluate_models`, and `test_precision_conversion_failure_preserves_sibling_rows`. Assert six current candidate exports, up to eighteen precision rows/artifacts, one checkpoint prune per ratio, and no TensorRT/evaluation/benchmark adapter calls.
- [ ] **Step 2: Run the focused tests** with `python -m pytest tests/unit/test_compression_matrix_workflow.py tests/unit/test_compression_matrix.py -q`; confirm the new expectations fail against the current coupled workflow.
- [ ] **Step 3: Refactor `CompressionMatrixAdapters` and `run_compression_matrix`** to cache one pruned checkpoint and one base ONNX per candidate, then write `candidate/{fp32,fp16,int8}/model.onnx`. Keep independent row statuses, source/checkpoint provenance, conversion logs, artifact paths, and resumability keyed by candidate plus precision/config fingerprint. Remove engine build, accuracy evaluation, selected-candidate logic, and latency benchmark from this command.
- [ ] **Step 4: Update compression YAMLs** `configs/experiments/compression_matrix.yaml`, `configs/experiments/rtx_compression_matrix.yaml`, and `configs/experiments/jetson_compression_matrix.yaml` to select `[fp32, fp16, int8]`, keep pruning parameters, and remove runtime/device/workspace/test settings from compression. Preserve user-specific paths and candidate layers.
- [ ] **Step 5: Run focused tests** with `python -m pytest tests/unit/test_compression_matrix_workflow.py tests/unit/test_compression_matrix.py -q`; verify the deterministic directory layout and six-base-export invariant.
- [ ] **Step 6: Commit** compression orchestration/config changes separately.

### Task 2: Implement ModelOpt FP16/INT8 ONNX conversion and calibration input

**Files:**
- Create: `src/infrared_detection/compression/quantization/onnx_precision.py`
- Modify: `src/infrared_detection/compression/quantization/__init__.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/unit/test_compression_matrix_workflow.py`
- Test: create `tests/unit/test_modelopt_onnx_precision.py`

**Interfaces:**
- Consumes: one FP32 ONNX file per candidate plus validated calibration settings from the compression config.
- Produces: `prepare_precision_onnx(source: Path, output: Path, precision: str, *, calibration_reader: Any | None = None, calibration_method: str = "entropy") -> dict[str, Any]`; FP32 copies the source into its durable output path, FP16 uses ModelOpt ONNX autocast, and INT8 uses `modelopt.onnx.quantization.quantize(..., quantize_mode="int8", calibration_data_reader=..., calibration_method=..., output_path=...)`.

- [ ] **Step 1: Add failing tests** named `test_fp32_reuses_source_graph`, `test_fp16_calls_modelopt_autocast_and_checks_output`, `test_int8_uses_streaming_calibration_reader`, `test_calibration_paths_cannot_resolve_to_test_split`, `test_empty_calibration_data_fails_before_quantize`, and `test_modelopt_error_is_preserved_in_variant_row`. Inject/mock ModelOpt APIs so unit tests do not require the package or GPU.
- [ ] **Step 2: Run** `python -m pytest tests/unit/test_modelopt_onnx_precision.py -q`; confirm missing module/API tests fail before implementation.
- [ ] **Step 3: Implement `prepare_precision_onnx` and a deterministic image calibration reader.** Read a sorted, configured subset from `data.calibration_image_dir`; use the same resize/letterbox, channel order, normalization, input name, and shape as the exported YOLO ONNX input; stream batches rather than retaining the whole dataset in memory. Support the ModelOpt INT8 calibration methods `max` and `entropy`, defaulting to `entropy`, and record sample count, image-size/preprocessing settings, method, input/output path, and ModelOpt output.
- [ ] **Step 4: Scope ModelOpt as an optional compression extra** in `pyproject.toml` and document its installation and supported compression command in `README.md`; leave base training dependencies unchanged.
- [ ] **Step 5: Run** `python -m pytest tests/unit/test_modelopt_onnx_precision.py tests/unit/test_compression_matrix_workflow.py -q`; all conversion, validation, and failure-reporting tests must pass.
- [ ] **Step 6: Commit** ModelOpt conversion and dependency/docs changes separately.

### Task 3: Add directory-driven model testing with target-local engine builds

**Files:**
- Create: `src/infrared_detection/evaluation/model_testing_matrix.py`
- Create: `apps/evaluate_model_testing_matrix.py`
- Create: `configs/experiments/model_testing_matrix.yaml`
- Modify as needed: `src/infrared_detection/benchmarking/rtx.py`
- Modify as needed: `src/infrared_detection/benchmarking/jetson.py`
- Use existing: `deploy/jetson/src/main.cpp`, `deploy/jetson/src/benchmark_runner.cpp`
- Test: create `tests/unit/test_model_testing_matrix.py`
- Test: `tests/unit/test_rtx_benchmarking.py`

**Interfaces:**
- Produces: `load_testing_config(path: str | Path) -> dict[str, Any]`, `discover_onnx_models(directory: Path, pattern: str) -> list[Path]`, `planned_test_rows(config) -> list[dict[str, Any]]`, and `run_model_testing_matrix(config_path, dry_run=False, adapters=None) -> list[dict[str, Any]]`.
- Adapter boundary: `evaluate_python(onnx_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]`; `build_engine(onnx_path: Path, engine_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]`; `benchmark_jetson(engine_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]`.
- Precision is read from validated `/fp32/`, `/fp16/`, or `/int8/` artifact paths (or equivalent explicit metadata); ambiguous files are rejected.

- [ ] **Step 1: Write failing tests** named `test_discovery_is_sorted_and_respects_configured_glob`, `test_ambiguous_precision_is_rejected`, `test_python_backend_reports_detection_metrics`, `test_jetson_backend_builds_engine_from_onnx_then_invokes_runner`, `test_both_backend_joins_metrics_by_candidate`, `test_backend_failure_isolated_per_model`, and `test_engine_artifact_is_not_reused_across_targets`. Assert Python returns mAP50, mAP50-95, precision, recall, per-class AP; Jetson reports latency/runtime provenance without invented quality metrics.
- [ ] **Step 2: Run** `python -m pytest tests/unit/test_model_testing_matrix.py -q`; confirm the new APIs and behaviors fail before implementation.
- [ ] **Step 3: Implement config loading, ONNX discovery, precision/path validation, candidate identity, and manifest/CSV rows** in `model_testing_matrix.py`. Validate backend choices (`python`, `jetson_cpp`, `both`), dataset split/image-size/threshold settings, TensorRT workspace, C++ runner path, image directory, warm-up, and iteration count before running.
- [ ] **Step 4: Implement Python evaluation** using the existing ONNX-capable Ultralytics loading path and `evaluation.detection_metrics.evaluate_yolo`; preserve per-class AP and record Python timing separately if enabled.
- [ ] **Step 5: Implement target-local TensorRT and Jetson runner adapters.** Build from the precision-specific ONNX with local `trtexec`, save engines under `<testing.output_dir>/<hardware_label>/<candidate_id>/<precision>/model.engine`, and invoke the configured `jetson_benchmark` executable with engine, test-image directory, output JSON, warm-up, and iteration count. Use strongly typed ONNX build mode where required by TensorRT 11; do not add precision flags that re-quantize the already-typed ONNX. Do not pass `.pt` checkpoints or use a transferred engine. Preserve and adapt the existing RTX utility interface without overwriting the user’s current backend/test edits.
- [ ] **Step 6: Run** `python -m pytest tests/unit/test_model_testing_matrix.py tests/unit/test_detection_evaluation.py tests/unit/test_rtx_benchmarking.py -q`; all backend, metric-separation, and per-row-failure tests must pass.
- [ ] **Step 7: Commit** the directory-driven test matrix and runner integration separately.

### Task 4: End-to-end config and non-hardware verification

**Files:**
- Modify: `configs/experiments/model_testing_matrix.yaml`
- Modify: `README.md`
- Test: `tests/unit/test_compression_matrix_workflow.py`
- Test: `tests/unit/test_model_testing_matrix.py`

**Interfaces:**
- Consumes: compression artifact directory from Task 1 and test-matrix CLI from Task 3.
- Produces: documented RTX Python and Jetson C++ invocation examples, dry-run results, and a testable contract for the eventual short Jetson hardware acceptance run.

- [ ] **Step 1: Add a dry-run integration test** named `test_compression_manifest_artifacts_are_discoverable_by_testing_matrix`. Create a synthetic manifest and precision ONNX paths for dense plus one pruned candidate; assert all expected test rows are planned without importing TensorRT, ModelOpt, CUDA, or Ultralytics.
- [ ] **Step 2: Run** `python -m pytest tests/unit/test_compression_matrix_workflow.py tests/unit/test_model_testing_matrix.py -q` and `python apps/evaluate_compression_matrix.py --config configs/experiments/jetson_compression_matrix.yaml --dry-run`; confirm six candidates × three precisions are represented.
- [ ] **Step 3: Document the handoff commands and limitations.** Explain that `.pt` exports to ONNX during compression, ModelOpt produces precision-specific ONNX, each target builds its own TensorRT engine, Python measures quality, and Jetson C++ measures native latency. Explicitly defer KD.
- [ ] **Step 4: Run the complete relevant unit subset** with `python -m pytest tests/unit/test_compression_matrix_workflow.py tests/unit/test_compression_matrix.py tests/unit/test_modelopt_onnx_precision.py tests/unit/test_model_testing_matrix.py tests/unit/test_detection_evaluation.py tests/unit/test_rtx_benchmarking.py -q`; all must pass. Do not claim hardware validation until the user runs the Jetson acceptance matrix.
- [ ] **Step 5: Commit** the integration test and usage documentation separately.
