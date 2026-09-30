# Compression and Screening Workflow Simplification — Implementation Plan

## Goal

Implement the approved five-workflow design: retain separate sensitivity screening, filterwise screening, cluster-size screening, ONNX compression, and directory-driven model testing; retain minimal config-driven YOLO training; remove only proven obsolete workflows and dependencies. Preserve the external-baseline contract, native Jetson runner, model/dataset configs, graph exports, user-owned edits, and historical run data.

## Architecture

Work in test-first increments, in dependency order. Keep the existing config-driven app boundaries and reusable primitives. Replace ModelOpt precision conversion with native ONNX tooling; optimize candidate scoring without changing ranking; isolate the cluster-size workflow from unrelated legacy orchestration; make testing-matrix resume decisions depend on input and execution provenance; then remove only code proven unreachable from retained workflows. Finish with host verification and a documented target-side Jetson smoke test. Faster R-CNN training files/config remain available, but its compression path is explicitly deferred.

## Tech Stack

Python, PyTorch/Ultralytics, ONNX, ONNX Runtime static quantization, `onnxconverter-common`, pytest, YAML configs, TensorRT/C++ runner on Jetson.

## Spec link

[`docs/superpowers/specs/2026-09-30-matrix-simplification-design.md`](../specs/2026-09-30-matrix-simplification-design.md)

## Global Constraints

- Preserve the six configured candidates, all pruning ratios, and FP32/FP16/INT8 coverage; export one base ONNX per candidate.
- Compression consumes only the explicit checkpoint in its config. It must not train, evaluate accuracy, build TensorRT engines, or benchmark latency.
- Calibration must use the configured representative non-test split and fail before conversion if invalid or empty.
- Keep output paths/provenance, row-level failure isolation, and resume behavior; never erase sibling outputs when a row fails.
- Preserve sensitivity and filterwise outputs, the separate cluster-size workflow, YOLO training, future Faster R-CNN source/config, native Jetson runner, graph exports, and historical run artifacts.
- Do not alter the existing user-owned `.gitignore`, deletion of `benchmarking/rtx.py`, or `.vscode/` changes. Do not use broad deletion of data, run outputs, or ignored files.
- Add each test to the existing tracked suite even though the user's `.gitignore` ignores `tests/*`; force-add only intentional new test files if needed.
- Each task starts with a failing focused test (RED), proves the intended failure, implements the smallest change, proves GREEN, runs its broader focused suite, and commits only task-owned files. Do not stage unrelated user changes.

## Review Focus

1. ONNX precision conversion preserves graph validity and external input/output types, and INT8 calibration cannot read the test split.
2. Candidate-only scoring exactly matches the existing score values, ordering, selected clusters, and pruned checkpoint behavior.
3. Cluster-size screening remains a distinct auditable CLI and does not silently retain fine-tuning, global-winner selection, or Jetson metric merging.
4. Testing-matrix resume keys cover model content and all result-affecting dataset, backend, input, and runtime settings; stale results rebuild/re-evaluate while unrelated rows survive failures.
5. Cleanup changes are import- and entrypoint-audited; retained YOLO/Faster R-CNN training files, screening, ONNX exports, native runner, configs, and historical data remain untouched and usable.

## Tasks

### Task 1 — Replace ModelOpt ONNX precision conversion

**Files:**

- Modify `src/infrared_detection/quantization/onnx_precision.py`.
- Modify `src/infrared_detection/quantization/__init__.py` only as required by retained public imports.
- Modify `pyproject.toml` compression extra.
- Replace/update `tests/unit/test_modelopt_onnx_precision.py` as `tests/unit/test_onnx_precision.py` (preserve useful behavior assertions, remove ModelOpt-specific expectations).
- Update compression documentation/config comments that name ModelOpt, identified by `rg` before editing.

**Interfaces:** Keep the current public precision-preparation entry point consumed by `evaluation/compression_matrix.py`; retain a precision enum/string contract for `fp32`, `fp16`, and `int8`. FP16 conversion uses `onnxconverter_common.float16.convert_float_to_float16` with `keep_io_types=True`. INT8 conversion uses `onnxruntime.quantization.quantize_static` with QDQ format and an explicit calibration reader. FP32 copies/validates the base graph. Each result is checked using `onnx.checker.check_model` before it is accepted.

**Failing test:** Add focused tests for valid FP32 copy, FP16 converted graph with unchanged external I/O types, INT8 QDQ graph creation using a deterministic calibration reader, invalid/empty/test-split calibration rejection before converter invocation, and row-local converter failure. Assert no test-split files are consumed.

**Verify RED:** Run only the new precision test module and confirm failures identify missing native conversion behavior (not import/setup errors).

**Implementation:** Remove ModelOpt imports/subprocess/autocast from the active ONNX path. Add `onnxconverter-common` to the compression optional dependency directly; use installed ONNX Runtime for static quantization. Record converter name/version, quantization mode, calibration split/source, and calibration sample count in row provenance. Keep existing output naming and avoid modifying base ONNX files.

**Verify GREEN:** Run the focused precision tests, then compression-matrix unit tests. Run the synthetic ONNX conversion test against installed dependencies. Commit only the precision implementation, its tests, dependency metadata, and relevant docs.

### Task 2 — Score only configured pruning candidates

**Files:**

- Modify `src/infrared_detection/evaluation/compression_matrix.py` at `build_pruned_checkpoint`.
- Add parity/regression coverage in `tests/unit/test_compression_matrix_workflow.py` or a focused compression-matrix scoring test module.

**Interfaces:** Preserve `build_pruned_checkpoint(...)` arguments/return behavior and `minimum_weight_scores(module: nn.Conv2d) -> torch.Tensor` in `src/infrared_detection/compression/pruning/importance.py`.

**Failing test:** Compare candidate-layer scores and resulting cluster membership/order against the existing all-module scoring reference for each configured candidate and at least two prune ratios, including ties if the current ranking permits them. Assert checkpoint structure and selected target layer are unchanged.

**Verify RED:** Run the focused test on a small deterministic model and demonstrate the existing implementation's whole-model scan through a test spy or equivalent observable, while score-parity assertions pass against the reference.

**Implementation:** Restrict score computation to the exact candidate convolution for the current variant; preserve score formula, stable tie-breaking, cluster size 8/config value, and pruning masks. Do not change the compression matrix dimensions.

**Verify GREEN:** Run the focused parity test and the full compression-matrix workflow tests. Re-run the recorded checkpoint microbenchmark using the same 16 layers and report both timings without claiming this dominates full matrix runtime. Commit task-owned files only.

### Task 3 — Keep cluster-size screening as a clean, distinct workflow

**Files:**

- Modify `src/infrared_detection/evaluation/cluster_workflow.py` only around the retained `run_cluster_screening` path and helpers proven exclusive to removed orchestration.
- Retain `apps/evaluate_cluster_pruning.py` and its config; update only if the audit finds stale arguments/output names.
- Update `tests/unit/test_cluster_workflow.py` and relevant CLI/config tests.
- Update/remove `configs/experiments/rtx_compression_matrix.yaml` or its stale assertion only after confirming the file's retained status; correct its expected candidate list to the approved 16 when retained.

**Interfaces:** Preserve the `run_cluster_screening` public signature/CLI inputs and configured layer/cluster-size inputs. Do not combine it with `apps/single_layer_performance_screening.py`.

**Failing test:** Cover a configured multi-size run on candidate layers, per-layer/per-size metrics, structural parameter reduction, resumable result output if currently supported, and no calls to fine-tuning, global-winner selection, or Jetson metric merging. Keep existing screening candidate and validation checks.

**Verify RED:** Run the focused cluster-size workflow tests and confirm the new isolation/metric assertions fail for the current mixed orchestration where applicable.

**Implementation:** Reuse existing structural-probe and validation primitives; remove only orchestration and helpers with no retained consumers. Keep sensitivity screening and filterwise-screening code/tests intact. Resolve the stale RTX candidate-list failure from config/import evidence, not by deleting the test blindly.

**Verify GREEN:** Run cluster-workflow and screening suites, then verify both retained screening CLIs still load their configs. Commit only cluster-size workflow, its tests, and directly related config changes.

### Task 4 — Make model-testing resume provenance-aware

**Files:**

- Modify `src/infrared_detection/evaluation/model_testing_matrix.py`.
- Modify `apps/evaluate_model_testing_matrix.py` only if CLI wiring needs new config fields.
- Modify the model-testing matrix config schema/examples and `tests/unit/test_model_testing_matrix.py` (or its tracked equivalent).

**Interfaces:** Keep configured `models.directory` plus glob discovery, Python/C++/both backend selection, existing backend metric fields, and current adapter boundary. Add a stable row fingerprint derived from ONNX identity/content and result-affecting dataset, input size, backend, runtime/device settings, and relevant evaluation options. Persist manifest/CSV per completed row.

**Failing test:** Prove an identical completed row skips engine build and evaluation; changing ONNX content or any relevant config/backend option invalidates that row; adding a model evaluates only the new row; failure in one model/backend preserves successful sibling rows and outputs.

**Verify RED:** Run the focused matrix module and confirm old resume logic incorrectly reuses stale work or repeats completed work.

**Implementation:** Add fingerprint/provenance comparison before reuse. Save each row as it completes and isolate backend/model errors. Do not hard-code model names or conflate Python accuracy with native C++ latency.

**Verify GREEN:** Run testing-matrix unit and CLI/config tests with fake adapters; verify generated output includes dataset/backend/device provenance. Commit only matrix implementation, tests, and configs/docs required for the schema.

### Task 5 — Audit and remove only obsolete workflow surface

**Files:** Use `rg`, Python import searches, app/config inventories, and tracked-file lists before selecting deletion targets. Likely candidates include legacy `apps/compress.py`, old `src/infrared_detection/training/compression_matrix.py`, FCPTS-only modules/tests, distillation-only modules/tests, and duplicate generic evaluate/report/frontier/benchmark paths. Do not assume every candidate is removable. Preserve `apps/train.py`, both model training source/configs, screening apps, `apps/export.py`, the two matrix apps, the native Jetson runner, dataset/model configs, and graph exports.

**Interfaces:** Keep `apps/train.py`'s YOLO subcommand config-driven and ensure it emits the configured checkpoint path. Compression continues to require an explicit checkpoint config value. Keep current Faster R-CNN training/config available but unsupported by YOLO compression. Preserve model/dataset config paths used by retained CLIs.

**Failing test:** Add/adjust entrypoint smoke tests for all five workflows plus YOLO training config resolution; add a guard that compression rejects missing explicit checkpoint rather than launching training or guessing a path. Retain tests for future Faster R-CNN training entrypoint if it remains exposed. Remove only tests whose only subject is a removed path; keep screening tests even when they share a test module with legacy workflow tests.

**Verify RED:** Run the entrypoint smoke tests before deletion and show which obsolete imports/commands are covered; newly added contract tests must fail until the cleanup/wiring is complete.

**Implementation:** Remove dead workflows, FCPTS dependencies, stale docs/configs/tests, and optional dependencies only after proving no retained consumer. Keep `albumentations` if retained Faster R-CNN augmentation imports it. Update README commands and install instructions for the five workflows and compression extra. Preserve user-owned changes and all run/data artifacts.

**Verify GREEN:** Run a static import-reference audit and all retained CLI `--help`/config-load checks. Run YOLO trainer config-resolution test without starting a full training job. Commit only audited deletions, retained-entrypoint changes, tests, configs, docs, and dependencies.

### Task 6 — End-to-end host verification and Jetson handoff

**Files:** Update only the compression/matrix README instructions and, if needed, a concise target smoke-test checklist in existing docs. Do not commit generated checkpoints, engines, runs, or datasets.

**Failing test:** Establish a pre-run check that fails with actionable errors for missing external checkpoint, calibration samples, or ONNX tools; establish output assertions for all three dense precision variants and one pruned candidate.

**Verify RED:** Run checks against an isolated temporary output/config fixture and confirm failures occur before partial/corrupt artifacts are published.

**Implementation:** Run the host unit/integration suites; use the actual supplied YOLO checkpoint to export/convert dense FP32, FP16, and INT8 ONNX and a configured pruned candidate if host resources permit. Validate each with ONNX checker and metadata. Document Jetson commands to build and infer one engine per precision locally, since host validation cannot prove target TensorRT/JetPack support.

**Verify GREEN:** Run the complete relevant test suite, compression config dry-run, full-model ONNX smoke where feasible, and repository status audit. Report exactly which target-side Jetson checks remain pending; do not claim deployability without target evidence. Commit only source/docs/tests/config changes.

## Self-review against the approved spec

- All five distinct workflows are represented; cluster-size comparison remains separate from filterwise screening.
- FP32/FP16/INT8, six base exports, all configured ratios, calibration safeguards, row isolation, and provenance are explicitly preserved.
- The external baseline and minimal YOLO trainer are both specified; no implicit training is allowed.
- Faster R-CNN training/config is retained as future work, with compression support explicitly deferred.
- TensorRT stays on the target testing path; host ONNX validity is not represented as Jetson validation.
- Import auditing precedes cleanup, and user changes/historical artifacts are protected.

## Review Request

Please review this plan and choose the execution method before implementation: **native, step-by-step execution in this task**, or **subagent-driven execution for independent tasks**. No implementation begins until you choose.
