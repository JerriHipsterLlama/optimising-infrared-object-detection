# Cluster Evaluation Production Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the cluster-pruning workflow so production candidates preserve physically pruned topology, use valid execution devices without fabricating Orin provenance, prune complete low-importance clusters, isolate artifacts, and require measured Orin results before selection.

**Architecture:** Keep `ClusterEvaluationAdapters` as the orchestration boundary. Strengthen the default Ultralytics adapters with a guarded custom `DetectionTrainer`, model-path validation, and artifact isolation; strengthen workflow rows and merge validation so logical target provenance, execution device, structural validation, and native benchmark provenance remain distinct and auditable.

**Tech Stack:** Python 3.10+, pytest 9.0.2, PyTorch 2.9.1, torch-pruning, Ultralytics 8.4.7, PyYAML.

## Global Constraints

- Write focused regression tests before every production correction and observe the expected RED result.
- Keep `C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection-cluster-evaluation\src` first on `PYTHONPATH` for every test command.
- Fail closed when topology preservation, Orin provenance, structural reduction, export isolation, or required native latency measurements cannot be established.
- Preserve the pre-existing untracked `docs/superpowers/plans/2026-07-20-cluster-pruning-evaluation.md` file and unrelated worktree state.
- Functional exported-model inference parity remains future work; record export structural/size validation explicitly.

---

### Task 1: Topology-preserving training and execution-device separation

**Files:**
- Modify: `tests/unit/test_cluster_workflow.py`
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `configs/experiments/cluster_pruning_evaluation.yaml`

**Interfaces:**
- Produces: `_topology_preserving_trainer(pruned_model, trainer_cls=None) -> type`, a guarded custom `DetectionTrainer` class whose `get_model(cfg, weights, verbose)` returns the exact physically pruned module.
- Consumes: `targets.orin_target == "jetson_orin_nano"` and `targets.orin_execution_device`, defaulting to `"0"`.

- [x] Add tests proving the custom trainer returns the pruned module, rejects an incompatible `get_model` API, and is passed to `model.train`.
- [x] Run those tests and confirm they fail because the guarded trainer does not exist and generic training is still selected.
- [x] Implement the minimal trainer factory/API guard and call `model.train(trainer=..., device=orin_execution_device, ...)`.
- [x] Add tests proving default non-dry execution rejects an Orin claim off-target without injected adapters, while injected adapters receive execution device `"0"` and rows retain logical target `jetson_orin_nano`.
- [x] Run the focused tests and confirm they pass.

### Task 2: Concrete safe layers and complete low-importance clusters

**Files:**
- Modify: `tests/unit/test_cluster_workflow.py`
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `configs/experiments/cluster_pruning_evaluation.yaml`

**Interfaces:**
- Consumes: `compute_channel_importance(model)` and `plan_low_importance_clusters(scores, cluster_size, protected_layers, max_clusters_per_layer=1)`.
- Produces: concrete Conv2d safe-layer validation and sequential application of the requested number of complete low-importance `ClusterSpec` objects.

- [x] Add tests proving C2f/container paths are rejected, concrete Conv2d paths are accepted, the lowest-scoring complete cluster is used, partial clusters are never generated, and multiple requested clusters are applied sequentially.
- [x] Run the tests and confirm the current `range(count)` implementation fails them.
- [x] Replace index-prefix pruning with importance computation and one-complete-cluster planning per sequential pruning step.
- [x] Set safe-layer defaults to `model.2.cv1.conv`, `model.4.cv1.conv`, and `model.6.cv1.conv`.
- [x] Run the focused tests and confirm they pass.

### Task 3: Collision-safe IDs and isolated export artifacts

**Files:**
- Modify: `tests/unit/test_cluster_workflow.py`
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`

**Interfaces:**
- Produces: `_append_row` allocation that loops until an unused suffix is found.
- Produces: `_export_yolo` behavior that exports from an output-local checkpoint copy/save and returns an artifact located inside the requested candidate directory.

- [x] Add a third-duplicate regression test and export tests with a pre-existing source-adjacent artifact.
- [x] Run the tests and confirm duplicate `-2` allocation and source-adjacent exporter output fail.
- [x] Implement the unique-ID loop and output-local export staging/copying without overwriting the source-adjacent artifact.
- [x] Run the focused tests and confirm they pass.

### Task 4: Measured Orin merge and validation-gated selection

**Files:**
- Modify: `tests/unit/test_cluster_workflow.py`
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `README.md`
- Modify: `deploy/jetson/README.md`

**Interfaces:**
- Requires: exact `candidate_id`, exact Jetson Orin Nano provenance, finite numeric `latency_p50_ms`, and finite numeric `latency_p95_ms` before setting `hardware_benchmarked=True`.
- Produces: row/manifest `benchmark_path`, `benchmark_provenance`, and explicit `export_validation_status`; selected rows must have passed export structural/size validation.

- [x] Add tests for missing/non-numeric latency rejection, hardware-only field preservation, benchmark binding, and selection exclusion when export validation has not passed.
- [x] Run the tests and confirm the current device-only merge and selection filter fail.
- [x] Implement measurement validation, provenance/path binding, explicit export validation state, and selection gating.
- [x] Run the focused tests and confirm they pass.

### Task 5: Documentation, report, full verification, and commit

**Files:**
- Modify: `.superpowers/sdd/task-4-report.md`
- Modify: `README.md`
- Modify: `deploy/jetson/README.md`

**Interfaces:**
- Produces: an auditable report containing confirmed root causes, bounded fixes, exact RED/GREEN/full-suite commands and results, commit hash, and residual concerns.

- [x] Update workflow documentation and the report without claiming exported-model inference parity.
- [x] Run focused tests, then `python -m pytest -q` with the isolated worktree `src` first on `PYTHONPATH`.
- [x] Run configuration parsing, compile checks, `git diff --check`, and inspect `git diff`/`git status` for unrelated files.
- [x] Request a focused code review, resolve Critical/Important findings, rerun full verification, and commit only relevant files.
