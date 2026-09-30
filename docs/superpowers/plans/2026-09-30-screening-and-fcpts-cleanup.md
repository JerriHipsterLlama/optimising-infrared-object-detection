# Screening and FCPTS Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the sensitivity and filterwise screens, reduce cluster pruning to a direct cluster-size comparison, remove active FCPTS code and public APIs, and protect existing training workflows.

**Architecture:** Keep three distinct screening stages: model sensitivity provides evidence for candidate layers; `single_layer_performance_screening.py` keeps its per-filter ranking and one-filter-at-a-time accuracy/latency curves on those selected layers; the cluster workflow compares structural cluster sizes on the same layers. Remove global pruning, fine-tuning, winner selection, and Jetson-result merging only from the cluster-size workflow. Remove active FCPTS imports/code while retaining its historical archive and all training entrypoints/configs.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics, YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-compression-matrix-and-testing-design.md`

## Global Constraints

- Preserve model sensitivity screening, including its layer-ranking evidence and graph/dependency exports.
- Preserve `single_layer_performance_screening.py`, its app, configs, tests, per-filter rankings, and one-filter-at-a-time accuracy/latency curves; restrict the YOLOv8n RTX/Orin configs to sensitivity-approved candidate layers.
- Compare configured cluster sizes on the configured candidate layers with dataset accuracy and a baseline.
- Remove global-candidate, fine-tuning, winner-selection, and Jetson-merge stages from the cluster-screening path.
- Remove active FCPTS implementation and public imports; do not remove shared dependencies unless retained-workflow imports prove them unused.
- Leave `archive/legacy_python` untouched and preserve existing training scripts/configs.
- Preserve uncommitted user changes in `.gitignore`, `src/infrared_detection/benchmarking/rtx.py`, related tests, `.vscode/settings.json`, and `apps/interactive_yolov8_accuracy_sensitivity.py`.
- Do not delete historical result artifacts.

## Review Focus

- An empty or malformed candidate-layer/cluster-size config must fail with an actionable validation error; pin in Task 1 config tests.
- A baseline failure must mark dependent probes skipped without hiding the baseline error; pin in Task 1 workflow tests.
- A failure in one layer/cluster probe must not prevent other configured probes from being recorded; pin in Task 1 workflow tests.
- Removing FCPTS must not break imports of physical structured-pruning APIs; pin in Task 2 public-import tests.
- The filterwise workflow must retain its exact per-filter ranking/K-1 curve behavior while targeting only the approved YOLOv8n layers; pin in Task 4 config and workflow tests.
- Both training families and sensitivity CLI/configs must remain callable/present after cleanup; pin in Task 3 preservation tests.

---

### Task 1: Make cluster screening a single-purpose cluster-size comparison

**Files:**
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `apps/evaluate_cluster_pruning.py`
- Modify: `configs/experiments/cluster_pruning_rtx_screening.yaml`
- Remove or consolidate after migrating settings: `configs/experiments/cluster_pruning_evaluation.yaml`
- Test: `tests/unit/test_cluster_workflow.py`

**Interfaces:**
- Consumes: existing `ClusterEvaluationAdapters` model-loading, evaluation, safe-layer, probe, export, and statistics behavior.
- Produces: `run_cluster_screening(config_path: Path, dry_run: bool = False, adapters: ClusterScreeningAdapters | None = None) -> list[Metrics]`; each result contains a baseline or probe candidate identity, layer, cluster size, ratio, metrics, status, and failure reason.

- [ ] **Step 1: Write failing tests** named `test_planned_rows_include_only_baseline_and_configured_probes`, `test_cluster_screening_records_each_layer_size_probe`, `test_baseline_failure_skips_all_probes`, `test_probe_failure_does_not_abort_remaining_rows`, and `test_invalid_screening_config_fails_before_loading_models`. Assert there are no `global` rows and probe rows preserve candidate layer, cluster size, accuracy, and structural-reduction fields.
- [ ] **Step 2: Run the focused tests** with `python -m pytest tests/unit/test_cluster_workflow.py -q`; confirm the new tests fail against the old global/fine-tuning workflow.
- [ ] **Step 3: Implement `ClusterScreeningAdapters` and `run_cluster_screening`** in `cluster_workflow.py`. Plan only a baseline plus the cartesian product of configured candidate layers, cluster sizes, and probe ratios; keep per-row failure isolation and the current manifest/CSV evidence. Remove global-pruning, fine-tuning, Orin selection, Jetson metric merge, and their unused adapter methods from this screen.
- [ ] **Step 4: Update the CLI and canonical config.** Remove `--screen-only`; the command always runs the retained screening stage. Use the same sensitivity-approved Backbone/Neck candidate layers listed in Task 4 and the compression configs, and keep cluster size 8 among the comparison values. Remove the obsolete duplicate cluster config only after all retained settings have moved to the canonical config.
- [ ] **Step 5: Run focused tests** with `python -m pytest tests/unit/test_cluster_workflow.py tests/unit/test_cluster_candidates.py tests/unit/test_cluster_probe.py -q`; all must pass, including the new failure-isolation cases.
- [ ] **Step 6: Commit** the cluster-screening refactor separately.

### Task 2: Remove active FCPTS implementation and references

**Files:**
- Modify: `src/infrared_detection/compression/pruning/__init__.py`
- Modify: `apps/compress.py`
- Remove: `src/infrared_detection/compression/pruning/fcpts/`
- Remove: `src/infrared_detection/compression/pruning/legacy/`
- Remove: `tests/unit/test_fcpts_calibration.py`
- Remove: `tests/unit/test_fcpts_representative_data.py`
- Test: `tests/unit/test_structured_pruning.py`

**Interfaces:**
- Consumes: existing dependency-graph, importance, cluster-selection, structural-probe, and channel-pruning APIs.
- Produces: `infrared_detection.compression.pruning` exports only active physical structured-pruning APIs; no active CLI message points users to FCPTS.

- [ ] **Step 1: Add failing tests** named `test_pruning_package_import_does_not_load_fcpts` and `test_physical_pruning_exports_remain_available`. Assert the package imports its supported APIs without importing `pruning.fcpts` or `pruning.legacy`.
- [ ] **Step 2: Run the focused tests** with `python -m pytest tests/unit/test_structured_pruning.py -q`; confirm the import test fails before removing the eager import.
- [ ] **Step 3: Remove FCPTS imports/exports and active modules.** Update `__all__` to retain only the physical pruning interfaces. Replace the stale `apps/compress.py` error text with the supported structured-pruning guidance. Do not alter `archive/legacy_python` or remove packages based only on FCPTS usage.
- [ ] **Step 4: Remove FCPTS-specific tests and search active source/config/docs** for `fcpts`/`FCPTS`; retain only historical archive references and any explicit migration note that does not advertise a runnable active API.
- [ ] **Step 5: Run focused tests** with `python -m pytest tests/unit/test_structured_pruning.py -q`, and verify the import search shows no active FCPTS dependency.
- [ ] **Step 6: Commit** FCPTS removal separately.

### Task 3: Add preservation checks for sensitivity analysis and training

**Files:**
- Create: `tests/unit/test_training_entrypoints.py`
- Test existing: `tests/unit/test_accuracy_sensitivity.py`
- Test existing: `tests/unit/test_cluster_workflow.py`
- Preserve unchanged: `apps/train.py`, `src/infrared_detection/models/yolov8/training.py`, `src/infrared_detection/models/faster_rcnn/training.py`, `configs/models/yolov8.yaml`, `configs/models/faster_rcnn.yaml`, sensitivity app/config, and graph/dependency export APIs.

**Interfaces:**
- Consumes: existing `apps.train.build_parser()` and the two supported model-family configs.
- Produces: regression coverage that both training subcommands still resolve to their current handlers and configs, and that sensitivity analysis remains separately runnable.

- [ ] **Step 1: Write failing preservation tests** named `test_training_cli_keeps_yolov8_and_faster_rcnn_families`, `test_training_default_configs_exist`, and `test_sensitivity_entrypoint_and_config_remain_present`. Assert the CLI includes `yolov8` and `faster-rcnn`, their default config paths exist, and the sensitivity app/config plus graph export helpers remain present.
- [ ] **Step 2: Run those tests** with `python -m pytest tests/unit/test_training_entrypoints.py tests/unit/test_accuracy_sensitivity.py -q`; the new preservation tests should pass against current behavior.
- [ ] **Step 3: Keep training and sensitivity behavior unchanged.** If cleanup reveals imports used by those workflows, retain or move the shared helper instead of removing it. Do not add pruning or distillation to training.
- [ ] **Step 4: Run preservation tests** with `python -m pytest tests/unit/test_training_entrypoints.py tests/unit/test_accuracy_sensitivity.py -q` and verify both entrypoints’ `--help` output without starting training.
- [ ] **Step 5: Commit** the preservation tests and any strictly necessary shared-helper relocation separately.

### Task 4: Preserve filterwise screening on the approved candidate layers

**Files:**
- Preserve unchanged: `src/infrared_detection/evaluation/single_layer_performance_screening.py`, `apps/single_layer_performance_screening.py`, `tests/unit/test_single_layer_performance_screening.py`
- Modify: `configs/experiments/single_layer_performance_yolov8n_rtx.yaml`
- Modify: `configs/experiments/single_layer_performance_yolov8n_orin.yaml`
- Preserve unchanged: `configs/experiments/single_layer_performance_yolov8n_cpu.yaml`, `configs/experiments/single_layer_performance_yolov8m_rtx.yaml`, `configs/experiments/single_layer_performance_yolov8m_orin.yaml`
- Test: `tests/unit/test_single_layer_performance_screening.py`, `tests/unit/test_cluster_workflow.py`

**Interfaces:**
- Consumes: `run_single_layer_performance_screening(config_path: str | Path, *, adapters: ScreeningAdapters | None = None, on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]` and `resolve_layer_patterns(model, patterns)`.
- Produces: the existing per-filter ranked K-1 curve remains available for sensitivity-approved YOLOv8n candidate layers; its rows continue to record layer, filter rank, removed/remaining counts, accuracy, latency, and structural status.

- [ ] **Step 1: Add failing tests** named `test_yolov8n_filterwise_configs_match_approved_candidates` and `test_filterwise_curve_keeps_per_filter_rows_for_selected_layers`. Assert the RTX/Orin configs select only the sixteen approved paths (backbone: `model.4.m.1.cv1.conv`, `model.5.conv`, `model.6.m.0.cv1.conv`, `model.6.m.1.cv1.conv`, `model.7.conv`, `model.8.cv2.conv`, `model.8.m.0.cv1.conv`, `model.9.cv2.conv`; neck: `model.12.m.0.cv1.conv`, `model.15.m.0.cv1.conv`, `model.16.conv`, `model.18.cv2.conv`, `model.18.m.0.cv1.conv`, `model.19.conv`, `model.21.cv2.conv`, `model.21.m.0.cv1.conv`) and retain one row per tested filter rank.
- [ ] **Step 2: Run** `python -m pytest tests/unit/test_single_layer_performance_screening.py -q`; verify the config-scope test fails against the broad wildcard patterns while the existing K-1 curve behavior remains green.
- [ ] **Step 3: Restrict only the YOLOv8n RTX/Orin `screening.layer_patterns`** to the approved candidate paths. Keep the implementation, CLI, resume/artifact behavior, tests, and CPU/YOLOv8m configs. The curve remains a per-filter ranking/accuracy/latency screen; cluster-size selection is answered by Task 1’s distinct cluster screen.
- [ ] **Step 4: Run** `python -m pytest tests/unit/test_single_layer_performance_screening.py tests/unit/test_cluster_workflow.py tests/unit/test_accuracy_sensitivity.py -q`; confirm the filterwise module and the sensitivity-to-candidate workflow remain intact.
- [ ] **Step 5: Commit** the candidate-layer scope update separately.
