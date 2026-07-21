# Task 4 Report: Merge authoritative Orin results and document execution

## Result

Implemented `merge_jetson_metrics(rows, benchmark_path)`. It requires one matching `candidate_id`, updates only native hardware fields, and refreshes `candidates.csv` plus `manifest.json` in the benchmark JSON directory. The documentation now describes RTX screening, native Orin/TensorRT benchmarking, and the merge procedure. It explicitly states that only Orin data ranks final candidates.

## RED

Initial required command:

```powershell
python -m pytest tests/unit/test_cluster_workflow.py -q
```

Result: failed before collection because `C:\Python314\python.exe` has no `pytest` module.

After creating the repository-local `.venv`, installing `pytest==9.0.2` and `PyYAML==6.0.3`, and installing the repository editable without dependencies, the RED command was:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py -q
```

Result: failed during collection exactly as intended:

```text
ImportError: cannot import name 'merge_jetson_metrics' from 'infrared_detection.evaluation.cluster_workflow'
```

## GREEN

The new behavior test was run after implementation:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py::test_merge_orin_metrics_updates_only_hardware_fields -q
```

Result: `1 passed in 0.13s`.

The requested focused suite was then run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py tests/unit/test_jetson_benchmark_config.py tests/unit/test_evaluation_apps.py -q
```

Result: `11 passed, 4 failed`. The four failures are existing cluster-workflow paths that import the torch-dependent pruning validator before their assertions. The local verification environment deliberately contains only pytest and PyYAML. Root cause confirmed with:

```powershell
& .\.venv\Scripts\python.exe -c "from infrared_detection.evaluation.cluster_workflow import _validate_structural_reduction; _validate_structural_reduction({'parameter_count': 10, 'serialized_bytes': 4}, {'parameter_count': 8, 'serialized_bytes': 2})"
```

Result: `ModuleNotFoundError: No module named 'torch'` from `infrared_detection.compression.distillation.detection_kd`.

## Changed files

- `src/infrared_detection/evaluation/cluster_workflow.py`
  - Added the constrained native hardware-field merge and artifact rewrite.
- `tests/unit/test_cluster_workflow.py`
  - Added the RED/GREEN merge test, including artifact persistence assertions.
- `README.md`
  - Added the cluster-pruning RTX-screening, Orin-native benchmark, candidate-ID annotation, and merge procedure.
- `deploy/jetson/README.md`
  - Added the required native-Orin benchmark and merge handoff for global candidates.

## Self-review

- The merge matches exactly one row by `candidate_id` and raises when the identifier is absent or ambiguous.
- Only latency, FPS, peak memory, power, energy, and temperature keys are copied; mAP and serialized size are not copied.
- Artifact regeneration uses the existing CSV writer and selection-manifest convention.
- Documentation directs each native JSON into the experiment artifact directory so regeneration targets the correct `candidates.csv` and `manifest.json`.
- The pre-existing untracked `docs/superpowers/plans/` directory was not modified.

## Concerns

- Full focused-suite verification remains blocked locally until the project’s torch-dependent dependencies are installed. The new Task 4 merge test passes independently.
- The native benchmark output must be annotated with the exact `candidate_id` before merging because the current native command-line interface does not receive a candidate identifier.

## Final corrective verification

The corrective tests were first run in RED:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py::test_merge_jetson_metrics_rejects_missing_or_wrong_orin_provenance tests/unit/test_cluster_workflow.py::test_valid_orin_merge_reclassifies_failed_global_and_selects_only_benchmarked_rows -q
```

Result: `4 failed in 0.28s` as expected before the provenance and Orin-gated selection implementation: the three provenance cases did not raise, and the failed global row remained failed.

The final Task 4 relevant tests were run with:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py -k "merge_orin_metrics or merge_jetson_metrics or valid_orin_merge or manifest_waits_for_authoritative_orin_global_winner" tests/unit/test_jetson_benchmark_config.py tests/unit/test_evaluation_apps.py -q
```

Result: `6 passed, 13 deselected in 0.17s`.

The complete requested Task 4 suite was also run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py tests/unit/test_jetson_benchmark_config.py tests/unit/test_evaluation_apps.py -q
```

Result: `16 passed, 3 failed in 1.08s`. The remaining failures are `test_successful_probes_are_profiled_on_rtx_before_global_orin_evaluation`, `test_baseline_and_global_metrics_preserve_authoritative_orin_provenance`, and `test_same_format_baseline_artifact_and_physical_reduction_are_required`; they depend on the project’s torch-based pruning validation stack, which is absent from the shared venv. No full-green claim is made.

## Production correction: final-review findings

This section supersedes the earlier incomplete-environment concern above. The shared project virtual environment contains the pinned production stack, and the complete suite now runs with the isolated worktree's `src` first on `PYTHONPATH`.

### Confirmed root causes

- Ultralytics 8.4.7 `Model.train` called `DetectionTrainer.get_model(cfg=self.model.yaml, weights=self.model)`, whose default implementation rebuilt `DetectionModel` from the dense YAML before loading compatible weights. A structurally pruned module could therefore be replaced by dense topology.
- The configured YOLOv8n paths `model.2`, `model.4`, and `model.6` are C2f containers, not directly prunable convolutions; the concrete targets are `model.2.cv1.conv`, `model.4.cv1.conv`, and `model.6.cv1.conv`.
- The logical label `orin` was passed to Ultralytics as a CUDA device even though Ultralytics rejects it.
- Production `_structural_probe` discarded channel-importance planning and removed `range(count)`, including a partial cluster when the requested cluster approached layer width.
- `_append_row` counted only the unsuffixed identifier, so the third duplicate reused the second duplicate's `-2` suffix.
- Ultralytics exports were returned from exporter-selected locations, allowing source-checkpoint-adjacent artifacts and cross-candidate collisions.
- The native merge trusted device provenance without requiring actual latency measurements, and final selection did not explicitly require completed structural/size export validation.

### Bounded fixes

- Added a guarded custom `DetectionTrainer` for pinned Ultralytics 8.4.7. Its `get_model` returns the exact physically pruned module and fails closed if the trainer signature or `Model.train` override path is incompatible. The workflow reloads `best.pt` and validates reloaded parameter and same-format serialized-size reduction before a candidate can proceed.
- Separated `targets.orin_target: jetson_orin_nano` from `targets.orin_execution_device: '0'`. Ultralytics evaluation/fine-tuning receives only the execution device. Default non-dry execution rejects an Orin claim on a non-Orin host unless adapters are injected for remote execution; RTX remains probe-screening evidence.
- Updated safe-layer defaults to concrete Conv2d paths and reject missing, protected, detection-head, and non-Conv2d configured targets with clear errors.
- Replaced prefix-index pruning with `compute_channel_importance` plus `plan_low_importance_clusters`. Each requested complete cluster is planned and applied sequentially; partial clusters and whole-layer removal are rejected.
- Allocated duplicate candidate IDs with a collision loop.
- Staged path checkpoints and save-capable pruned models inside each candidate directory before export, then copied the reported artifact to `candidate.<format>` in that directory. Source-adjacent artifacts are not used or overwritten.
- Required exact candidate ID, exact Jetson Orin Nano provenance, and positive finite numeric p50/p95 latency before `hardware_benchmarked=True`. Only hardware fields are copied; benchmark path/provenance are bound into rows and manifest records.
- Added `export_validation_status` and require `passed` together with a real Orin benchmark before winner selection. Functional exported-model inference parity remains explicitly out of scope for this correction.

### TDD evidence

All commands used:

```powershell
$env:PYTHONPATH = 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection-cluster-evaluation\src'
& 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest <focused selectors> -q
```

- Baseline before changes: `118 passed, 12 warnings in 16.46s`.
- Trainer/device/safe-layer RED: `6 failed, 1 passed in 5.10s`; failures were the expected missing custom trainer/API guard, logical-target device propagation, absent off-target guard, and non-prunable C2f acceptance.
- Trainer/device/safe-layer GREEN: `7 passed in 4.03s`.
- Importance/cluster/ID/export RED: `6 failed in 4.28s`; failures showed `(0, 1)` instead of low-importance `(1, 2)`, one four-index spec instead of two complete specs, an accepted partial cluster, duplicate `-2`, and two non-isolated export paths.
- Importance/cluster/ID/export GREEN: `6 passed in 4.08s`.
- Merge/validation RED: `11 failed, 1 passed in 4.46s`; failures showed missing export-validation state, absent benchmark path/provenance, accepted missing/non-numeric latency, and selection of an unvalidated export.
- Merge/validation GREEN: `12 passed in 4.17s`.
- Complete workflow module: `35 passed in 4.50s`.
- Complete repository suite with worktree `src` first: `136 passed, 12 warnings in 15.70s`; the final pre-commit verification repeated this successfully as `136 passed, 12 warnings in 15.82s`. The warnings are the existing PyTorch pin-memory deprecations from dataset tests.

### Remaining concern

Automated tests validate trainer selection, topology guards, reloaded parameter/serialized-size reduction, artifact isolation, orchestration, and provenance contracts without running the infrared training dataset or physical TensorRT hardware. Functional inference parity of each exported model and native Orin measurements remain required execution evidence, not claims made by this correction.
