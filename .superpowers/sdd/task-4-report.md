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
