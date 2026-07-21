# Task 3 Report: Config-Driven Two-Stage Cluster Workflow

## Implementation

Implemented `run_cluster_evaluation(config_path, dry_run=False, adapters=None)` and the `ClusterEvaluationAdapters` dependency boundary. The workflow:

- plans a baseline, one-cluster RTX probes, and surviving global candidates from YAML;
- keeps dry runs to YAML planning and artifact writing only, without invoking model, data, CUDA, TensorRT, or checkpoint adapters;
- evaluates the baseline, preserves an explicit failure row for each failed probe/global candidate, and continues remaining candidates;
- profiles successful probes on the RTX screening device and evaluates/profiles global candidates on the Orin target;
- fine-tunes, reloads the reported best checkpoint, validates, exports, profiles, and classifies each successful global candidate;
- writes `candidates.csv` through `write_metrics_csv` and `manifest.json` through `write_experiment_manifest`;
- guarantees distinct candidate IDs, including when a configuration repeats a candidate shape.

The CLI app accepts `--config` and `--dry-run`, and the supplied YAML declares the checkpoint, dataset YAML, image size, seed, protected/safe layers, cluster sizes, probe/global ratios, short fine-tune duration, output directory, RTX screening device, and Orin target.

## Changed Files

- `src/infrared_detection/evaluation/cluster_workflow.py`
- `apps/evaluate_cluster_pruning.py`
- `configs/experiments/cluster_pruning_evaluation.yaml`
- `tests/unit/test_cluster_workflow.py`
- `tests/integration/test_cluster_evaluation_app.py`

## RED Evidence

The requested system-Python command could not collect tests because its Python installation does not contain pytest:

```text
python -m pytest tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py -q
Exit code: 1
C:\Python314\python.exe: No module named pytest
```

Using the available project virtual environment with this branch's `src` first on `PYTHONPATH`, the initial RED command failed for the intended reason:

```text
..\optimising-infrared-object-detection\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py -q
Exit code: 1
ImportError: cannot import name 'cluster_workflow' from 'infrared_detection.evaluation'
```

After the first GREEN cycle, a self-review exposed missing RTX profiling for probe rows. A test was added first and failed as expected:

```text
Exit code: 1
1 failed, 3 passed
KeyError: 'latency_p50_ms'
```

## GREEN Evidence

Final focused verification used:

```text
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
..\optimising-infrared-object-detection\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py -q
....                                                                     [100%]
4 passed in 0.24s
```

Additional final checks passed:

```text
git diff --check
..\optimising-infrared-object-detection\.venv\Scripts\python.exe -m py_compile src\infrared_detection\evaluation\cluster_workflow.py apps\evaluate_cluster_pruning.py
..\optimising-infrared-object-detection\.venv\Scripts\python.exe -c "import yaml; ... validate cluster_pruning_evaluation.yaml"
```

## Self-Review

- Verified dry-run uses `_planned_rows` and writes artifacts before adapters are constructed or called.
- Verified failures are caught independently for probes and globals, while the baseline row is retained.
- Added de-duplication at row insertion so candidate IDs remain unique even for repeated/sanitized configuration values.
- Verified probes use RTX profiling and global rows use the Orin target before classification.
- Kept tests focused on orchestration behavior through injected adapters; they do not require checkpoints, data, CUDA, TensorRT, or Ultralytics execution.
- No code-review subagent capability was available in this session, so review was performed directly against the requirements and the exact changed files.

## Concerns

- The default TensorRT profiler is local-only; an actual Orin deployment needs an injected remote/Jetson-capable profiling adapter (or execution on the Orin). The workflow records such adapter failures per candidate rather than aborting the experiment.
- Full infrared-data training and physical TensorRT/Orin measurements remain hardware/manual validation work; the automated suite deliberately exercises the injected-adapter orchestration contract.

## Corrective Pass: Final Verification

Exact final focused test command and result:

```text
$env:PYTHONPATH = 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection-cluster-evaluation\src'
C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py -q
...........                                                              [100%]
11 passed in 4.67s
```

Additional TDD regression results:

```text
tests/unit/test_cluster_workflow.py::test_each_preplanned_row_is_resolved_when_safe_layers_or_candidate_shapes_repeat
1 passed in 5.45s

tests/unit/test_cluster_workflow.py::test_candidate_artifact_must_match_the_baseline_export_format
1 passed in 0.10s
```

Remaining concerns:

- The automated tests use injected adapters and artifact suffixes to validate orchestration and same-format comparisons. A physical Orin deployment must still perform the real export, profiling, and dataset evaluation on the target hardware.
- The existing local-only TensorRT profiling adapter remains unsuitable for remote Orin measurements unless replaced or run on the Orin.
