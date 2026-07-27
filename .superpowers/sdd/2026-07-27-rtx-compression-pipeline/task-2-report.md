# Task 2 implementation report

## Status

Complete. Task 2 adds explicit filterwise-candidate selection and produces a physically structured-pruned `pruned.pt` without modifying the filterwise manifest or its checkpoint.

## Implementation

- Added `select_filterwise_candidate(manifest_path, layer, filters_removed)`. It requires exactly one matching `screened_in` row with a checkpoint path, rejecting unreadable, missing, ambiguous, failed, and incomplete candidates.
- Added `build_pruned_checkpoint(config, output_dir)`. It loads the selected checkpoint, computes Minimum-Weight (L1) filter scores, removes exactly `filter_removal_count` lowest-scoring output filters through `run_filterwise_probe` and its dependency graph, saves `pruned.pt`, reloads it, and writes `pruning_summary.json` with before/after parameter counts and serialized sizes.
- Updated the RTX matrix configuration to use the explicit `filter_removal_count: 8` provenance field required for deterministic candidate selection. The official precision matrix remains exactly FP32/FP16/INT8.
- Added tests for layer/count matching, failed-row rejection even when a path is present, physical checkpoint output/reload, exact selected indices, and recorded metrics.

## TDD evidence

Initial manifest-selection test run before implementation:

```text
ImportError: cannot import name 'select_filterwise_candidate' from 'infrared_detection.evaluation.compression_matrix'
```

Initial checkpoint-builder test run before implementation:

```text
ImportError: cannot import name 'build_pruned_checkpoint' from 'infrared_detection.evaluation.compression_matrix'
```

Explicit-count configuration test run before configuration update:

```text
KeyError: 'filter_removal_count'
1 failed, 12 passed in 4.52s
```

## Tests run

```powershell
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q
```

```text
...............                                                          [100%]
15 passed in 4.68s
```

`git diff --check` also exited successfully. Git emitted only the repository's existing LF-to-CRLF conversion warnings.

## Commits

- Pending commit: Task 2 implementation, tests, configuration update, and this report.

## Concerns

- The actual structured-pruning run still depends on the selected filterwise manifest and a usable Ultralytics/torch-pruning environment; this task intentionally does not run or alter the active filterwise experiment.
- The isolated worktree has no local virtual environment, so the parent repository environment was used with `PYTHONPATH=src`.
