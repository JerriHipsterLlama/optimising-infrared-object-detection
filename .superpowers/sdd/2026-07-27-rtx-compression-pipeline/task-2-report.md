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

- `c6a250c` — `Build structured-pruned RTX checkpoint`

## Concerns

- The actual structured-pruning run still depends on the selected filterwise manifest and a usable Ultralytics/torch-pruning environment; this task intentionally does not run or alter the active filterwise experiment.
- The isolated worktree has no local virtual environment, so the parent repository environment was used with `PYTHONPATH=src`.

## Round 1 semantic correction

The original Task 2 implementation incorrectly used the already-pruned filterwise checkpoint as the input to structural pruning. Under the user-approved clarification, that checkpoint is evidence only.

- `select_filterwise_candidate` remains a read-only evidence lookup and validation helper.
- `build_pruned_checkpoint` now verifies the selected evidence row and path, then loads only dense `model.checkpoint` as the pruning input.
- The YAML configuration now explicitly declares `candidate_layers`, positive `cluster_size`, `prune_ratio`, and `importance: minimum_weight`.
- Each configured dense-model layer is pruned by the requested ratio rounded down to complete low-importance clusters; before/after metrics compare the dense source against final `pruned.pt`.
- Regression coverage makes the evidence checkpoint loader raise if invoked and verifies two configured layers are each pruned through the dependency-graph adapter.

Exact focused verification:

```powershell
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q
```

```text
...................                                                      [100%]
19 passed in 4.51s
```

`git diff --check` exited successfully. The user-edited Task 2 plan remains uncommitted and outside this correction's scope.

## Round 2 ratio-grid correction

The single-ratio wording in the prior correction has been replaced. The Task 2 configuration now contains the candidate grid and acceptance budget only:

- `cluster_size: 8`
- explicit `candidate_layers`
- `prune_ratios: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]`
- `allowed_map50_95_drop: 0.02`
- `importance: minimum_weight`

`build_pruned_checkpoint` accepts optional `requested_ratio` solely to deterministically build one member of that configured grid. It rejects requests outside the grid and does not choose a research ratio. Task 4 will evaluate the grid and select the highest-compression candidate whose measured mAP50_95 drop is within the configured budget. Filterwise checkpoints remain evidence only; all physical pruning starts from dense `model.checkpoint`.

The planner validates the ratio grid and non-negative accuracy budget and records both in structured-pruning provenance. Tests cover invalid grids and invalid budgets, dense-source pruning across each configured layer, evidence-checkpoint non-use, and the requested-grid-member helper.

Exact focused verification:

```powershell
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q
```

```text
....................                                                     [100%]
20 passed in 4.79s
```

## Final candidate-layer scope correction

The RTX configuration now uses exactly the five agreed representative pruning layers:

- `model.0.conv`
- `model.2.cv2.conv`
- `model.4.cv2.conv`
- `model.6.cv2.conv`
- `model.8.cv2.conv`

The configuration regression test asserts this complete ordered list. `prune_ratios`, `allowed_map50_95_drop`, the dense-source pruning behavior, and read-only filterwise evidence handling remain unchanged.

Exact focused verification:

```powershell
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q
```

```text
....................                                                     [100%]
20 passed in 4.60s
```

## Round 2 reviewer fix: zero complete clusters

For a requested ratio that yields zero complete `cluster_size` groups in a configured layer (for example, 16 output filters with cluster size 8 at ratios 0.05–0.30), the builder now records that layer with an empty `prune_indices` list and includes it in `skipped_layers`. It continues pruning every later eligible configured layer. The build raises only when every configured layer yields zero complete clusters, avoiding both whole-candidate failure and silent forced over-pruning.

Regression coverage uses a 16-filter first layer and a 32-filter second layer at ratio 0.30. It verifies that the first layer is recorded as skipped and the second layer removes exactly one low-importance eight-filter cluster.

Exact focused verification:

```powershell
$env:PYTHONPATH='src'; ..\..\.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q
```

```text
.....................                                                    [100%]
21 passed in 4.68s
```
