# Task 2 report: classify candidates and select deterministic winners

## Implementation

Added a pure evaluation module with the requested interfaces:

- `classify_candidate(baseline_map50_95, candidate_map50_95)` classifies candidates using absolute mAP50-95 limits:
  - `primary_feasible` when the candidate is within 0.01 of baseline, inclusive;
  - `exploratory_feasible` when the candidate is within 0.02, inclusive, but outside the primary limit;
  - `rejected_accuracy` otherwise.
- `select_cluster_candidates(rows)` returns separate `primary` and `exploratory` winners, excluding rejected rows.
- Ranking is deterministic by `serialized_bytes`, then measured `latency_p50_ms`, then descending `map50_95`. Missing latency sorts after measured latency. The input iterable is materialized so both classifications can be selected from generators as well as lists.

The exact boundary comparison uses direct candidate thresholds rather than subtracting floating-point values, so an exact 0.01 drop is treated as inclusive.

## Changed files

- `src/infrared_detection/evaluation/cluster_candidates.py` — new implementation.
- `src/infrared_detection/evaluation/__init__.py` — exports both public functions.
- `tests/unit/test_cluster_candidates.py` — boundary, ranking, missing-latency, and separate exploratory-winner tests.

The requested report is this file. The pre-existing untracked `docs/superpowers/plans/` directory was not modified or staged.

## RED evidence

Command, using the shared virtual environment and `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest tests/unit/test_cluster_candidates.py -q
```

Result before implementation:

```text
ModuleNotFoundError: No module named 'infrared_detection.evaluation.cluster_candidates'
1 error during collection
```

The initial repository-local environment check also showed that no local `.venv` existed and system Python had no pytest; the shared environment resolved that execution issue.

## GREEN evidence

Required focused command:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest tests/unit/test_cluster_candidates.py tests/unit/test_detection_evaluation.py -q
```

Final result:

```text
........                                                                 [100%]
8 passed in 0.07s
```

`git diff --check` also passed.

## Self-review

- Accuracy thresholds are absolute and inclusive at both boundaries.
- Missing or `None` latency cannot outrank measured latency.
- Only the established `latency_p50_ms` field is used for final ranking; no host or non-Orin latency field is consulted.
- Primary and exploratory winners are independent; exploratory candidates never replace the primary winner.
- Empty feasible groups return `None` for the corresponding result.
- Rows are returned by identity, without mutation or copying.

## Concerns

The module assumes each candidate row has the required `serialized_bytes` and `map50_95` fields and that their values are numeric, matching the task’s row contract. It does not add validation for malformed rows because validation was outside Task 2’s requested scope.

## Review fixes

### Changes

- Classification now converts the two supplied float values to decimal strings before calculating the drop. This preserves the required inclusive decimal limits even when binary float arithmetic would make an exact 0.01 or 0.02 difference slightly larger.
- The final ranking tie-breaker is now ascending `candidate_id`, after serialized size, missing/measured latency ordering, latency, and descending mAP50-95. Earlier precedence is unchanged.
- No generic safety fields, structural checks, or hardware-source filtering were added. Task 1 owns the physical pruning boundary; Tasks 3–4 own final Orin authority and benchmark merging.

### RED evidence

Added regression coverage for:

- `classify_candidate(0.0102, 0.0002) == "primary_feasible"` and the corresponding exact 0.02 exploratory boundary;
- a complete equal-size, equal-latency, equal-mAP tie where `cluster-16` must beat `cluster-32` by `candidate_id`, despite appearing second in input order.

Before the fixes, the Task 2 test command failed as expected:

```text
FAILED ...[0.0102-0.0002-primary_feasible]
AssertionError: 'exploratory_feasible' != 'primary_feasible'

FAILED test_full_ranking_ties_use_candidate_id
AssertionError: selected 'cluster-32' instead of 'cluster-16'
2 failed, 7 passed
```

Command:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest tests/unit/test_cluster_candidates.py -q
```

### GREEN evidence

Covering Task 2 verification command:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest tests/unit/test_cluster_candidates.py tests/unit/test_detection_evaluation.py -q
```

Result:

```text
...........                                                              [100%]
11 passed in 0.06s
```

### Review self-check

- Decimal-boundary tests fail against the prior float implementation and pass with the decimal conversion.
- The `candidate_id` test fails against input-order tie selection and passes with the final key.
- Existing size, latency, missing-latency, and mAP precedence tests remain green.
