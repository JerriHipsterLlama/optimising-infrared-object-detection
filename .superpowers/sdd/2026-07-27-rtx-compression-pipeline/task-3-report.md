# Task 3 Report: Local RTX TensorRT Build and Benchmark Adapters

## Status

Implemented and verified. Task 3 adds local `trtexec` build and benchmark adapters for the RTX 3070 boundary. Active filterwise outputs were not modified.

## Commits

- Approved base: `dc78f73` (`Skip ineligible structured pruning layers`)
- Task 3 implementation: `9a4b208` (`feat: add local RTX TensorRT adapters`)
- The report is committed in the follow-up documentation commit.

## Implementation

- Added `src/infrared_detection/benchmarking/rtx.py`.
- Added `tests/unit/test_rtx_benchmarking.py`.
- Supported precisions are exactly `fp32`, `fp16`, and `int8`.
- FP32 emits no precision flag; FP16 emits `--fp16`; INT8 emits `--int8` and requires a calibration directory.
- Build results record the exact command, precision, ONNX and engine paths, engine size, calibration provenance, workspace, and `trtexec` output.
- Benchmark results record the exact command, engine size, device label, iterations, warmup, p50, p95 when present, FPS, and raw output.
- Missing `trtexec` raises `RtxToolsUnavailable` with an installation/PATH diagnostic.
- TensorRT is not imported; subprocess and tool lookup remain injectable in unit tests.

## TDD evidence

### Red

Command:

```text
C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe -m pytest tests\unit\test_rtx_benchmarking.py -q
```

Output:

```text
ERROR collecting tests/unit/test_rtx_benchmarking.py
ModuleNotFoundError: No module named 'infrared_detection.benchmarking.rtx'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

The expected failure was caused by the missing production module before implementation.

### Green

Exact required focused test selection, run with the isolated worktree source path first on `PYTHONPATH`:

```text
$env:PYTHONPATH = 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.worktrees\rtx-compression-pipeline\src'; & 'C:\Users\gerth\Documents\Engineering\optimising-infrared-object-detection\.venv\Scripts\python.exe' -m pytest tests\unit\test_rtx_benchmarking.py tests\unit\test_model_stats.py -q
```

Output:

```text
........                                                                 [100%]
8 passed in 2.71s
```

Additional focused RTX-only run:

```text
.....                                                                    [100%]
5 passed in 0.09s
```

`git diff --check` completed without output. Scope verification reported only:

```text
?? src/infrared_detection/benchmarking/rtx.py
?? tests/unit/test_rtx_benchmarking.py
```

## Concerns

- The requested `.venv\Scripts\python.exe` path does not exist inside the linked worktree. The repository-root virtualenv was used instead.
- That shared virtualenv had the parent checkout on `PYTHONPATH`; the focused tests were rerun with the isolated worktree `src` explicitly selected.
- No local TensorRT runtime was required or exercised; command construction and output normalization are covered with subprocess/tool lookup test doubles.
