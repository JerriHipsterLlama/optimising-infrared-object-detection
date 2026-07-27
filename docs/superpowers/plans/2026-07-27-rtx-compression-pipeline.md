# RTX Compression Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configuration-driven RTX 3070 pipeline that creates and benchmarks dense and structured-pruned YOLO TensorRT variants at FP32, FP16, and INT8.

**Architecture:** A read-only filterwise-manifest selector feeds a physical structured-pruning artifact builder. A precision builder exports each checkpoint to ONNX/TensorRT, and a runner evaluates accuracy and TensorRT latency into a single resumable manifest. Existing pruning, export, quantization, and benchmark helpers remain reusable; KD is not part of this implementation.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics YOLO, Torch-Pruning, TensorRT/trtexec, PyYAML, pytest.

## Global Constraints

- Use the same CAMEL validation split, image size, batch size, confidence threshold, and IoU threshold for every variant.
- Official deployment precisions are FP32, FP16, and INT8; do not label the PyTorch int16 payload as a TensorRT INT16 engine.
- Keep the active `runs/experiments/filterwise_rtx_screening/` output read-only.
- Preserve partial results and record per-variant failures without discarding successful variants.
- Require explicit candidate layer, cluster size, and pruning ratio; never silently infer the research candidate.
- Do not add knowledge-distillation training in this plan.

---

### Task 1: Define the compression-matrix configuration and result schema

**Files:**
- Create: `configs/experiments/rtx_compression_matrix.yaml`
- Create: `src/infrared_detection/evaluation/compression_matrix.py`
- Test: `tests/unit/test_compression_matrix_workflow.py`

**Interfaces:**
- `load_compression_config(path: str | Path) -> dict[str, Any]`
- `planned_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]`
- `write_compression_manifest(output_dir: Path, rows: list[Mapping[str, Any]]) -> None`

- [ ] **Step 1: Write failing tests**

```python
def test_planned_variants_include_dense_and_pruned_precision_matrix():
    rows = planned_variants({
        "pruning": {"enabled": True, "candidate_id": "cluster-8-ratio-0.25"},
        "precisions": ["fp32", "fp16", "int8"],
    })
    assert [row["variant_id"] for row in rows] == [
        "dense-fp32", "dense-fp16", "dense-int8",
        "cluster-8-ratio-0.25-fp32",
        "cluster-8-ratio-0.25-fp16",
        "cluster-8-ratio-0.25-int8",
    ]

def test_manifest_writer_preserves_failed_and_successful_rows(tmp_path):
    write_compression_manifest(tmp_path, [{"variant_id": "dense-fp32", "status": "completed"}])
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["rows"][0]["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py -q`

Expected: FAIL because the compression-matrix workflow module and functions do not yet exist.

- [ ] **Step 3: Implement the minimal planner and manifest writer**

Implement strict validation for `precisions` values `fp32`, `fp16`, and `int8`; create dense rows plus pruned rows; initialize status and provenance fields; write `manifest.json` and `results.csv` using existing artifact-writing conventions.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add configs/experiments/rtx_compression_matrix.yaml src/infrared_detection/evaluation/compression_matrix.py tests/unit/test_compression_matrix_workflow.py
git commit -m "Add RTX compression matrix planning"
```

### Task 2: Build and validate the selected structured-pruned checkpoint

**Files:**
- Modify: `src/infrared_detection/evaluation/compression_matrix.py`
- Modify: `src/infrared_detection/compression/pruning/cluster_selection.py`
- Test: `tests/unit/test_compression_matrix_workflow.py`
- Test: `tests/unit/test_cluster_selection.py`

**Interfaces:**
- `build_pruned_checkpoint(config: Mapping[str, Any], output_dir: Path) -> Path`
- `select_filterwise_candidate(manifest_path: Path, layer: str, filters_removed: int) -> Mapping[str, Any]`

- [ ] **Step 1: Write failing tests**

```python
def test_select_filterwise_candidate_requires_matching_layer_and_count(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(json.dumps({"rows": [{"layer": "model.2.cv2.conv", "filters_removed": 8, "status": "screened_in", "checkpoint_path": "candidate.pt"}]}))
    row = select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)
    assert row["checkpoint_path"] == "candidate.pt"

def test_select_filterwise_candidate_rejects_failed_row(tmp_path):
    manifest = tmp_path / "filterwise.json"
    manifest.write_text(json.dumps({"rows": [{"layer": "model.2.cv2.conv", "filters_removed": 8, "status": "failed"}]}))
    with pytest.raises(ValueError, match="not usable"):
        select_filterwise_candidate(manifest, "model.2.cv2.conv", 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py -k candidate -q`

Expected: FAIL because candidate selection is not implemented.

- [ ] **Step 3: Implement candidate selection and pruning**

Load the selected source checkpoint, compute Minimum-Weight scores, remove exactly the configured number of complete low-importance filters or clusters through the existing dependency graph, save `pruned.pt`, reload it, and record parameter-count and serialized-size changes. Reject missing, failed, or incomplete filterwise rows before pruning.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_compression_matrix_workflow.py tests\unit\test_cluster_selection.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/infrared_detection/evaluation/compression_matrix.py src/infrared_detection/compression/pruning/cluster_selection.py tests/unit/test_compression_matrix_workflow.py tests/unit/test_cluster_selection.py
git commit -m "Add selected structured-pruning artifact builder"
```

### Task 3: Add local TensorRT precision export and benchmark adapters

**Files:**
- Create: `src/infrared_detection/benchmarking/rtx.py`
- Modify: `src/infrared_detection/export/__init__.py`
- Modify: `src/infrared_detection/benchmarking/jetson.py`
- Test: `tests/unit/test_rtx_benchmarking.py`

**Interfaces:**
- `build_tensorrt_engine(onnx_path: Path, engine_path: Path, precision: str, calibration_dir: Path | None, workspace_mb: int) -> dict[str, Any]`
- `benchmark_tensorrt_engine(..., device_label: str = "rtx3070") -> dict[str, Any]`

- [ ] **Step 1: Write failing tests**

```python
def test_tensorrt_command_uses_fp16_flag(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command) or _completed_trtexec())
    build_tensorrt_engine(tmp_path / "model.onnx", tmp_path / "model.engine", "fp16", None, 1024)
    assert "--fp16" in commands[0]

def test_int8_build_requires_calibration_directory(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        build_tensorrt_engine(tmp_path / "model.onnx", tmp_path / "model.engine", "int8", None, 1024)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_rtx_benchmarking.py -q`

Expected: FAIL because the local RTX adapter does not yet exist.

- [ ] **Step 3: Implement command construction and normalized parsing**

Build FP32 without a precision flag, FP16 with `--fp16`, and INT8 with `--int8` plus explicit calibration provenance. Parse median and p95 latency when present, compute FPS, include engine size, iterations, warm-up, device label, and the exact command. Keep missing `trtexec` as a clear runtime error.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\test_rtx_benchmarking.py tests\unit\test_model_stats.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/infrared_detection/benchmarking/rtx.py src/infrared_detection/export/__init__.py src/infrared_detection/benchmarking/jetson.py tests/unit/test_rtx_benchmarking.py
git commit -m "Add local RTX TensorRT precision benchmarking"
```

### Task 4: Connect accuracy evaluation, engine generation, and resumable results

**Files:**
- Modify: `src/infrared_detection/evaluation/compression_matrix.py`
- Create: `apps/evaluate_compression_matrix.py`
- Modify: `tests/unit/test_evaluation_apps.py`
- Test: `tests/integration/test_compression_matrix_app.py`

**Interfaces:**
- `run_compression_matrix(config_path: str | Path, dry_run: bool = False) -> list[dict[str, Any]]`
- CLI: `python apps/evaluate_compression_matrix.py --config ... [--dry-run]`

- [ ] **Step 1: Write failing tests**

```python
def test_matrix_app_exposes_dry_run_and_config_arguments():
    result = subprocess.run([sys.executable, "apps/evaluate_compression_matrix.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout

def test_matrix_runner_keeps_success_when_int8_variant_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(compression_matrix, "build_tensorrt_engine", _fake_engine_builder)
    monkeypatch.setattr(compression_matrix, "benchmark_tensorrt_engine", _fake_benchmark)
    rows = run_compression_matrix(write_matrix_config(tmp_path))
    assert any(row["status"] == "completed" and row["precision"] == "fp16" for row in rows)
    assert any(row["status"] == "failed" and row["precision"] == "int8" for row in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_compression_matrix_app.py tests\unit\test_evaluation_apps.py -q`

Expected: FAIL because the new runner and CLI do not yet exist.

- [ ] **Step 3: Implement the runner**

For each planned row, skip only rows already completed with matching provenance; prepare the dense or selected pruned checkpoint; export ONNX; build the requested TensorRT engine; evaluate the associated checkpoint; benchmark the engine; merge all metrics; and write the manifest after every row. Catch exceptions per row and continue.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests\integration\test_compression_matrix_app.py tests\unit\test_evaluation_apps.py tests\unit\test_compression_matrix_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/infrared_detection/evaluation/compression_matrix.py apps/evaluate_compression_matrix.py tests/integration/test_compression_matrix_app.py tests/unit/test_evaluation_apps.py
git commit -m "Add resumable compression matrix runner"
```

### Task 5: Validate the local RTX smoke path and document usage

**Files:**
- Modify: `configs/experiments/rtx_compression_matrix.yaml`
- Modify: `README.md`
- Create: `docs/research/rtx-preliminary-results.md`
- Test: `tests/integration/test_compression_matrix_app.py`

- [ ] **Step 1: Add a dry-run integration assertion**

Assert that the configured experiment plans exactly six rows and never includes `int16` as an official TensorRT precision.

- [ ] **Step 2: Run the dry run**

Run:

```powershell
.venv\Scripts\python.exe apps/evaluate_compression_matrix.py --config configs\experiments\rtx_compression_matrix.yaml --dry-run
```

Expected: six planned rows with dense and structured-pruned FP32/FP16/INT8 variants.

- [ ] **Step 3: Run the local smoke test**

Run:

```powershell
.venv\Scripts\python.exe apps/evaluate_compression_matrix.py --config configs\experiments\rtx_compression_matrix.yaml
```

Expected: each available variant writes a checkpoint/export/engine/metrics row; unavailable TensorRT or calibration functionality is recorded as a row-level failure with a diagnostic.

- [ ] **Step 4: Run the complete test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all existing and new tests pass.

- [ ] **Step 5: Commit documentation and verified configuration**

```powershell
git add configs/experiments/rtx_compression_matrix.yaml README.md docs/research/rtx-preliminary-results.md tests/integration/test_compression_matrix_app.py
git commit -m "Document RTX preliminary compression workflow"
```
