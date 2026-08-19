# Single-Layer Performance Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all legacy filterwise sensitivity/screening orchestration with one resumable `single_layer_performance_screening` workflow that produces complete MinimumWeight-ranked accuracy and direct CUDA latency curves for YOLOv8n and YOLOv8m.

**Architecture:** Add exact paper-faithful ranking and a reusable CUDA forward profiler, then implement a dedicated screening module whose side effects are isolated behind adapters. Each selected layer starts from the dense checkpoint, follows one fixed original-filter ranking until one filter remains, and stores only one transient resume checkpoint. The old filterwise modes are removed from the cluster workflow and CLI.

**Tech Stack:** Python 3.12/3.13, PyTorch, CUDA events, Ultralytics 8.4.7, Torch-Pruning, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-single-layer-performance-screening-design.md`

## Global Constraints

- The ranking criterion is exactly `mean(weight.float().square(), dim=(1, 2, 3))` for each convolution output filter.
- Each layer ranking is calculated once from the dense checkpoint and remains fixed.
- Every selected valid layer is pruned from one removed filter through `K - 1`; there is no accuracy early stop.
- The ten approved wildcard patterns are copied verbatim into all four experiment configurations.
- Accuracy and raw PyTorch forward latency are measured for every valid candidate.
- Sensitivity screening must not export ONNX or TensorRT artifacts.
- Storage is bounded to the dense checkpoint, one transient `resume.pt`, and metadata/results.
- Structural topology failures terminate only the affected layer and permanently skip its remaining candidates.
- CUDA OOM and environmental failures stop the run and remain retryable.
- Existing historical run directories and historical design documents are not deleted.
- Existing user changes in `configs/models/yolov8.yaml`, `.pytest-tmp/`, and `Ultralytics/` are unrelated and must not be staged, modified, or removed.

## File Structure

### Create

- `src/infrared_detection/benchmarking/pytorch_cuda.py` — direct CUDA-event forward profiler.
- `src/infrared_detection/evaluation/single_layer_performance_screening.py` — selection, planning, persistence, resume, and screening orchestration.
- `apps/single_layer_performance_screening.py` — thin CLI.
- `configs/experiments/single_layer_performance_yolov8n_rtx.yaml`
- `configs/experiments/single_layer_performance_yolov8m_rtx.yaml`
- `configs/experiments/single_layer_performance_yolov8n_orin.yaml`
- `configs/experiments/single_layer_performance_yolov8m_orin.yaml`
- `tests/unit/test_pytorch_cuda_benchmark.py`
- `tests/unit/test_single_layer_performance_screening.py`
- `tests/unit/test_supervisor_pruning_report.py`

### Modify

- `src/infrared_detection/compression/pruning/importance.py` — add exact MinimumWeight scoring and stable ranking.
- `src/infrared_detection/compression/pruning/__init__.py` — export new ranking functions.
- `src/infrared_detection/evaluation/artifacts.py` — optional leading/trailing CSV columns.
- `src/infrared_detection/evaluation/cluster_workflow.py` — remove legacy sensitivity orchestration only.
- `apps/evaluate_cluster_pruning.py` — remove `--filterwise` and `--full-curve`.
- `tests/unit/test_structured_pruning.py`
- `tests/unit/test_artifacts.py`
- `tests/unit/test_cluster_workflow.py`
- `tests/unit/test_evaluation_apps.py`
- `tools/generate_supervisor_pruning_report.py`
- `README.md`

### Delete

- `configs/experiments/filterwise_rtx_screening.yaml`
- `tools/build_kaggle_filterwise_notebook.py`
- `tests/unit/test_build_kaggle_filterwise_notebook.py`

---

### Task 1: Implement Exact MinimumWeight Ranking

**Files:**
- Modify: `src/infrared_detection/compression/pruning/importance.py`
- Modify: `src/infrared_detection/compression/pruning/__init__.py`
- Modify: `tests/unit/test_structured_pruning.py`

**Interfaces:**
- Produces: `minimum_weight_scores(module: nn.Conv2d) -> torch.Tensor`
- Produces: `rank_filters_by_minimum_weight(module: nn.Conv2d) -> tuple[tuple[int, float], ...]`
- Preserves: `compute_channel_importance(model, graph=None, criterion="l1")`

- [ ] **Step 1: Write failing tests for Equation 3 and stable ranking**

Add tests with an explicitly assigned weight tensor:

```python
def test_minimum_weight_scores_are_mean_squared_filter_weights():
    conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)
    conv.weight.data.copy_(torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[0.0, 0.0], [0.0, 2.0]]],
        [[[2.0, 2.0], [2.0, 2.0]]],
    ]))

    scores = minimum_weight_scores(conv)

    assert scores.tolist() == pytest.approx([1.0, 1.0, 4.0])


def test_minimum_weight_ranking_uses_original_index_as_tie_breaker():
    conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)
    conv.weight.data.copy_(torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[0.0, 0.0], [0.0, 2.0]]],
        [[[2.0, 2.0], [2.0, 2.0]]],
    ]))

    assert rank_filters_by_minimum_weight(conv) == (
        (0, pytest.approx(1.0)),
        (1, pytest.approx(1.0)),
        (2, pytest.approx(4.0)),
    )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_structured_pruning.py -k "minimum_weight" -q
```

Expected: collection/import failure because the new functions do not exist.

- [ ] **Step 3: Implement the minimal scoring and ranking functions**

Add:

```python
def minimum_weight_scores(module: nn.Conv2d) -> torch.Tensor:
    if not isinstance(module, nn.Conv2d):
        raise TypeError("MinimumWeight scoring requires torch.nn.Conv2d.")
    weight = module.weight.detach().to(dtype=torch.float32)
    return weight.square().mean(dim=tuple(range(1, weight.ndim)))


def rank_filters_by_minimum_weight(module: nn.Conv2d) -> tuple[tuple[int, float], ...]:
    scores = minimum_weight_scores(module).cpu().tolist()
    return tuple(sorted(enumerate(scores), key=lambda item: (item[1], item[0])))
```

Export both from the package `__init__.py`. Do not change the legacy L1/L2 criteria consumed by cluster/compression workflows.

- [ ] **Step 4: Run focused and pruning tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_structured_pruning.py tests\unit\test_cluster_probe.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/infrared_detection/compression/pruning/importance.py src/infrared_detection/compression/pruning/__init__.py tests/unit/test_structured_pruning.py
git commit -m "feat: add exact minimum-weight filter ranking"
```

---

### Task 2: Add Research-Oriented CSV Column Ordering

**Files:**
- Modify: `src/infrared_detection/evaluation/artifacts.py`
- Modify: `tests/unit/test_artifacts.py`

**Interfaces:**
- Extends: `write_metrics_csv(path, rows, *, leading_columns=(), trailing_columns=()) -> None`
- Existing two-argument callers retain alphabetical ordering.

- [ ] **Step 1: Write a failing column-order test**

```python
def test_metrics_csv_places_research_columns_first_and_diagnostics_last(tmp_path):
    output = tmp_path / "results.csv"
    write_metrics_csv(
        output,
        [{"error": "boom", "map50_95": 0.8, "candidate_id": "c1", "status": "failed", "zeta": 4, "reason": "x"}],
        leading_columns=("candidate_id", "status", "map50_95"),
        trailing_columns=("reason", "error"),
    )

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["candidate_id", "status", "map50_95", "zeta", "reason", "error"]
```

- [ ] **Step 2: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_artifacts.py -q
```

Expected: `TypeError` because ordering keywords are unsupported.

- [ ] **Step 3: Implement deterministic ordering**

Use this ordering algorithm:

```python
all_keys = {key for row in rows for key in row}
leading = [key for key in leading_columns if key in all_keys]
trailing = [key for key in trailing_columns if key in all_keys and key not in leading]
middle = sorted(all_keys - set(leading) - set(trailing))
keys = [*leading, *middle, *trailing]
```

Keep `_csv_value()` unchanged.

- [ ] **Step 4: Run artifact tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_artifacts.py -q
```

Expected: PASS, including the existing nested-value/newline test.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/infrared_detection/evaluation/artifacts.py tests/unit/test_artifacts.py
git commit -m "feat: order screening result columns"
```

---

### Task 3: Implement Direct PyTorch/CUDA Forward Profiling

**Files:**
- Create: `src/infrared_detection/benchmarking/pytorch_cuda.py`
- Create: `tests/unit/test_pytorch_cuda_benchmark.py`

**Interfaces:**
- Produces protocol: `ForwardTimer` with `synchronize() -> None` and `measure(callable) -> float`
- Produces: `CudaEventTimer`
- Produces: `benchmark_pytorch_cuda_forward(model, example_input, *, warmup, iterations, timer=None) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests with a fake timer**

```python
class FakeTimer:
    def __init__(self, values):
        self.values = iter(values)
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def measure(self, operation):
        operation()
        return next(self.values)


def test_forward_profiler_warms_up_and_reports_distribution():
    model = nn.Identity().eval()
    batch = torch.zeros(1, 3, 8, 8)
    timer = FakeTimer([1.0, 2.0, 4.0, 8.0])

    result = benchmark_pytorch_cuda_forward(
        model, batch, warmup=2, iterations=4, timer=timer
    )

    assert result["latency_mean_ms"] == pytest.approx(3.75)
    assert result["latency_p50_ms"] == pytest.approx(3.0)
    assert result["latency_p95_ms"] == pytest.approx(7.4)
    assert result["fps"] == pytest.approx(1000.0 / 3.75)
    assert result["warmup"] == 2
    assert result["iterations"] == 4
    assert result["input_shape"] == [1, 3, 8, 8]
    assert timer.synchronizations == 2
```

Also test rejection of `warmup < 0`, `iterations <= 0`, and production timing with a non-CUDA input.

- [ ] **Step 2: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_pytorch_cuda_benchmark.py -q
```

Expected: import failure because the profiler module is absent.

- [ ] **Step 3: Implement timer and profiler**

Production event timing must follow:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
operation()
end.record()
end.synchronize()
return float(start.elapsed_time(end))
```

The profiler must:

```python
with torch.inference_mode():
    for _ in range(warmup):
        model(example_input)
    timer.synchronize()
    samples = [timer.measure(lambda: model(example_input)) for _ in range(iterations)]
    timer.synchronize()
```

Use `statistics.mean`, `statistics.median`, and `torch.quantile(torch.tensor(samples), 0.95)` for the summary. If `timer is None`, require `example_input.is_cuda` and instantiate `CudaEventTimer`.

- [ ] **Step 4: Run profiler tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_pytorch_cuda_benchmark.py -q
```

Expected: PASS without requiring CUDA because functional tests inject `FakeTimer`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/infrared_detection/benchmarking/pytorch_cuda.py tests/unit/test_pytorch_cuda_benchmark.py
git commit -m "feat: benchmark pytorch forward latency on cuda"
```

---

### Task 4: Build Layer Resolution, Ranking Records, and Candidate Planning

**Files:**
- Create: `src/infrared_detection/evaluation/single_layer_performance_screening.py`
- Create: `tests/unit/test_single_layer_performance_screening.py`

**Interfaces:**
- Produces dataclass: `FilterRanking(original_index: int, score: float)`
- Produces dataclass: `LayerPlan(name: str, filters_before: int, ranking: tuple[FilterRanking, ...])`
- Produces: `resolve_layer_patterns(model, patterns) -> list[str]`
- Produces: `build_layer_plan(model, layer_name) -> LayerPlan`
- Produces: `physical_index(remaining_original_indices, original_index) -> int`
- Produces: `planned_rows(model_variant, hardware, layer_plans) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing selection and planning tests**

Build a tiny nested fixture whose `named_modules()` contains:

```text
model.6.m.0.cv1.conv
model.6.m.0.cv2.conv
model.6.m.1.cv1.conv
model.6.m.1.cv2.conv
```

Assert:

```python
assert resolve_layer_patterns(model, ["model.6.m.*.cv1.conv"]) == [
    "model.6.m.0.cv1.conv",
    "model.6.m.1.cv1.conv",
]
```

Add tests that unmatched patterns and matches to non-`Conv2d` modules raise `ValueError`. Test that a four-filter layer plans exactly three candidates and that `physical_index([0, 2, 3], 2) == 1`.

- [ ] **Step 2: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -k "patterns or plan or physical" -q
```

Expected: import failure because the workflow module is absent.

- [ ] **Step 3: Implement deterministic selection and plans**

Use `fnmatch.fnmatchcase` and model-order iteration. `build_layer_plan` must call `rank_filters_by_minimum_weight` and reject widths below two.

Candidate IDs use:

```python
slug = re.sub(r"[^A-Za-z0-9]+", "-", layer_name).strip("-")
candidate_id = f"{model_variant}-{hardware}-{slug}-rank-{rank}"
```

Every planned row includes the leading CSV fields from the spec, `status="planned"`, and `filters_remaining = filters_before - filter_rank`.

- [ ] **Step 4: Add supplied-architecture count tests**

Load `yolov8n.pt` and `yolov8m.pt` only when those files exist; mark the tests with `pytest.mark.skipif` otherwise. Assert:

```python
assert (len(n_layers), sum(plan.filters_before - 1 for plan in n_plans)) == (12, 1012)
assert (len(m_layers), sum(plan.filters_before - 1 for plan in m_plans)) == (24, 5352)
```

- [ ] **Step 5: Run focused tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -k "patterns or plan or physical or supplied" -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/infrared_detection/evaluation/single_layer_performance_screening.py tests/unit/test_single_layer_performance_screening.py
git commit -m "feat: plan single-layer performance sweeps"
```

---

### Task 5: Implement Artifact Persistence and Resume State

**Files:**
- Modify: `src/infrared_detection/evaluation/single_layer_performance_screening.py`
- Modify: `tests/unit/test_single_layer_performance_screening.py`

**Interfaces:**
- Produces dataclass: `ResumeState(version, fingerprint, active_layer, next_filter_rank, remaining_original_indices, last_candidate_id)`
- Produces: `experiment_fingerprint(config, checkpoint_path, dataset_path) -> str`
- Produces: `write_screening_artifacts(output_dir, config_path, rows, rankings, state) -> None`
- Produces: `load_screening_artifacts(output_dir, expected_fingerprint) -> tuple[list[dict], dict, ResumeState | None]`
- Produces: `save_pending_checkpoint(model, output_dir) -> Path`
- Produces: `promote_pending_checkpoint(output_dir) -> Path`
- Produces: `clear_resume_checkpoint(output_dir) -> None`

- [ ] **Step 1: Write failing persistence tests**

Tests must prove:

```python
state = ResumeState(
    version=1,
    fingerprint="abc123",
    active_layer="model.6.m.0.cv1.conv",
    next_filter_rank=2,
    remaining_original_indices=(0, 2, 3),
    last_candidate_id="yolov8n-rtx3070-model-6-m-0-cv1-conv-rank-1",
)
write_screening_artifacts(
    output_dir,
    config_path,
    rows=[{"candidate_id": "baseline", "status": "completed"}],
    rankings={"model.6.m.0.cv1.conv": [{"original_index": 1, "score": 0.01}]},
    state=state,
)
assert (output_dir / "results.csv").exists()
assert (output_dir / "manifest.json").exists()
assert (output_dir / "rankings.json").exists()
assert (output_dir / "state.json").exists()
```

Read `results.csv` and assert the first 21 fields and final `reason,error` fields match the specification. Write a saved state with fingerprint `A`, then assert loading with fingerprint `B` raises `ValueError("experiment fingerprint")`.

Add a fake save-capable model and assert `save_pending_checkpoint` creates `resume.pending.pt`, `promote_pending_checkpoint` replaces the prior `resume.pt`, and no pending file remains afterward.

- [ ] **Step 2: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -k "artifact or fingerprint or resume_checkpoint" -q
```

Expected: failures because persistence functions are absent.

- [ ] **Step 3: Implement atomic metadata writes**

Implement `_atomic_write_text(path, text)` with sibling `path.with_suffix(path.suffix + ".tmp")`, `write_text`, then `Path.replace`. Use SHA-256 over canonical JSON containing:

```python
{
    "checkpoint": {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns},
    "dataset": {"path": str(dataset.resolve()), "size": dataset.stat().st_size, "mtime_ns": dataset.stat().st_mtime_ns},
    "model_variant": config["experiment"]["model_variant"],
    "hardware": config["experiment"]["hardware_label"],
    "image_size": config["experiment"]["image_size"],
    "precision": config["runtime"]["precision"],
    "layer_patterns": config["screening"]["layer_patterns"],
    "ranking_version": "minimum-weight-mean-square-v1",
}
```

Write JSON with sorted keys. Use `write_metrics_csv` with the exact leading/trailing schema.

For checkpoint replacement, save the newly pruned model to `resume.pending.pt` and verify it exists with non-zero size. Keep the previous `resume.pt` untouched until accuracy and latency both succeed. After both measurements succeed, atomically replace `resume.pt` with `resume.pending.pt`, then write the completed result and advanced state. Delete `resume.pending.pt` after any failed measurement so the prior valid resume point remains authoritative.

- [ ] **Step 4: Run persistence tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -k "artifact or fingerprint or resume_checkpoint" -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/infrared_detection/evaluation/single_layer_performance_screening.py tests/unit/test_single_layer_performance_screening.py
git commit -m "feat: persist resumable screening state"
```

---

### Task 6: Implement the Independent Complete-Curve Workflow

**Files:**
- Modify: `src/infrared_detection/evaluation/single_layer_performance_screening.py`
- Modify: `tests/unit/test_single_layer_performance_screening.py`

**Interfaces:**
- Produces dataclass: `ScreeningAdapters(load_model, evaluate, prune_filter, profile, stats, save_checkpoint)`
- Produces: `complete_candidate_row(*, layer_plan: LayerPlan, ranking_entry: FilterRanking, filter_rank: int, physical_filter_index: int, remaining_original_indices: Sequence[int], accuracy: Mapping[str, Any], latency: Mapping[str, Any], baseline: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `run_single_layer_performance_screening(config_path, *, adapters=None, on_result=None) -> list[dict[str, Any]]`
- `on_result(row)` receives every newly persisted baseline/candidate row for CLI progress.

- [ ] **Step 1: Write a failing full-sweep test**

Use adapters over a four-filter fake layer. Record load and prune calls. Assert:

```python
rows = run_single_layer_performance_screening(config_path, adapters=adapters)
candidates = [row for row in rows if row["stage"] == "single_layer"]
assert [row["filter_rank"] for row in candidates] == [1, 2, 3]
assert [row["filters_remaining"] for row in candidates] == [3, 2, 1]
assert [row["status"] for row in candidates] == ["completed"] * 3
assert not (output_dir / "resume.pt").exists()
```

Use ranking `[original 2, original 0, original 3, original 1]` and assert physical prune indices `[2, 0, 1]` after index translation.

- [ ] **Step 2: Write a failing independent-layer test**

Use two selected layers and assert the dense checkpoint is loaded once for baseline and once at the start of each layer. Assert no model pruned in layer A is passed into layer B.

- [ ] **Step 3: Write failing resume and failure-semantics tests**

Resume test:

1. Interrupt after rank 2 by raising `RuntimeError("interrupted")` from the injected profiler.
2. Confirm ranks 1 and 2 are persisted and `resume.pt` exists.
3. Restart with working adapters.
4. Assert ranks 1 and 2 are not evaluated again and rank 3 completes.

Structural failure test: raise a dependency/channel mismatch at rank 2; assert rank 2 and all later ranks for that layer are `skipped`, then assert the next layer runs.

OOM test: raise `torch.OutOfMemoryError`; assert the candidate is `failed`, later rows stay `planned`, state remains at the failed rank, and the exception stops the workflow after artifacts are written.

- [ ] **Step 4: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -k "complete_curve or independent or resume or structural or out_of_memory" -q
```

Expected: failures because orchestration is absent.

- [ ] **Step 5: Implement orchestration**

The core loop must follow:

```python
for layer_plan in layer_plans:
    working_model = load_dense_or_resume(layer_plan, state)
    remaining = state.remaining_original_indices if resuming else list(range(layer_plan.filters_before))
    for rank, ranking_entry in enumerate(layer_plan.ranking, start=1):
        if rank < next_filter_rank:
            continue
        physical = physical_index(remaining, ranking_entry.original_index)
        working_model = adapters.prune_filter(working_model, layer_plan.name, physical, config)
        pending_checkpoint = save_pending_checkpoint(working_model, output_dir)
        evaluation_model = adapters.load_model(pending_checkpoint)
        accuracy = adapters.evaluate(evaluation_model, config)
        latency = adapters.profile(working_model, config)
        remaining.remove(ranking_entry.original_index)
        promote_pending_checkpoint(output_dir)
        row = complete_candidate_row(
            layer_plan=layer_plan,
            ranking_entry=ranking_entry,
            filter_rank=rank,
            physical_filter_index=physical,
            remaining_original_indices=remaining,
            accuracy=accuracy,
            latency=latency,
            baseline=baseline,
            stats=adapters.stats(working_model),
        )
        rows_by_id[row["candidate_id"]] = row
        state = ResumeState(
            version=1,
            fingerprint=fingerprint,
            active_layer=layer_plan.name,
            next_filter_rank=rank + 1,
            remaining_original_indices=tuple(remaining),
            last_candidate_id=row["candidate_id"],
        )
        write_screening_artifacts(output_dir, config_path, list(rows_by_id.values()), rankings, state)
        on_result(row) if on_result else None
    clear_resume_checkpoint(output_dir)
```

Do not save or export any candidate artifact other than `resume.pt`. Calculate baseline accuracy and latency once per model/hardware output directory and retain the baseline row across resumes.

Classify structural errors by dependency-graph rejection or the existing channel-mismatch regex. Mark the rest of that layer skipped. For all other exceptions, persist `failed`, keep the last valid checkpoint/state, and re-raise after writing artifacts.

- [ ] **Step 6: Run all new workflow tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```powershell
git add src/infrared_detection/evaluation/single_layer_performance_screening.py tests/unit/test_single_layer_performance_screening.py
git commit -m "feat: run complete independent layer screens"
```

---

### Task 7: Add Production YOLO/CUDA Adapters, CLI, and Four Configurations

**Files:**
- Modify: `src/infrared_detection/evaluation/single_layer_performance_screening.py`
- Create: `apps/single_layer_performance_screening.py`
- Create: `configs/experiments/single_layer_performance_yolov8n_rtx.yaml`
- Create: `configs/experiments/single_layer_performance_yolov8m_rtx.yaml`
- Create: `configs/experiments/single_layer_performance_yolov8n_orin.yaml`
- Create: `configs/experiments/single_layer_performance_yolov8m_orin.yaml`
- Modify: `tests/unit/test_single_layer_performance_screening.py`
- Modify: `tests/unit/test_evaluation_apps.py`

**Interfaces:**
- Produces: `ScreeningAdapters.defaults(config)`
- CLI example: `.venv\Scripts\python.exe apps\single_layer_performance_screening.py --config configs\experiments\single_layer_performance_yolov8n_rtx.yaml`

- [ ] **Step 1: Write failing production-adapter tests**

Monkeypatch Ultralytics `YOLO`, explicit pruning, model stats, and CUDA profiler boundaries. Assert:

- Evaluation receives `data`, `split`, `imgsz=352`, `device`, `conf`, `iou`, and `half=True` for FP16.
- Pruning creates an aligned `[1, 3, 352, 352]` example input and passes one physical index to `run_filterwise_probe`.
- Profiling moves the unwrapped model and example input to the configured CUDA device and precision before calling `benchmark_pytorch_cuda_forward`.
- CUDA unavailability raises a clear `RuntimeError` before baseline evaluation.

- [ ] **Step 2: Write failing CLI/config tests**

Replace the legacy app test with:

```python
def test_single_layer_performance_app_exposes_config_only():
    result = subprocess.run(
        [sys.executable, "apps/single_layer_performance_screening.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--full-curve" not in result.stdout
```

Load all four YAML files and assert the exact ten patterns, `image_size == 352`, `batch_size == 1`, `warmup == 20`, `iterations == 100`, and unique output directories.

- [ ] **Step 3: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py tests\unit\test_evaluation_apps.py -k "production or config or app" -q
```

Expected: failures because production adapters, app, and configs are absent.

- [ ] **Step 4: Implement production adapters**

Use existing helpers as references without importing private cluster workflow functions:

- Load with `YOLO(str(checkpoint))`.
- Unwrap with `getattr(wrapper, "model", wrapper)`.
- Evaluate with `wrapper.val(data=dataset_yaml, split="val", imgsz=352, device=device, conf=conf, iou=iou, batch=1, half=precision == "fp16", verbose=False)` and normalise `map50_95`, `map50`, precision, recall, and per-class AP.
- Prune through `run_filterwise_probe(unwrapped, example_input, layer, (physical_index,))`.
- Profile through `benchmark_pytorch_cuda_forward` on the unwrapped model.
- Collect stats through `collect_model_stats`.
- Save through the wrapper's `save()` method.

The CLI progress callback prints one concise line:

```text
[yolov8n/rtx3070] model.6.m.0.cv1.conv rank 17/63 | remaining=47 | mAP50-95=0.8421 | p50=3.217 ms | completed
```

- [ ] **Step 5: Add the four configs**

Each config contains the exact approved patterns and an isolated output directory under:

```text
runs/experiments/single_layer_performance_screening/{yolov8n|yolov8m}/{rtx3070|jetson_orin_nano}
```

RTX configs use `hardware_label: rtx3070`; Orin configs use `hardware_label: jetson_orin_nano`. Both use CUDA device `0`, FP16, 352 pixels, batch 1, warm-up 20, iterations 100, and the CAMEL validation YAML.

Use the CAMEL-trained checkpoints, not the generic COCO weights:

- YOLOv8n: `models/checkpoints/yolov8/train3/weights/best.pt` (3,011,628 parameters; best recorded CAMEL mAP50-95 0.75408).
- YOLOv8m: `models/checkpoints/yolov8/train4/weights/best.pt` (25,858,636 parameters; best recorded CAMEL mAP50-95 0.80401).

Every config follows this concrete schema, with the model, hardware label, and output directory varied as described above:

```yaml
model:
  checkpoint: models/checkpoints/yolov8/train3/weights/best.pt

data:
  dataset_yaml: data/camel/camel.yaml
  split: val

experiment:
  model_variant: yolov8n
  hardware_label: rtx3070
  image_size: 352
  num_classes: 4
  seed: 7
  batch_size: 1
  output_dir: runs/experiments/single_layer_performance_screening/yolov8n/rtx3070

runtime:
  device: "0"
  precision: fp16
  conf: 0.25
  iou: 0.6

screening:
  warmup: 20
  iterations: 100
  layer_patterns:
    - model.6.m.*.cv1.conv
    - model.6.m.*.cv2.conv
    - model.8.m.*.cv1.conv
    - model.8.m.*.cv2.conv
    - model.12.m.*.cv1.conv
    - model.12.m.*.cv2.conv
    - model.18.m.*.cv1.conv
    - model.18.m.*.cv2.conv
    - model.21.m.*.cv1.conv
    - model.21.m.*.cv2.conv
```

- [ ] **Step 6: Run app/config and workflow tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_single_layer_performance_screening.py tests\unit\test_evaluation_apps.py -q
.venv\Scripts\python.exe apps\single_layer_performance_screening.py --help
```

Expected: PASS and help output containing only `--config` for the new app.

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/infrared_detection/evaluation/single_layer_performance_screening.py apps/single_layer_performance_screening.py configs/experiments/single_layer_performance_yolov8n_rtx.yaml configs/experiments/single_layer_performance_yolov8m_rtx.yaml configs/experiments/single_layer_performance_yolov8n_orin.yaml configs/experiments/single_layer_performance_yolov8m_orin.yaml tests/unit/test_single_layer_performance_screening.py tests/unit/test_evaluation_apps.py
git commit -m "feat: expose single-layer performance screening"
```

---

### Task 8: Remove Legacy Screening Workflows and Update Research Consumers

**Files:**
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `apps/evaluate_cluster_pruning.py`
- Modify: `tests/unit/test_cluster_workflow.py`
- Modify: `tests/unit/test_evaluation_apps.py`
- Delete: `configs/experiments/filterwise_rtx_screening.yaml`
- Delete: `tools/build_kaggle_filterwise_notebook.py`
- Delete: `tests/unit/test_build_kaggle_filterwise_notebook.py`
- Modify: `tools/generate_supervisor_pruning_report.py`
- Create: `tests/unit/test_supervisor_pruning_report.py`
- Modify: `README.md`

**Interfaces:**
- Removes: `run_filterwise_evaluation`
- Removes CLI flags: `--filterwise`, `--full-curve`
- Preserves: `run_cluster_evaluation`, `run_filterwise_probe`, compression-matrix workflows, and historical run directories.

- [ ] **Step 1: Write failing legacy-removal assertions**

Update app tests:

```python
result = subprocess.run(
    [sys.executable, "apps/evaluate_cluster_pruning.py", "--help"],
    capture_output=True, text=True, check=False,
)
assert "--filterwise" not in result.stdout
assert "--full-curve" not in result.stdout
```

Add repository assertions that the old config/generator files do not exist and README contains `single_layer_performance_screening.py` but not `kaggle_filterwise_sensitivity.ipynb`.

- [ ] **Step 2: Run and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_evaluation_apps.py tests\unit\test_cluster_workflow.py -q
```

Expected: legacy-removal assertions fail while old flags and code remain.

- [ ] **Step 3: Remove filterwise code from cluster workflow**

Delete only sensitivity-specific constants, adapter fields, planners, workflow, resume helpers, step helpers, and tests:

```text
_RESUMABLE_FILTERWISE_STATUSES
_TERMINAL_FILTERWISE_STATUSES
_FILTERWISE_CHANNEL_MISMATCH_RE
make_filterwise_step
save_checkpoint
_filterwise_candidate_id
_planned_filterwise_rows
_filterwise_layer_widths
run_filterwise_evaluation
_filterwise_probe
_row_is_recorded
_should_profile_filterwise
_filterwise_stop_reached
_load_filterwise_checkpoint
_row_is_terminal_exclusion
_is_filterwise_channel_mismatch
_filterwise_step
_save_filterwise_checkpoint
```

Remove the special filterwise branch from `_failure`. Keep structural probe/global pruning, `_safe_layers`, `compute_channel_importance`, export, fine-tuning, and hardware benchmarking unchanged.

- [ ] **Step 4: Remove legacy files and CLI modes**

Delete the old YAML, Kaggle generator, and its tests. Remove `--filterwise`, `--full-curve`, and the `run_filterwise_evaluation` import/branch from `apps/evaluate_cluster_pruning.py`.

- [ ] **Step 5: Write the report regression test, then update README and supervisor report**

Add a fixture CSV with a baseline and completed `single_layer` rows, invoke the report generator through its public entry point, and assert the generated plot/table data uses `filters_remaining`, `map50_95`, and `latency_p50_ms`. Assert failed and planned rows are excluded and multiple `--single-layer-results` inputs remain distinguishable by model/hardware label.

Run the focused test before implementation and confirm it fails because the new argument and schema are unsupported:

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_supervisor_pruning_report.py -q
```

Then replace the Kaggle section with four commands for the new configs. State that results are `results.csv`, latency is direct PyTorch/CUDA forward latency, and no deployment artifacts are created.

Change `generate_supervisor_pruning_report.py` to accept repeatable:

```python
parser.add_argument("--single-layer-results", action="append", type=Path, default=[])
```

When no paths are supplied, use the two RTX defaults for YOLOv8n and YOLOv8m. Read `stage == "single_layer"`, `status == "completed"`, plot `filters_remaining` on the x-axis, and label latency as `PyTorch CUDA forward latency p50 (ms)`. Preserve the separate compression-matrix plots.

- [ ] **Step 6: Run focused legacy and report tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_cluster_workflow.py tests\unit\test_evaluation_apps.py tests\unit\test_artifacts.py tests\unit\test_supervisor_pruning_report.py -q
rg -n "kaggle_filterwise_sensitivity|filterwise_rtx_screening|--filterwise|--full-curve|run_filterwise_evaluation" README.md apps configs src tests tools
```

Expected: tests PASS and `rg` returns no current executable/documentation references. Historical files under `docs/superpowers` are excluded from the search intentionally.

- [ ] **Step 7: Commit Task 8**

```powershell
git add src/infrared_detection/evaluation/cluster_workflow.py apps/evaluate_cluster_pruning.py tests/unit/test_cluster_workflow.py tests/unit/test_evaluation_apps.py tests/unit/test_supervisor_pruning_report.py tools/generate_supervisor_pruning_report.py README.md
git add -u configs/experiments/filterwise_rtx_screening.yaml tools/build_kaggle_filterwise_notebook.py tests/unit/test_build_kaggle_filterwise_notebook.py
git commit -m "refactor: replace legacy filterwise screening"
```

---

### Task 9: Verify the Complete Replacement and Run Hardware Smoke Tests

**Files:**
- Modify only if verification reveals a defect in files already listed above.

**Interfaces:**
- Verifies all interfaces from Tasks 1–8.

- [ ] **Step 1: Run the complete automated suite**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests PASS. Existing pin-memory deprecation warnings may remain; no new warnings from the screening implementation are accepted.

- [ ] **Step 2: Run repository consistency checks**

```powershell
git diff --check
rg -n "kaggle_filterwise_sensitivity|filterwise_rtx_screening|--filterwise|--full-curve|run_filterwise_evaluation" README.md apps configs src tests tools
```

Expected: no whitespace errors and no live legacy references.

- [ ] **Step 3: Run a bounded RTX production smoke test**

Create a temporary test configuration outside Git by copying the YOLOv8n RTX config and replacing the layer pattern with `model.6.m.0.cv1.conv`, latency warm-up with `2`, and iterations with `5`. Run:

```powershell
.venv\Scripts\python.exe apps\single_layer_performance_screening.py --config runs\experiments\smoke\single_layer_yolov8n_rtx_smoke.yaml
```

Stop after the first two completed candidates. Restart the same command and verify ranks 1 and 2 are not repeated, `results.csv` remains unique by candidate ID, and only one `resume.pt` exists.

- [ ] **Step 4: Run the equivalent Jetson smoke test**

On the Jetson Orin Nano, copy the YOLOv8n Orin config to an untracked smoke path, select `model.6.m.0.cv1.conv`, set warm-up 2 and iterations 5, and repeat the interruption/resume check. Record the resolved CUDA device and software versions from `manifest.json`.

- [ ] **Step 5: Inspect the first result CSV manually**

Confirm the first columns are the 21 research fields in the specification, `reason,error` are last, baseline is first, filter ranks are ascending, `filters_remaining` decreases by one, and all completed rows contain `map50_95` plus mean/P50/P95 latency.

- [ ] **Step 6: Commit any verification-only corrections**

If no corrections were needed, do not create an empty commit. If corrections were required:

```powershell
git add -p
git commit -m "fix: complete screening verification"
```
