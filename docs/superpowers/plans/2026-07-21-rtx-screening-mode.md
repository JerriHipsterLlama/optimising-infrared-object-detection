# RTX Screening Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit RTX-only screening mode for cluster-pruning experiments.

**Architecture:** Extend `run_cluster_evaluation` with `screen_only=False`. In screen-only mode, baseline evaluation uses `rtx_screening_device`, probes are evaluated/profiled on RTX, global rows are marked skipped, and Orin winner selection remains disabled. Add a CLI flag and a separate screening config/output directory.

**Tech Stack:** Python, PyTorch, PyYAML, pytest, Ultralytics/TensorRT.

## Global Constraints

- RTX screening results must never select or classify a final Orin winner.
- The existing non-screen-only workflow remains Jetson Orin-gated.
- Screen-only mode must not fine-tune global candidates.
- Screen-only artifacts use `runs/experiments/cluster_pruning_rtx_screening`.

---

### Task 1: Add and verify RTX-only screening mode

**Files:**

- Modify: `src/infrared_detection/evaluation/cluster_workflow.py`
- Modify: `apps/evaluate_cluster_pruning.py`
- Create: `configs/experiments/cluster_pruning_rtx_screening.yaml`
- Test: `tests/unit/test_cluster_workflow.py`
- Test: `tests/integration/test_cluster_evaluation_app.py`

**Interfaces:**

- `run_cluster_evaluation(config_path: Path, dry_run: bool = False, adapters: ClusterEvaluationAdapters | None = None, screen_only: bool = False) -> list[dict]`
- CLI flag: `--screen-only`

- [ ] **Step 1: Write failing tests.**

```python
def test_screen_only_uses_rtx_and_skips_global_candidates(tmp_path, adapters):
    config_path = write_screening_config(tmp_path)
    evaluated_devices = []
    adapters.evaluate = lambda model, config, device: evaluated_devices.append(device) or _metrics(0.50)

    rows = run_cluster_evaluation(config_path, adapters=adapters, screen_only=True)

    assert evaluated_devices[0] == "0"
    assert all(row["status"] == "skipped" for row in rows if row["stage"] == "global")
    assert all(row.get("hardware_benchmarked") is not True for row in rows)
```

- [ ] **Step 2: Run RED.**

Run: `python -m pytest tests/unit/test_cluster_workflow.py -q`

Expected: collection or call failure because `screen_only` is not accepted.

- [ ] **Step 3: Implement the mode.**

Add `screen_only=False` to the workflow. Use `rtx_device` for baseline evaluation when true, bypass the Orin-host guard, and after completing probes mark every planned global row as `skipped` with reason `screen-only mode does not run global candidates`. Do not call `fine_tune`, `make_global`, or Orin profiling in this branch. Add the CLI argument and pass it through.

- [ ] **Step 4: Add the separate config.**

Copy the existing experiment configuration with:

```yaml
experiment:
  output_dir: runs/experiments/cluster_pruning_rtx_screening
targets:
  rtx_screening_device: 0
  orin_target: jetson_orin_nano
  orin_execution_device: '0'
```

- [ ] **Step 5: Run GREEN and the CLI smoke test.**

Run: `python -m pytest tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py -q`

Run: `python apps/evaluate_cluster_pruning.py --config configs/experiments/cluster_pruning_rtx_screening.yaml --screen-only --dry-run`

Expected: tests pass and dry-run emits planned baseline/probe/global rows without loading a model or accessing CUDA.

- [ ] **Step 6: Commit.**

```powershell
git add src/infrared_detection/evaluation/cluster_workflow.py apps/evaluate_cluster_pruning.py configs/experiments/cluster_pruning_rtx_screening.yaml tests/unit/test_cluster_workflow.py tests/integration/test_cluster_evaluation_app.py
git commit -m "feat: add RTX cluster pruning screening mode"
```
