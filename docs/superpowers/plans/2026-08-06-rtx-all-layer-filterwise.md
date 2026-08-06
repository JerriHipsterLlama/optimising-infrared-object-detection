# RTX All-Layer Filterwise Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run sequential one-filter sensitivity sweeps for every dependency-safe YOLOv8 convolution layer on the RTX 3070, stopping an individual layer after mAP50-95 falls below 0.30.

**Architecture:** Keep the existing explicit `filter_sweep_layers`/`filter_sweep_widths` mode intact. Add an `auto` layer-selection mode that loads the dense baseline, obtains safe layers through the existing adapter, derives each selected `Conv2d.out_channels` from the same model, and expands the candidate plan before the per-layer loop. Early stopping remains per layer and uses the existing status/checkpoint artifact format.

**Tech Stack:** Python 3.13, PyYAML, PyTorch, Ultralytics 8.4.7, existing `ClusterEvaluationAdapters`, pytest.

## Global Constraints

- Use `mAP50-95` as the early-stop metric with `early_stop_map50_95: 0.30`.
- Do not invoke the runner with `full_curve=True`; that mode suppresses early stopping by design.
- Process automatic layers in the order returned by `active.safe_layers`.
- Exclude protected/detection-head layers through the existing `safe_layers` adapter behavior.
- Preserve explicit layer and width configuration behavior for existing experiments.
- Preserve the current per-candidate checkpoint, CSV, manifest, and resume schemas.

---

### Task 1: Add automatic filterwise candidate planning

**Files:**
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py:278-304, 550-685`
- Test: `tests/unit/test_cluster_workflow.py`

**Interfaces:**
- Consumes: `ClusterEvaluationAdapters.safe_layers(model, config) -> list[str]` and a dense baseline wrapper.
- Produces: `_planned_filterwise_rows(config, layer_widths: Mapping[str, int] | None = None) -> list[Metrics]` with one baseline row plus candidates from either explicit config widths or runtime-derived widths.

- [ ] **Step 1: Write failing tests for auto planning and explicit compatibility**

```python
def test_filterwise_auto_plan_uses_safe_layer_order_and_runtime_widths(monkeypatch, tmp_path, adapters):
    config = yaml.safe_load(write_config(tmp_path).read_text())
    config["pruning"]["filter_sweep_layers"] = "auto"
    config["pruning"].pop("filter_sweep_widths", None)
    config_path = tmp_path / "auto.yaml"
    config_path.write_text(yaml.safe_dump(config))
    adapters.safe_layers = lambda model, config: ["model.2", "model.1"]
    monkeypatch.setattr(cluster_workflow, "_filterwise_layer_widths", lambda model, layers: {"model.2": 3, "model.1": 2})

    rows = run_filterwise_evaluation(config_path, adapters=adapters)

    assert [(row["layer"], row["filters_removed"]) for row in rows[1:]] == [
        ("model.2", 1), ("model.2", 2), ("model.1", 1)
    ]


def test_filterwise_explicit_plan_still_requires_configured_widths(tmp_path):
    config = yaml.safe_load(write_config(tmp_path).read_text())
    config["pruning"]["filter_sweep_layers"] = ["model.1"]
    config["pruning"].pop("filter_sweep_widths", None)

    with pytest.raises(ValueError, match="filter_sweep_widths"):
        _planned_filterwise_rows(config)


def test_filterwise_layer_widths_reads_conv_output_channels():
    model = _importance_model()

    widths = _filterwise_layer_widths(model, ["model.2.cv1.conv"])

    assert widths == {"model.2.cv1.conv": 4}
```

- [ ] **Step 2: Run the focused tests and confirm the auto-planning test fails**

Run: `python -m pytest tests/unit/test_cluster_workflow.py -k "filterwise_auto_plan or filterwise_explicit_plan" -q`

Expected: the explicit compatibility test passes; the auto-planning test fails because the runner treats the string `auto` as an iterable layer name and cannot derive widths.

- [ ] **Step 3: Implement runtime width derivation and auto plan expansion**

```python
def _filterwise_layer_widths(model: Any, layers: Sequence[str]) -> dict[str, int]:
    from torch import nn

    modules = dict(_unwrap_model(model).named_modules())
    widths: dict[str, int] = {}
    for layer in layers:
        module = modules.get(layer)
        if not isinstance(module, nn.Conv2d) or int(module.out_channels) <= 1:
            raise ValueError(f"Automatic filter-wise layer {layer!r} is not a prunable Conv2d module.")
        widths[layer] = int(module.out_channels)
    return widths
```

Make `_planned_filterwise_rows` accept supplied `layer_widths`; when `filter_sweep_layers == "auto"`, use the supplied mapping's insertion order. In `run_filterwise_evaluation`, evaluate the baseline first, obtain `available_layers = active.safe_layers(baseline_model, config)`, derive widths, expand the full candidate list, and then merge resume artifacts against that expanded plan. Keep explicit configuration on the existing preplanned path.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run: `python -m pytest tests/unit/test_cluster_workflow.py -k "filterwise_auto_plan or filterwise_explicit_plan" -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/infrared_detection/evaluation/cluster_workflow.py tests/unit/test_cluster_workflow.py
git commit -m "feat: plan filterwise candidates from auto layers"
```

### Task 2: Configure all-layer RTX early stopping

**Files:**
- Modify: `configs/experiments/filterwise_rtx_screening.yaml`
- Modify: `tests/unit/test_cluster_workflow.py:190-205`

**Interfaces:**
- Consumes: automatic planning added in Task 1.
- Produces: the standard RTX screening configuration with `filter_sweep_layers: auto` and mAP50-95 threshold `0.30`.

- [ ] **Step 1: Write the failing configuration test**

```python
def test_filterwise_rtx_config_selects_all_layers_and_stops_below_point_three():
    config = yaml.safe_load(Path("configs/experiments/filterwise_rtx_screening.yaml").read_text())

    assert config["pruning"]["filter_sweep_layers"] == "auto"
    assert "filter_sweep_widths" not in config["pruning"]
    assert config["screening"]["early_stop"] is True
    assert config["screening"]["early_stop_map50_95"] == 0.30
    assert config["screening"]["early_stop_consecutive"] == 1
```

- [ ] **Step 2: Run the configuration test and confirm it fails**

Run: `python -m pytest tests/unit/test_cluster_workflow.py::test_filterwise_rtx_config_selects_all_layers_and_stops_below_point_three -q`

Expected: FAIL because the configuration still names five layers, includes widths, and uses `0.001`.

- [ ] **Step 3: Update the RTX configuration**

```yaml
pruning:
  protected_layers: [model.22]
  filter_sweep_layers: auto

screening:
  max_map50_95_drop: 0.02
  early_stop: true
  early_stop_map50_95: 0.30
  early_stop_consecutive: 1
```

Retain the existing latency profile removals and all runtime, target, and export settings.

- [ ] **Step 4: Run the configuration test and early-stop tests**

Run: `python -m pytest tests/unit/test_cluster_workflow.py -k "filterwise_rtx_config or filterwise_workflow_early_stops" -q`

Expected: PASS. The existing early-stop test proves that rows after the threshold point are marked skipped for the current layer only.

- [ ] **Step 5: Commit Task 2**

```bash
git add configs/experiments/filterwise_rtx_screening.yaml tests/unit/test_cluster_workflow.py
git commit -m "config: sweep all RTX filterwise layers"
```

### Task 3: Verify automatic-plan resume and hand off the local command

**Files:**
- Modify: `src/infrared_detection/evaluation/cluster_workflow.py:550-685`
- Modify: `tests/unit/test_cluster_workflow.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a verified local command that uses automatic planning and early stopping.

- [ ] **Step 1: Write a failing resume test for an automatic plan**

```python
def test_filterwise_auto_plan_resumes_completed_candidates(monkeypatch, tmp_path, adapters):
    config = yaml.safe_load(write_config(tmp_path).read_text())
    config["pruning"]["filter_sweep_layers"] = "auto"
    config["pruning"].pop("filter_sweep_widths", None)
    config["export"] = {"format": "onnx"}
    config_path = tmp_path / "auto-resume.yaml"
    config_path.write_text(yaml.safe_dump(config))
    adapters.safe_layers = lambda model, config: ["model.1"]
    monkeypatch.setattr(cluster_workflow, "_filterwise_layer_widths", lambda model, layers: {"model.1": 3})
    calls = {"evaluate": 0}
    adapters.evaluate = lambda model, config, device: calls.__setitem__("evaluate", calls["evaluate"] + 1) or _metrics(0.50)

    run_filterwise_evaluation(config_path, adapters=adapters)
    first_run_evaluations = calls["evaluate"]
    run_filterwise_evaluation(config_path, adapters=adapters)

    assert calls["evaluate"] == first_run_evaluations
```

- [ ] **Step 2: Run the resume test and confirm it fails if automatic rows are not merged**

Run: `python -m pytest tests/unit/test_cluster_workflow.py::test_filterwise_auto_plan_resumes_completed_candidates -q`

Expected: FAIL because automatic candidate IDs are initially planned only after safe-layer discovery, so the pre-expansion manifest cannot be merged.

- [ ] **Step 3: Make the minimal resume correction**

Ensure automatic expansion calls `_load_filterwise_checkpoint(output_dir, resolved_config_path, rows)` after it has built deterministic layer/width candidate IDs, and rebind `baseline = rows[0]` before checking its resumable state.

- [ ] **Step 4: Run focused and full verification**

Run: `python -m pytest tests/unit/test_cluster_workflow.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: PASS with only the repository's existing PyTorch pin-memory deprecation warnings.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/infrared_detection/evaluation/cluster_workflow.py tests/unit/test_cluster_workflow.py
git commit -m "test: cover automatic filterwise resume"
```

## Local execution

After the implementation is verified, run without the full-curve flag:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
.venv\Scripts\python.exe apps\evaluate.py filterwise --config configs\experiments\filterwise_rtx_screening.yaml
```

This command evaluates every discovered safe layer, preserves one-filter increments, and stops the current layer after its measured mAP50-95 is below 0.30.
