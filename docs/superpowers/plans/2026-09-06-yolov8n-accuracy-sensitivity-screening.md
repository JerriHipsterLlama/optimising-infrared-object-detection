# YOLOv8n Accuracy Sensitivity Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, resumable YOLOv8n workstation tool that measures immediate validation-accuracy sensitivity for independently valid structural pruning units at four ratios.

**Architecture:** A pruning-discovery module classifies model regions, ranks L1 filters, audits Torch-Pruning groups, rejects coupled roots, applies accepted structural groups, and verifies the Detect contract. A separate evaluation module owns configuration, fresh-checkpoint orchestration, metrics, normalized AUC, atomic CSV/manifest persistence, and resume behavior; a thin CLI supplies the production Ultralytics adapters.

**Tech Stack:** Python 3.10+, PyTorch 2.9.1, Ultralytics 8.4.7, Torch-Pruning 1.6.0-compatible APIs, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-yolov8n-accuracy-sensitivity-screening-design.md`

## Global Constraints

- Default pruning ratios are exactly `0.125`, `0.25`, `0.375`, and `0.50`.
- Each `(unit, ratio)` must load the original checkpoint anew; pruning must never accumulate.
- Every ratio receives its own dependency graph, dependency-group audit, structural prune, and Detect-contract validation.
- A group that prunes output channels from another meaningful `Conv2d` or `Linear` is `GROUPED` and must not be evaluated.
- Accepted dependency propagation may affect only the root output, normalization channels, downstream input channels, and mechanical graph operations.
- Baseline and candidates use the identical resolved validation argument mapping.
- No fine-tuning, retraining, calibration, or candidate-model persistence is permitted.
- Valid hidden Detect convolutions remain candidates; Detect prediction outputs and DFL semantics remain fixed.
- Every candidate is tagged `backbone`, `neck`, or `detect_head`.
- Complete-curve AUC uses achieved ratios, includes `(0, 0)`, uses trapezoidal integration, is normalized by maximum achieved ratio, and preserves negative degradation.
- Preserve unrelated worktree modifications in `.gitignore`, `tests/unit/test_pytorch_cuda_benchmark.py`, and `tests/unit/test_single_layer_performance_screening.py`.

---

### Task 1: L1 Ranking and Ratio Planning

**Files:**
- Modify: `src/infrared_detection/compression/pruning/importance.py`
- Create: `src/infrared_detection/compression/pruning/unit_discovery.py`
- Create: `tests/unit/test_pruning_unit_discovery.py`

**Interfaces:**
- Consumes: `torch.nn.Conv2d` weights.
- Produces: `l1_filter_scores(module: nn.Conv2d) -> torch.Tensor`, `rank_output_channels(module: nn.Conv2d, criterion: str = "l1") -> tuple[tuple[int, float], ...]`, and `requested_prune_count(channels: int, ratio: float) -> int`.

- [ ] **Step 1: Write failing L1 and ratio tests**

```python
def test_l1_ranking_sums_absolute_filter_weights_and_breaks_ties_by_index():
    conv = nn.Conv2d(1, 4, 1, bias=False)
    with torch.no_grad():
        conv.weight[:, 0, 0, 0] = torch.tensor([-3.0, 1.0, -1.0, 2.0])
    assert l1_filter_scores(conv).tolist() == [3.0, 1.0, 1.0, 2.0]
    assert [index for index, _ in rank_output_channels(conv)] == [1, 2, 3, 0]


@pytest.mark.parametrize(("channels", "ratio", "expected"), [
    (16, 0.125, 2), (16, 0.25, 4), (16, 0.375, 6), (16, 0.50, 8),
    (80, 0.125, 10),
])
def test_requested_prune_count_uses_dense_output_width(channels, ratio, expected):
    assert requested_prune_count(channels, ratio) == expected
```

- [ ] **Step 2: Run the new tests and confirm missing imports fail**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py -q`

Expected: collection failure because the new APIs do not exist.

- [ ] **Step 3: Implement the minimal ranking and count APIs**

```python
def l1_filter_scores(module: nn.Conv2d) -> torch.Tensor:
    if not isinstance(module, nn.Conv2d):
        raise TypeError("L1 filter scoring requires torch.nn.Conv2d.")
    return module.weight.detach().to(torch.float32).flatten(1).abs().sum(dim=1)


def rank_output_channels(module: nn.Conv2d, criterion: str = "l1") -> tuple[tuple[int, float], ...]:
    if criterion != "l1":
        raise ValueError("Supported importance criteria: l1")
    values = l1_filter_scores(module).cpu().tolist()
    return tuple(sorted(enumerate(values), key=lambda item: (item[1], item[0])))


def requested_prune_count(channels: int, ratio: float) -> int:
    if channels < 2:
        raise ValueError("A pruning unit must have at least two output channels.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("Pruning ratio must be between zero and one.")
    count = round(channels * ratio)
    if not 1 <= count < channels:
        raise ValueError("Requested ratio does not remove a valid channel count.")
    return count
```

- [ ] **Step 4: Run the focused tests and existing importance tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py tests/unit/test_structured_pruning.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the ranking slice**

```powershell
git add src/infrared_detection/compression/pruning/importance.py src/infrared_detection/compression/pruning/unit_discovery.py tests/unit/test_pruning_unit_discovery.py
git commit -m "feat: add L1 pruning unit planning"
```

### Task 2: Region Discovery and Detect Contract

**Files:**
- Modify: `src/infrared_detection/compression/pruning/unit_discovery.py`
- Modify: `tests/unit/test_pruning_unit_discovery.py`

**Interfaces:**
- Consumes: an Ultralytics-like model with top-level `model`, `Upsample`/`Concat`, and `Detect` modules.
- Produces: `PruningUnit(name, region, original_channels)`, `DetectContract`, `discover_pruning_units(model)`, `capture_detect_contract(model, output)`, and `validate_detect_contract(expected, model, output)`.

- [ ] **Step 1: Write failing structural-region tests**

```python
def test_discovery_tags_convolutions_from_structural_boundaries():
    model = TinyYoloGraph()
    units = {unit.name: unit for unit in discover_pruning_units(model)}
    assert units["model.0.conv"].region == "backbone"
    assert units["model.3.conv"].region == "neck"
    assert units["model.4.hidden.conv"].region == "detect_head"


def test_single_output_dfl_convolution_is_discovered_for_audit_but_not_pruneable():
    unit = next(unit for unit in discover_pruning_units(TinyYoloGraph()) if unit.name.endswith("dfl.conv"))
    assert unit.original_channels == 1
```

- [ ] **Step 2: Write failing Detect-contract tests**

```python
def test_detect_contract_checks_scale_tensor_and_class_semantics():
    model, output = four_class_detect_fixture()
    contract = capture_detect_contract(model, output)
    assert contract.number_of_scales == 3
    assert contract.prediction_rank == 3
    assert contract.prediction_channels == 8
    assert contract.box_channels == 64
    assert contract.class_channels == 4
    assert (contract.nc, contract.reg_max, contract.no, contract.nl) == (4, 16, 68, 3)


def test_detect_contract_rejects_changed_class_output_width():
    model, output = four_class_detect_fixture()
    expected = capture_detect_contract(model, output)
    changed = changed_score_width(output, channels=3)
    with pytest.raises(ValueError, match="Detect contract changed"):
        validate_detect_contract(expected, model, changed)
```

- [ ] **Step 3: Run tests and confirm the missing discovery APIs fail**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py -q`

Expected: failures name the missing dataclasses and functions.

- [ ] **Step 4: Implement structural region classification and complete Detect-contract equality**

```python
@dataclass(frozen=True)
class PruningUnit:
    name: str
    region: Literal["backbone", "neck", "detect_head"]
    original_channels: int


@dataclass(frozen=True)
class DetectContract:
    number_of_scales: int
    prediction_rank: int
    prediction_channels: int
    boxes_rank: int
    box_channels: int
    scores_rank: int
    class_channels: int
    nc: int
    reg_max: int
    no: int
    nl: int
```

Implement `discover_pruning_units` by locating the top-level Detect parent, locating the first top-level upsample/concatenation before it, and walking `named_modules()` in order. Implement contract extraction from the Ultralytics `(prediction, {"boxes", "scores", "feats"})` output, with a compatibility branch for list-based raw outputs that derives equivalent scale and channel facts.

- [ ] **Step 5: Run tests and refactor contract parsing without changing behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py -q`

Expected: all region and contract tests pass.

- [ ] **Step 6: Commit the discovery slice**

```powershell
git add src/infrared_detection/compression/pruning/unit_discovery.py tests/unit/test_pruning_unit_discovery.py
git commit -m "feat: classify YOLO pruning regions and Detect contract"
```

### Task 3: Dependency-Group Audit and Structural Probe

**Files:**
- Modify: `src/infrared_detection/compression/pruning/unit_discovery.py`
- Modify: `src/infrared_detection/compression/pruning/__init__.py`
- Modify: `tests/unit/test_pruning_unit_discovery.py`

**Interfaces:**
- Consumes: `build_yolo_dependency_graph`, a root name, ranked indices, ratio, and baseline `DetectContract`.
- Produces: `DependencyOperation`, `StructuralProbe`, `inspect_dependency_group(...)`, and `validate_and_prune_unit(...) -> StructuralProbe` where `StructuralProbe.model` is populated only for `VALID`.

- [ ] **Step 1: Write a failing real sequential-group audit test**

```python
def test_sequential_group_allows_root_bn_and_downstream_input_changes():
    model = SequentialNet()
    probe = validate_and_prune_unit(
        model, torch.randn(1, 3, 8, 8), "root", 0.5,
        baseline_contract=None, forward=lambda candidate, image: candidate(image),
    )
    assert probe.status == "VALID"
    assert probe.original_channels == 8
    assert probe.requested_pruned_channels == 4
    assert probe.actual_remaining_channels == 4
    assert probe.actual_pruning_ratio == 0.5
    assert {operation.module_name for operation in probe.operations} >= {"root", "bn", "consumer"}
    assert probe.model.consumer.in_channels == 4
```

- [ ] **Step 2: Write a failing residual coupling test**

```python
def test_group_that_prunes_another_convolution_output_is_grouped_and_not_mutated():
    model = ResidualNet()
    original = {name: module.out_channels for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    probe = validate_and_prune_unit(model, torch.randn(1, 3, 8, 8), "branch_b", 0.25, baseline_contract=None)
    assert probe.status == "GROUPED"
    assert "branch_a" in probe.touched_modules
    assert probe.model is None
    assert {name: module.out_channels for name, module in model.named_modules() if isinstance(module, nn.Conv2d)} == original
```

- [ ] **Step 3: Write failing per-ratio and serialization tests**

```python
def test_each_ratio_builds_and_inspects_a_fresh_dependency_group():
    calls = []
    for ratio in (0.125, 0.25, 0.375, 0.5):
        validate_and_prune_unit(make_sequential_net(), example(), "root", ratio, None, graph_builder=recording_builder(calls))
    assert calls == [0.125, 0.25, 0.375, 0.5]


def test_dependency_operations_serialize_deterministically():
    operations = (DependencyOperation("root", "Conv2d", "prune_out_channels", (0, 2)),)
    assert serialize_dependency_group(operations) == (
        '[{"affected_count":2,"indices":[0,2],"module_name":"root",'
        '"module_type":"Conv2d","operation":"prune_out_channels"}]'
    )
```

- [ ] **Step 4: Run tests and confirm dependency behavior is absent**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py -q`

Expected: failures reference missing probe/audit APIs.

- [ ] **Step 5: Implement dependency operation extraction and `GROUPED` classification**

```python
@dataclass(frozen=True)
class DependencyOperation:
    module_name: str
    module_type: str
    operation: str
    indices: tuple[int, ...]

    @property
    def affected_count(self) -> int:
        return len(self.indices)


@dataclass
class StructuralProbe:
    status: Literal["VALID", "GROUPED", "INVALID", "SEMANTICS_CHANGED"]
    unit_name: str
    original_channels: int
    requested_pruned_channels: int
    requested_remaining_channels: int
    actual_remaining_channels: int | None
    requested_pruning_ratio: float
    actual_pruning_ratio: float | None
    operations: tuple[DependencyOperation, ...]
    touched_modules: tuple[str, ...]
    reason: str | None
    error: str | None
    model: nn.Module | None
```

Build a reverse module-name map including Conv, Linear, BatchNorm, and named wrapper modules. Normalize Torch-Pruning handler names to `prune_out_channels`, `prune_in_channels`, or a stable mechanical-operation name. Before calling `group.prune()`, mark `GROUPED` if an operation targets output channels of a named `Conv2d`/`Linear` other than the selected root. Store unnamed operations using deterministic graph-order identifiers rather than object addresses.

- [ ] **Step 6: Implement structural pruning, physical count checks, and Detect validation**

Apply only accepted groups. Synchronize channel metadata through `synchronize_module_channel_metadata`, read actual root width from `weight.shape[0]`, run the forward callback under inference mode, and compare the complete Detect contract. Convert group rejection, invalid counts, forward exceptions, and contract mismatches to terminal probe statuses without hiding their reasons.

- [ ] **Step 7: Run dependency tests plus existing structural tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py tests/unit/test_structured_pruning.py tests/unit/test_cluster_probe.py -q`

Expected: all tests pass and existing pruning behavior remains green.

- [ ] **Step 8: Export the public discovery APIs and commit**

```powershell
git add src/infrared_detection/compression/pruning/unit_discovery.py src/infrared_detection/compression/pruning/__init__.py tests/unit/test_pruning_unit_discovery.py
git commit -m "feat: audit independent Torch-Pruning groups"
```

### Task 4: Sensitivity Metrics and AUC Ranking

**Files:**
- Create: `src/infrared_detection/evaluation/accuracy_sensitivity.py`
- Create: `tests/unit/test_accuracy_sensitivity.py`

**Interfaces:**
- Consumes: baseline metrics and successful detailed rows.
- Produces: `ValidationMetrics`, `calculate_point_metrics`, `normalized_degradation_auc`, and `build_unit_rankings`.

- [ ] **Step 1: Write failing point-sensitivity tests**

```python
@pytest.mark.parametrize(("baseline", "candidate", "ratio", "degradation", "sensitivity"), [
    (0.50, 0.45, 0.25, 0.05, 0.20),
    (0.50, 0.52, 0.25, -0.02, -0.08),
])
def test_point_metrics_preserve_signed_accuracy_change(baseline, candidate, ratio, degradation, sensitivity):
    result = calculate_point_metrics(baseline, candidate, ratio)
    assert result.accuracy_degradation == pytest.approx(degradation)
    assert result.map50_95_change_from_baseline == pytest.approx(-degradation)
    assert result.point_sensitivity == pytest.approx(sensitivity)
```

- [ ] **Step 2: Write failing normalized AUC tests**

```python
def test_normalized_auc_includes_zero_origin_and_uses_achieved_ratios():
    points = [(0.125, 0.01), (0.25, 0.03), (0.375, 0.06), (0.5, 0.10)]
    assert normalized_degradation_auc(points) == pytest.approx(0.0375)


def test_rankings_sort_complete_curves_descending_and_leave_incomplete_unranked():
    rows = completed_curve("sensitive", [0.01, 0.03, 0.06, 0.10])
    rows += completed_curve("robust", [0.00, 0.01, 0.02, 0.03])
    rows += completed_curve("missing", [0.01, 0.02, 0.03], ratios=(0.125, 0.25, 0.375))
    ranking = build_unit_rankings(rows, DEFAULT_RATIOS)
    assert [(row["pruning_unit"], row["rank"]) for row in ranking[:2]] == [("sensitive", 1), ("robust", 2)]
    assert next(row for row in ranking if row["pruning_unit"] == "missing")["curve_status"] == "INCOMPLETE"
```

- [ ] **Step 3: Run tests and confirm metric APIs are missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py -q`

Expected: import failure for the new evaluation module.

- [ ] **Step 4: Implement signed point metrics and hand-written trapezoidal AUC**

```python
def normalized_degradation_auc(points: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((float(ratio), float(drop)) for ratio, drop in points)
    if not ordered or ordered[0][0] <= 0 or any(b[0] <= a[0] for a, b in pairwise(ordered)):
        raise ValueError("Achieved ratios must be positive and strictly increasing.")
    curve = [(0.0, 0.0), *ordered]
    area = sum((right_r - left_r) * (left_y + right_y) / 2 for (left_r, left_y), (right_r, right_y) in pairwise(curve))
    return area / ordered[-1][0]
```

Implement `build_unit_rankings` with one row per meaningful unit, `COMPLETE` only when all configured requested ratios have successful rows, descending AUC order, module-name tie breaking, and blank rank/AUC for incomplete curves.

- [ ] **Step 5: Run metric tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py -q`

Expected: all point and AUC tests pass.

- [ ] **Step 6: Commit the metric slice**

```powershell
git add src/infrared_detection/evaluation/accuracy_sensitivity.py tests/unit/test_accuracy_sensitivity.py
git commit -m "feat: calculate pruning sensitivity curves"
```

### Task 5: Configuration, Atomic Artifacts, Fingerprinting, and Resume

**Files:**
- Modify: `src/infrared_detection/evaluation/accuracy_sensitivity.py`
- Modify: `tests/unit/test_accuracy_sensitivity.py`

**Interfaces:**
- Consumes: config path, detailed rows, rankings, manifest data.
- Produces: `load_sensitivity_config`, `experiment_fingerprint`, `write_artifacts`, and `load_resume_artifacts`.

- [ ] **Step 1: Write failing configuration-validation tests**

```python
def test_config_accepts_changeable_ratios_and_rejects_invalid_values(tmp_path):
    path = write_config(tmp_path, ratios=[0.125, 0.25, 0.375, 0.5])
    assert load_sensitivity_config(path).ratios == DEFAULT_RATIOS
    assert load_sensitivity_config(write_config(tmp_path, ratios=[0.2, 0.4], name="custom")).ratios == (0.2, 0.4)
    for invalid in ([0.25, 0.25], [0.0], [1.0], []):
        with pytest.raises(ValueError):
            load_sensitivity_config(write_config(tmp_path, ratios=invalid, name=str(len(invalid))))
```

- [ ] **Step 2: Write failing artifact and resume tests**

```python
def test_artifacts_write_required_csv_columns_and_manifest_atomically(tmp_path):
    write_artifacts(tmp_path, [baseline_row(), successful_row()], [ranking_row()], manifest_fixture())
    assert read_header(tmp_path / "results.csv")[:20] == DETAILED_FIELDS[:20]
    assert read_header(tmp_path / "unit_sensitivity_ranking.csv") == AGGREGATE_FIELDS
    assert json.loads((tmp_path / "manifest.json").read_text())["fingerprint"] == "abc"
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_reuses_terminal_rows_retries_error_and_rejects_mismatch(tmp_path):
    write_artifacts(tmp_path, [row("a", "COMPLETED"), row("b", "GROUPED"), row("c", "ERROR")], [], manifest_fixture())
    state = load_resume_artifacts(tmp_path, expected_fingerprint="abc")
    assert state.reusable_candidate_ids == {"a", "b"}
    assert "c" not in state.reusable_candidate_ids
    with pytest.raises(ValueError, match="fingerprint"):
        load_resume_artifacts(tmp_path, expected_fingerprint="different")
```

- [ ] **Step 3: Run tests and confirm persistence APIs are missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py -q`

Expected: failures reference missing config and artifact functions.

- [ ] **Step 4: Implement immutable configuration and exact validation mapping**

Use frozen dataclasses for resolved paths, ratios, importance, example size, output directory, and a read-only validation mapping. Validate a nonempty, unique ratio tuple with every ratio in `(0, 1)`, `importance == "l1"`, existing input files, and a positive example size. The shipped configuration uses the four required defaults while orchestration remains reusable with later ratio sets.

- [ ] **Step 5: Implement canonical fingerprint and atomic writers**

Canonicalize checkpoint/dataset path, size, and nanosecond mtime; validation mapping; ratios; importance version; and Torch/Ultralytics/Torch-Pruning versions as sorted compact JSON before SHA-256 hashing. Write sibling `.tmp` files and replace `results.csv`, `unit_sensitivity_ranking.csv`, and `manifest.json` only after complete serialization.

- [ ] **Step 6: Implement resume classification**

Treat `COMPLETED`, `GROUPED`, `INVALID`, and `SEMANTICS_CHANGED` as reusable terminal rows. Retry `ERROR`. Reuse `BASELINE` only when its manifest fingerprint matches. Reject any partially present artifact set or header mismatch with a descriptive error.

- [ ] **Step 7: Run persistence tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py -q`

Expected: all configuration, artifact, fingerprint, and resume tests pass.

```powershell
git add src/infrared_detection/evaluation/accuracy_sensitivity.py tests/unit/test_accuracy_sensitivity.py
git commit -m "feat: persist and resume sensitivity screening"
```

### Task 6: Fresh-Checkpoint Screening Orchestration

**Files:**
- Modify: `src/infrared_detection/evaluation/accuracy_sensitivity.py`
- Modify: `tests/unit/test_accuracy_sensitivity.py`

**Interfaces:**
- Consumes: `SensitivityAdapters`, config, pruning discovery/probe APIs.
- Produces: `run_accuracy_sensitivity(config_path, adapters=None, on_result=None) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write a failing independence test**

```python
def test_every_unit_ratio_loads_the_original_checkpoint_and_never_accumulates(tmp_path):
    adapters, loads, prune_histories = recording_adapters()
    run_accuracy_sensitivity(write_config(tmp_path), adapters=adapters)
    assert loads == ["baseline", *(["candidate"] * (len(UNITS) * 4))]
    assert prune_histories == [()] * (len(UNITS) * 4)
```

- [ ] **Step 2: Write failing validation parity and complete-row tests**

```python
def test_baseline_and_candidates_receive_identical_validation_arguments(tmp_path):
    adapters, evaluation_calls = validation_recording_adapters()
    rows = run_accuracy_sensitivity(write_config(tmp_path), adapters=adapters)
    assert len({canonical_json(call.kwargs) for call in evaluation_calls}) == 1
    completed = next(row for row in rows if row["status"] == "COMPLETED")
    assert completed["architectural_region"] == "detect_head"
    assert completed["requested_pruned_channels"] == 8
    assert completed["actual_remaining_channels"] == 56
    assert completed["touched_modules"]
    assert json.loads(completed["dependency_group_json"])
```

- [ ] **Step 3: Write failing grouped/invalid/error continuation tests**

```python
def test_nonvalid_structural_rows_are_not_evaluated_and_next_dense_candidate_continues(tmp_path):
    adapters = adapters_with_probe_statuses(["GROUPED", "INVALID", "SEMANTICS_CHANGED", "VALID"])
    rows = run_accuracy_sensitivity(write_config(tmp_path), adapters=adapters)
    assert adapters.candidate_evaluations == 1
    assert [row["status"] for row in candidate_rows(rows)] == ["GROUPED", "INVALID", "SEMANTICS_CHANGED", "COMPLETED"]


def test_runtime_error_row_is_retried_on_resume_from_a_fresh_checkpoint(tmp_path):
    first = adapters_failing_candidate_once()
    run_accuracy_sensitivity(write_config(tmp_path), adapters=first)
    resumed = successful_recording_adapters()
    run_accuracy_sensitivity(write_config(tmp_path), adapters=resumed)
    assert resumed.loaded_candidate_histories == [()]
```

- [ ] **Step 4: Run tests and confirm orchestration is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py -q`

Expected: failures reference `run_accuracy_sensitivity`.

- [ ] **Step 5: Implement adapters and deterministic orchestration**

```python
@dataclass(frozen=True)
class SensitivityAdapters:
    load_model: Callable[[Path], Any]
    unwrap_model: Callable[[Any], nn.Module]
    evaluate: Callable[[Any, Mapping[str, Any]], ValidationMetrics]
    make_example_input: Callable[[nn.Module, int], torch.Tensor]
    probe: Callable[..., StructuralProbe] = validate_and_prune_unit
```

Load or reuse the baseline, capture its Detect contract, enumerate all Conv units from that unmodified baseline model, and iterate in model order then ratio order. For each non-reused candidate, call `load_model(checkpoint)` directly, resolve the unit on that dense copy, recompute ranking, audit/apply the group, validate only `VALID` models, calculate point metrics, drop all candidate references, regenerate rankings, and atomically persist. Catch ordinary candidate exceptions as `ERROR`; allow `KeyboardInterrupt`/`SystemExit` to persist and propagate.

- [ ] **Step 6: Implement the production Ultralytics adapters**

Load `YOLO(checkpoint)` with the existing legacy pathlib compatibility helper. For evaluation call `wrapper.val(**resolved_validation_kwargs)` and return `box.map`, `box.map50`, `box.mp`, and `box.mr`. Replace `wrapper.model` only for the in-memory candidate. Never call `train`, `save`, `export`, or a calibration API. Set `plots=False`, `save_json=False`, and `verbose=False` identically for baseline and candidates.

- [ ] **Step 7: Run orchestration tests and the complete focused unit suite**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_accuracy_sensitivity.py tests/unit/test_pruning_unit_discovery.py -q`

Expected: all tests pass, including fresh-load and resumed-fresh-load assertions.

- [ ] **Step 8: Commit orchestration**

```powershell
git add src/infrared_detection/evaluation/accuracy_sensitivity.py tests/unit/test_accuracy_sensitivity.py
git commit -m "feat: run independent accuracy sensitivity experiments"
```

### Task 7: CLI, Default Configuration, and Documentation

**Files:**
- Create: `apps/yolov8_accuracy_sensitivity.py`
- Create: `configs/experiments/yolov8n_accuracy_sensitivity.yaml`
- Modify: `tests/unit/test_evaluation_apps.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `--config PATH`.
- Produces: exit status, concise progress output, documented workstation command.

- [ ] **Step 1: Write failing CLI and config tests**

```python
def test_accuracy_sensitivity_app_exposes_required_config_argument():
    result = subprocess.run([sys.executable, "apps/yolov8_accuracy_sensitivity.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_default_accuracy_sensitivity_config_has_required_ratios_and_paths():
    config = yaml.safe_load(Path("configs/experiments/yolov8n_accuracy_sensitivity.yaml").read_text())
    assert config["pruning"]["ratios"] == [0.125, 0.25, 0.375, 0.5]
    assert config["pruning"]["importance"] == "l1"
    assert config["model"]["checkpoint"].endswith("train3/weights/best.pt")
```

- [ ] **Step 2: Run CLI tests and confirm files are absent**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_apps.py -q`

Expected: failures indicate the app/config do not exist.

- [ ] **Step 3: Create the thin CLI**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Screen YOLOv8n structural pruning accuracy sensitivity.")
    parser.add_argument("--config", required=True, help="Sensitivity experiment YAML configuration.")
    args = parser.parse_args()
    run_accuracy_sensitivity(args.config, on_result=print_progress)
```

Progress output includes status, region, unit, requested/actual ratios, mAP50-95, and point sensitivity, with blank-safe formatting for non-evaluated rows.

- [ ] **Step 4: Create the workstation YAML**

Use the trained `train3/weights/best.pt`, `data/camel/camel.yaml`, split `val`, image size `352`, batch `1`, configured workstation device, FP32 by default, `conf=0.25`, `iou=0.6`, `workers=0`, example image size `352`, and output `runs/experiments/yolov8n_accuracy_sensitivity`.

- [ ] **Step 5: Document behavior and outputs**

Add a README section with the command, independent fresh-checkpoint semantics, 39-unit observed graph result, `GROUPED` meaning, the two CSVs, resume behavior, lack of fine-tuning, and lack of candidate model files.

- [ ] **Step 6: Run app tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_apps.py -q`

Expected: all application tests pass.

```powershell
git add apps/yolov8_accuracy_sensitivity.py configs/experiments/yolov8n_accuracy_sensitivity.yaml tests/unit/test_evaluation_apps.py README.md
git commit -m "feat: expose YOLOv8n accuracy sensitivity CLI"
```

### Task 8: Real YOLOv8n Graph Integration and Final Verification

**Files:**
- Create: `tests/integration/test_yolov8n_accuracy_sensitivity_graph.py`
- Modify: `src/infrared_detection/compression/pruning/unit_discovery.py` only when a failing real-graph test proves a compatibility defect
- Modify: `src/infrared_detection/evaluation/accuracy_sensitivity.py` only when a failing real-graph test proves an orchestration defect

**Interfaces:**
- Consumes: `models/checkpoints/yolov8/train3/weights/best.pt` and in-memory example tensors.
- Produces: evidence that actual YOLOv8n graph decisions and Detect semantics satisfy the specification.

- [ ] **Step 1: Write the real-checkpoint integration tests**

```python
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained YOLOv8n checkpoint unavailable")
@pytest.mark.parametrize("ratio", DEFAULT_RATIOS)
def test_actual_yolov8n_representative_units_are_independent_at_every_ratio(ratio):
    for name, region in (
        ("model.0.conv", "backbone"),
        ("model.12.cv2.conv", "neck"),
        ("model.22.cv2.0.0.conv", "detect_head"),
        ("model.22.cv3.2.1.conv", "detect_head"),
    ):
        model = load_model()
        contract, example = baseline_contract_and_example(model)
        probe = validate_and_prune_unit(model, example, name, ratio, contract)
        assert probe.status == "VALID"
        assert probe.actual_pruning_ratio == pytest.approx(ratio)


@pytest.mark.parametrize("ratio", DEFAULT_RATIOS)
def test_actual_c2f_residual_root_is_grouped_at_every_ratio(ratio):
    model = load_model()
    probe = validate_and_prune_unit(model, example(model), "model.6.m.0.cv2.conv", ratio, baseline_contract(model))
    assert probe.status == "GROUPED"
    assert "model.6.cv1.conv" in probe.touched_modules


def test_actual_model_discovers_expected_audit_inventory():
    audit = audit_all_combinations(load_model(), DEFAULT_RATIOS)
    assert count_units_complete_at_all_ratios(audit, "VALID") == 39
    assert count_units_complete_at_all_ratios(audit, "GROUPED") == 10
    assert count_units_with_any_grouped_ratio(audit) == 18
    assert all(row.region in {"backbone", "neck", "detect_head"} for row in audit)
```

The inventory assertion distinguishes ten roots that are `GROUPED` at all ratios from eight split roots that become dependency-rejected at higher ratios but are never valid independent units.

- [ ] **Step 2: Run the integration file and observe any real compatibility failures**

Run: `$env:YOLO_CONFIG_DIR="$PWD\tmp\ultralytics-config"; .venv\Scripts\python.exe -m pytest tests/integration/test_yolov8n_accuracy_sensitivity_graph.py -q`

Expected: tests pass against the trained four-class checkpoint. If a test fails, invoke `superpowers:systematic-debugging`, add the smallest reproducing unit test, and fix only the verified incompatibility.

- [ ] **Step 3: Run the full focused regression suite**

Run: `$env:YOLO_CONFIG_DIR="$PWD\tmp\ultralytics-config"; .venv\Scripts\python.exe -m pytest tests/unit/test_pruning_unit_discovery.py tests/unit/test_accuracy_sensitivity.py tests/integration/test_yolov8n_accuracy_sensitivity_graph.py tests/unit/test_structured_pruning.py tests/unit/test_cluster_probe.py tests/unit/test_evaluation_apps.py -q`

Expected: zero failures.

- [ ] **Step 4: Run static and CLI smoke checks**

Run: `.venv\Scripts\python.exe -m compileall -q src/infrared_detection apps/yolov8_accuracy_sensitivity.py`

Run: `.venv\Scripts\python.exe apps/yolov8_accuracy_sensitivity.py --help`

Expected: both commands exit `0`; help lists `--config`.

- [ ] **Step 5: Inspect the final diff and verify requirements line by line**

Run: `git diff --check`

Run: `git status --short`

Confirm the diff touches only planned files plus the committed design/plan, and confirm each global constraint has a passing test or an explicit runtime check.

- [ ] **Step 6: Commit integration coverage or final compatibility corrections**

```powershell
git add tests/integration/test_yolov8n_accuracy_sensitivity_graph.py src/infrared_detection/compression/pruning/unit_discovery.py src/infrared_detection/evaluation/accuracy_sensitivity.py
git commit -m "test: verify YOLOv8n pruning unit discovery"
```

- [ ] **Step 7: Request code review before integration**

Invoke `superpowers:requesting-code-review`, review the complete branch diff against the specification, address any verified findings with a failing test first, then repeat Steps 3–5 before claiming completion.
