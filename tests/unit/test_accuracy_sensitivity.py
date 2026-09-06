from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import yaml

from infrared_detection.compression.pruning.unit_discovery import (
    PruningUnit,
    StructuralProbe,
)
from infrared_detection.evaluation.accuracy_sensitivity import (
    AGGREGATE_FIELDS,
    DEFAULT_RATIOS,
    DETAILED_FIELDS,
    SensitivityAdapters,
    ValidationMetrics,
    build_unit_rankings,
    calculate_point_metrics,
    experiment_fingerprint,
    load_resume_artifacts,
    load_sensitivity_config,
    normalized_degradation_auc,
    run_accuracy_sensitivity,
    validate_yolov8n_model,
    write_artifacts,
)


@pytest.mark.parametrize(
    ("baseline", "candidate", "ratio", "degradation", "change", "sensitivity"),
    [
        (0.50, 0.45, 0.25, 0.05, -0.05, 0.20),
        (0.50, 0.52, 0.25, -0.02, 0.02, -0.08),
    ],
)
def test_point_metrics_preserve_signed_accuracy_change(
    baseline, candidate, ratio, degradation, change, sensitivity
):
    result = calculate_point_metrics(baseline, candidate, ratio)

    assert result.accuracy_degradation == pytest.approx(degradation)
    assert result.map50_95_change_from_baseline == pytest.approx(change)
    assert result.point_sensitivity == pytest.approx(sensitivity)


def test_point_metrics_reject_zero_achieved_ratio():
    with pytest.raises(ValueError, match="positive"):
        calculate_point_metrics(0.5, 0.4, 0.0)


def test_normalized_auc_includes_zero_origin_and_uses_achieved_ratios():
    points = [(0.125, 0.01), (0.25, 0.03), (0.375, 0.06), (0.5, 0.10)]

    assert normalized_degradation_auc(points) == pytest.approx(0.0375)


def test_normalized_auc_preserves_negative_accuracy_degradation():
    assert normalized_degradation_auc([(0.25, -0.02), (0.5, -0.04)]) == pytest.approx(-0.02)


def test_normalized_auc_rejects_duplicate_achieved_ratios():
    with pytest.raises(ValueError, match="strictly increasing"):
        normalized_degradation_auc([(0.25, 0.01), (0.25, 0.02)])


def _curve(unit: str, drops: list[float], ratios=DEFAULT_RATIOS) -> list[dict]:
    return [
        {
            "status": "COMPLETED",
            "pruning_unit": unit,
            "architectural_region": "neck",
            "original_channels": 64,
            "requested_pruning_ratio": requested,
            "actual_pruning_ratio": requested,
            "map50_95": 0.5 - drop,
            "accuracy_degradation": drop,
        }
        for requested, drop in zip(ratios, drops)
    ]


def test_rankings_sort_complete_curves_descending_and_leave_incomplete_unranked():
    rows = _curve("sensitive", [0.01, 0.03, 0.06, 0.10])
    rows += _curve("robust", [0.00, 0.01, 0.02, 0.03])
    rows += _curve("missing", [0.01, 0.02, 0.03], ratios=DEFAULT_RATIOS[:3])

    ranking = build_unit_rankings(rows, DEFAULT_RATIOS)

    ranked = [row for row in ranking if row["rank"] is not None]
    assert [(row["pruning_unit"], row["rank"]) for row in ranked] == [
        ("sensitive", 1),
        ("robust", 2),
    ]
    incomplete = next(row for row in ranking if row["pruning_unit"] == "missing")
    assert incomplete["curve_status"] == "INCOMPLETE"
    assert incomplete["rank"] is None
    assert incomplete["normalized_auc_sensitivity"] is None


def _write_config(tmp_path: Path, ratios=None, name="screening") -> Path:
    checkpoint = tmp_path / "model.pt"
    dataset = tmp_path / "dataset.yaml"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text("path: images\n", encoding="utf-8")
    path = tmp_path / f"{name}.yaml"
    payload = {
        "model": {"checkpoint": str(checkpoint)},
        "data": {"dataset_yaml": str(dataset), "split": "val"},
        "validation": {
            "imgsz": 352,
            "batch": 1,
            "device": "0",
            "half": False,
            "conf": 0.25,
            "iou": 0.6,
            "workers": 0,
        },
        "pruning": {
            "ratios": list(DEFAULT_RATIOS if ratios is None else ratios),
            "importance": "l1",
            "example_image_size": 352,
        },
        "experiment": {"output_dir": str(tmp_path / "artifacts")},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_config_accepts_changeable_ratios_and_rejects_invalid_values(tmp_path):
    config = load_sensitivity_config(_write_config(tmp_path))
    assert config.ratios == DEFAULT_RATIOS
    assert dict(config.validation)["half"] is False
    assert load_sensitivity_config(
        _write_config(tmp_path, ratios=[0.2, 0.4], name="custom")
    ).ratios == (0.2, 0.4)

    for index, invalid in enumerate(([0.25, 0.25], [0.0], [1.0], [])):
        with pytest.raises(ValueError, match="ratio"):
            load_sensitivity_config(_write_config(tmp_path, ratios=invalid, name=f"bad-{index}"))


def test_config_rejects_non_fp32_validation(tmp_path):
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["validation"]["half"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="FP32"):
        load_sensitivity_config(path)


def test_config_rejects_non_cuda_validation(tmp_path):
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["validation"]["device"] = "cpu"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="GPU"):
        load_sensitivity_config(path)


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _manifest(fingerprint="abc") -> dict:
    return {"fingerprint": fingerprint, "baseline": {"map50_95": 0.5}}


def test_artifacts_write_required_csv_columns_and_manifest_atomically(tmp_path):
    output = tmp_path / "artifacts"
    rows = [
        {"candidate_id": "baseline", "status": "BASELINE"},
        {
            "candidate_id": "unit-0.25",
            "status": "COMPLETED",
            "pruning_unit": "unit",
            "dependency_group_json": [{"module_name": "unit"}],
        },
    ]
    rankings = [{"rank": 1, "pruning_unit": "unit", "curve_status": "COMPLETE"}]

    write_artifacts(output, rows, rankings, _manifest())

    assert _csv_header(output / "results.csv") == list(DETAILED_FIELDS)
    assert _csv_header(output / "unit_sensitivity_ranking.csv") == list(AGGREGATE_FIELDS)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["fingerprint"] == "abc"
    assert not list(output.glob("*.tmp"))


def test_resume_reuses_terminal_rows_retries_error_and_rejects_mismatch(tmp_path):
    output = tmp_path / "artifacts"
    rows = [
        {"candidate_id": "a", "status": "COMPLETED"},
        {"candidate_id": "b", "status": "GROUPED"},
        {"candidate_id": "c", "status": "INVALID"},
        {"candidate_id": "d", "status": "SEMANTICS_CHANGED"},
        {"candidate_id": "e", "status": "ERROR"},
    ]
    write_artifacts(output, rows, [], _manifest())

    state = load_resume_artifacts(output, expected_fingerprint="abc")

    assert state.reusable_candidate_ids == {"a", "b", "c", "d"}
    assert "e" not in state.reusable_candidate_ids
    with pytest.raises(ValueError, match="fingerprint"):
        load_resume_artifacts(output, expected_fingerprint="different")


def test_experiment_fingerprint_changes_with_validation_or_input_metadata(tmp_path):
    config = load_sensitivity_config(_write_config(tmp_path))
    first = experiment_fingerprint(config, versions={"torch": "test"})
    same = experiment_fingerprint(config, versions={"torch": "test"})
    assert first == same
    config.dataset_yaml.write_text("path: changed\n", encoding="utf-8")
    assert first != experiment_fingerprint(config, versions={"torch": "test"})


class FakeModel(nn.Module):
    def __init__(self, unit_names=("unit",)) -> None:
        super().__init__()
        for unit_name in unit_names:
            self.add_module(unit_name, nn.Conv2d(1, 8, 1, bias=False))
        self.history: tuple[tuple[str, float], ...] = ()


class FakeWrapper:
    def __init__(self, model: FakeModel) -> None:
        self.model = model


def _probe(status_by_unit, histories):
    def probe(model, example_input, unit_name, ratio, baseline_contract, criterion="l1"):
        histories.append(model.history)
        status = status_by_unit.get(unit_name, "VALID")
        pruned_model = model if status == "VALID" else None
        if pruned_model is not None:
            pruned_model.history = ((unit_name, ratio),)
        return StructuralProbe(
            status=status,
            unit_name=unit_name,
            original_channels=8,
            requested_pruned_channels=round(8 * ratio),
            requested_remaining_channels=8 - round(8 * ratio),
            actual_remaining_channels=8 - round(8 * ratio) if status == "VALID" else None,
            requested_pruning_ratio=ratio,
            actual_pruning_ratio=ratio if status == "VALID" else None,
            operations=(),
            touched_modules=(unit_name,),
            reason=None if status == "VALID" else status.lower(),
            error=None,
            model=pruned_model,
        )
    return probe


def _recording_adapters(unit_names=("unit",), status_by_unit=None, fail_once=False):
    loads = []
    histories = []
    evaluations = []
    failure = {"pending": fail_once}

    def load_model(_checkpoint):
        wrapper = FakeWrapper(FakeModel(unit_names))
        loads.append(wrapper.model.history)
        return wrapper

    def evaluate(wrapper, validation):
        evaluations.append(dict(validation))
        if wrapper.model.history and failure["pending"]:
            failure["pending"] = False
            raise RuntimeError("temporary validation failure")
        drop = 0.01 if wrapper.model.history else 0.0
        return ValidationMetrics(0.5 - drop, 0.6, 0.7, 0.8)

    units = tuple(PruningUnit(name, "detect_head", 8) for name in unit_names)
    adapters = SensitivityAdapters(
        load_model=load_model,
        unwrap_model=lambda wrapper: wrapper.model,
        replace_model=lambda wrapper, model: setattr(wrapper, "model", model),
        evaluate=evaluate,
        make_example_input=lambda model, size: torch.zeros(1, 1, 4, 4),
        discover_units=lambda model: units,
        capture_contract=lambda model, output: None,
        forward=lambda model, example: None,
        probe=_probe(status_by_unit or {}, histories),
        versions=lambda: {"test": "1"},
    )
    return adapters, loads, histories, evaluations


def test_every_unit_ratio_loads_the_original_checkpoint_and_never_accumulates(tmp_path):
    adapters, loads, histories, _evaluations = _recording_adapters()

    run_accuracy_sensitivity(_write_config(tmp_path), adapters=adapters)

    assert loads == [()] * 5  # baseline plus four independent candidates
    assert histories == [()] * 4


def test_baseline_and_candidates_receive_identical_validation_arguments(tmp_path):
    adapters, _loads, _histories, evaluations = _recording_adapters()

    rows = run_accuracy_sensitivity(_write_config(tmp_path), adapters=adapters)

    assert len(evaluations) == 5
    assert all(arguments == evaluations[0] for arguments in evaluations)
    completed = next(row for row in rows if row["status"] == "COMPLETED")
    assert completed["architectural_region"] == "detect_head"
    assert completed["requested_pruned_channels"] == 1
    assert completed["actual_remaining_channels"] == 7
    assert completed["actual_pruning_ratio"] == 0.125
    assert completed["touched_modules"] == "unit"


def test_nonvalid_structural_rows_are_not_evaluated_and_next_candidate_continues(tmp_path):
    names = ("grouped", "invalid", "semantic", "valid")
    adapters, _loads, _histories, evaluations = _recording_adapters(
        names,
        {"grouped": "GROUPED", "invalid": "INVALID", "semantic": "SEMANTICS_CHANGED"},
    )

    rows = run_accuracy_sensitivity(
        _write_config(tmp_path, ratios=[0.5]), adapters=adapters
    )

    assert len(evaluations) == 2  # baseline and the sole valid candidate
    assert [row["status"] for row in rows if row["status"] != "BASELINE"] == [
        "GROUPED", "INVALID", "SEMANTICS_CHANGED", "COMPLETED"
    ]


def test_runtime_error_is_retried_on_resume_from_a_fresh_checkpoint(tmp_path):
    config = _write_config(tmp_path, ratios=[0.5])
    first, _loads, _histories, _evaluations = _recording_adapters(fail_once=True)
    first_rows = run_accuracy_sensitivity(config, adapters=first)
    assert next(row for row in first_rows if row["status"] != "BASELINE")["status"] == "ERROR"

    resumed, loads, histories, evaluations = _recording_adapters()
    resumed_rows = run_accuracy_sensitivity(config, adapters=resumed)

    assert loads == [(), ()]  # planning/baseline model plus retried candidate
    assert histories == [()]
    assert len(evaluations) == 1  # matching baseline metrics were resumed
    assert next(row for row in resumed_rows if row["status"] != "BASELINE")["status"] == "COMPLETED"


def test_default_adapters_return_ultralytics_box_metrics_and_forward_arguments(monkeypatch):
    calls = []

    class Wrapper:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
            self.model = nn.Conv2d(3, 8, 1).half()

        def val(self, **kwargs):
            calls.append(kwargs)
            box = types.SimpleNamespace(map=0.51, map50=0.72, mp=0.63, mr=0.64)
            return types.SimpleNamespace(box=box)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Wrapper, __version__="test"))
    adapters = SensitivityAdapters.defaults()
    wrapper = adapters.load_model(Path("dense.pt"))

    metrics = adapters.evaluate(wrapper, {"data": "dataset.yaml", "half": False})

    assert metrics == ValidationMetrics(0.51, 0.72, 0.63, 0.64)
    assert calls == [{"data": "dataset.yaml", "half": False}]
    assert wrapper.model.weight.dtype == torch.float32
    assert tuple(adapters.make_example_input(wrapper.model, 32).shape) == (1, 3, 32, 32)


def test_yolov8n_validation_rejects_other_model_scales():
    model = nn.Conv2d(3, 8, 1)
    model.yaml = {"scale": "m"}

    with pytest.raises(ValueError, match="YOLOv8n"):
        validate_yolov8n_model(model)


def test_default_forward_adapter_disables_gradients():
    grad_enabled = []

    class RecordingModel(nn.Conv2d):
        def forward(self, inputs):
            grad_enabled.append(torch.is_grad_enabled())
            return super().forward(inputs)

    model = RecordingModel(3, 8, 1)
    output = SensitivityAdapters.defaults().forward(model, torch.randn(1, 3, 8, 8))

    assert output.shape == (1, 8, 8, 8)
    assert grad_enabled == [False]
