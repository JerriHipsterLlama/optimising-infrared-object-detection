from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from infrared_detection.evaluation.accuracy_sensitivity import (
    AGGREGATE_FIELDS,
    DEFAULT_RATIOS,
    DETAILED_FIELDS,
    build_unit_rankings,
    calculate_point_metrics,
    experiment_fingerprint,
    load_resume_artifacts,
    load_sensitivity_config,
    normalized_degradation_auc,
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
            "device": "cpu",
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
