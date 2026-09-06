from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

from infrared_detection.compression.pruning.unit_discovery import (
    capture_detect_contract,
    discover_pruning_units,
    validate_and_prune_unit,
)
from infrared_detection.evaluation.accuracy_sensitivity import DEFAULT_RATIOS


CHECKPOINT = Path(
    os.environ.get(
        "YOLOV8N_TEST_CHECKPOINT",
        Path(__file__).resolve().parents[2]
        / "models"
        / "checkpoints"
        / "yolov8"
        / "train3"
        / "weights"
        / "best.pt",
    )
)


def _load_model():
    from ultralytics import YOLO

    return YOLO(str(CHECKPOINT)).model.cuda().float().eval().requires_grad_(True)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="trained YOLOv8n checkpoint unavailable")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_actual_yolov8n_audits_all_ratios_and_preserves_detect_contract():
    baseline = _load_model()
    assert next(baseline.parameters()).is_cuda
    assert next(baseline.parameters()).dtype == torch.float32
    example = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        contract = capture_detect_contract(baseline, baseline(example))
    units = discover_pruning_units(baseline)

    statuses: dict[str, list[str]] = {unit.name: [] for unit in units}
    for unit in units:
        for ratio in DEFAULT_RATIOS:
            candidate = copy.deepcopy(baseline)
            probe = validate_and_prune_unit(
                candidate,
                example.clone(),
                unit.name,
                ratio,
                contract,
            )
            statuses[unit.name].append(probe.status)
            if probe.status == "VALID":
                assert probe.actual_pruning_ratio == pytest.approx(ratio)

    valid_at_all_ratios = {
        name for name, outcomes in statuses.items() if outcomes == ["VALID"] * 4
    }
    grouped_at_all_ratios = {
        name for name, outcomes in statuses.items() if outcomes == ["GROUPED"] * 4
    }
    grouped_at_any_ratio = {
        name for name, outcomes in statuses.items() if "GROUPED" in outcomes
    }
    valid_detect_hidden = {
        unit.name
        for unit in units
        if unit.region == "detect_head" and unit.name in valid_at_all_ratios
    }

    assert len(units) == 64
    assert len(valid_at_all_ratios) == 39
    assert len(grouped_at_all_ratios) == 10
    assert len(grouped_at_any_ratio) == 18
    assert len(valid_detect_hidden) == 12
    assert statuses["model.6.m.0.cv2.conv"] == ["GROUPED"] * 4
    assert statuses["model.22.cv2.0.2"] == ["INVALID"] * 4
    assert statuses["model.22.dfl.conv"] == ["INVALID"] * 4
    assert {unit.region for unit in units} == {"backbone", "neck", "detect_head"}
