from __future__ import annotations

import pytest
import torch
from torch import nn

from infrared_detection.compression.pruning import (
    validate_and_prune_unit as public_validate_and_prune_unit,
)
from infrared_detection.compression.pruning.importance import (
    l1_filter_scores,
    rank_output_channels,
)
from infrared_detection.compression.pruning.unit_discovery import requested_prune_count
from infrared_detection.compression.pruning.unit_discovery import (
    DependencyOperation,
    capture_detect_contract,
    discover_pruning_units,
    serialize_dependency_group,
    validate_and_prune_unit,
    validate_detect_contract,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)


class Upsample(nn.Module):
    pass


class Concat(nn.Module):
    pass


class Detect(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = ConvBlock(8, 8)
        self.dfl = ConvBlock(16, 1)
        self.nc = 4
        self.reg_max = 16
        self.no = 68
        self.nl = 3


class TinyYoloGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            [ConvBlock(3, 8), ConvBlock(8, 8), Upsample(), ConvBlock(8, 8), Detect()]
        )


def four_class_detect_fixture():
    model = TinyYoloGraph()
    output = (
        torch.zeros(1, 8, 84),
        {
            "boxes": torch.zeros(1, 64, 84),
            "scores": torch.zeros(1, 4, 84),
            "feats": [
                torch.zeros(1, 64, 8, 8),
                torch.zeros(1, 128, 4, 4),
                torch.zeros(1, 256, 2, 2),
            ],
        },
    )
    return model, output


class SequentialNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root = nn.Conv2d(3, 8, 1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.consumer = nn.Conv2d(8, 4, 1, bias=False)

    def forward(self, image):
        return self.consumer(self.bn(self.root(image)))


class ResidualNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch_a = nn.Conv2d(3, 8, 1, bias=False)
        self.branch_b = nn.Conv2d(8, 8, 1, bias=False)

    def forward(self, image):
        hidden = self.branch_a(image)
        return hidden + self.branch_b(hidden)


def test_l1_ranking_sums_absolute_filter_weights_and_breaks_ties_by_index():
    conv = nn.Conv2d(1, 4, 1, bias=False)
    with torch.no_grad():
        conv.weight[:, 0, 0, 0] = torch.tensor([-3.0, 1.0, -1.0, 2.0])

    assert l1_filter_scores(conv).tolist() == [3.0, 1.0, 1.0, 2.0]
    assert [index for index, _score in rank_output_channels(conv)] == [1, 2, 3, 0]


@pytest.mark.parametrize(
    ("channels", "ratio", "expected"),
    [
        (16, 0.125, 2),
        (16, 0.25, 4),
        (16, 0.375, 6),
        (16, 0.50, 8),
        (80, 0.125, 10),
    ],
)
def test_requested_prune_count_uses_dense_output_width(channels, ratio, expected):
    assert requested_prune_count(channels, ratio) == expected


@pytest.mark.parametrize(
    ("channels", "ratio", "message"),
    [(1, 0.5, "at least two"), (8, 0.0, "between zero and one"), (8, 1.0, "between zero and one")],
)
def test_requested_prune_count_rejects_invalid_requests(channels, ratio, message):
    with pytest.raises(ValueError, match=message):
        requested_prune_count(channels, ratio)


def test_ranking_rejects_unknown_importance_criterion():
    with pytest.raises(ValueError, match="Supported importance criteria"):
        rank_output_channels(nn.Conv2d(1, 2, 1), criterion="unknown")


def test_discovery_tags_convolutions_from_structural_boundaries():
    units = {unit.name: unit for unit in discover_pruning_units(TinyYoloGraph())}

    assert units["model.0.conv"].region == "backbone"
    assert units["model.3.conv"].region == "neck"
    assert units["model.4.hidden.conv"].region == "detect_head"


def test_single_output_dfl_convolution_is_discovered_for_audit():
    unit = next(
        unit
        for unit in discover_pruning_units(TinyYoloGraph())
        if unit.name == "model.4.dfl.conv"
    )

    assert unit.original_channels == 1
    assert unit.region == "detect_head"


def test_detect_contract_captures_scale_tensor_and_class_semantics():
    model, output = four_class_detect_fixture()

    contract = capture_detect_contract(model, output)

    assert contract.number_of_scales == 3
    assert contract.prediction_rank == 3
    assert contract.prediction_channels == 8
    assert contract.boxes_rank == 3
    assert contract.box_channels == 64
    assert contract.scores_rank == 3
    assert contract.class_channels == 4
    assert (contract.nc, contract.reg_max, contract.no, contract.nl) == (4, 16, 68, 3)


def test_detect_contract_rejects_changed_class_output_width():
    model, output = four_class_detect_fixture()
    expected = capture_detect_contract(model, output)
    changed = (
        output[0],
        {**output[1], "scores": torch.zeros(1, 3, 84)},
    )

    with pytest.raises(ValueError, match="Detect contract changed"):
        validate_detect_contract(expected, model, changed)


def test_sequential_group_allows_root_bn_and_downstream_input_changes():
    model = SequentialNet().eval()

    probe = validate_and_prune_unit(
        model,
        torch.randn(1, 3, 8, 8),
        "root",
        0.5,
        baseline_contract=None,
    )

    assert probe.status == "VALID"
    assert probe.original_channels == 8
    assert probe.requested_pruned_channels == 4
    assert probe.requested_remaining_channels == 4
    assert probe.actual_remaining_channels == 4
    assert probe.actual_pruning_ratio == 0.5
    assert set(probe.touched_modules) >= {"root", "bn", "consumer"}
    assert {operation.operation for operation in probe.operations} >= {
        "prune_out_channels",
        "prune_in_channels",
    }
    assert probe.model is model
    assert model.consumer.in_channels == 4


def test_structural_unit_probe_is_available_from_pruning_package():
    model = SequentialNet().eval()

    probe = public_validate_and_prune_unit(
        model, torch.randn(1, 3, 8, 8), "root", 0.25, baseline_contract=None
    )

    assert probe.status == "VALID"


def test_group_that_prunes_another_convolution_output_is_grouped_and_not_mutated():
    model = ResidualNet().eval()
    before = {
        name: module.out_channels
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }

    probe = validate_and_prune_unit(
        model,
        torch.randn(1, 3, 8, 8),
        "branch_b",
        0.25,
        baseline_contract=None,
    )

    assert probe.status == "GROUPED"
    assert "branch_a" in probe.touched_modules
    assert probe.model is None
    assert {
        name: module.out_channels
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    } == before


def test_dependency_operations_serialize_deterministically():
    operations = (
        DependencyOperation(
            module_name="root",
            module_type="Conv2d",
            operation="prune_out_channels",
            indices=(0, 2),
        ),
    )

    assert serialize_dependency_group(operations) == (
        '[{"affected_count":2,"indices":[0,2],"module_name":"root",'
        '"module_type":"Conv2d","operation":"prune_out_channels"}]'
    )


def test_single_channel_unit_is_invalid_without_building_a_pruning_group():
    model = nn.Module()
    model.root = nn.Conv2d(3, 1, 1)

    probe = validate_and_prune_unit(
        model,
        torch.randn(1, 3, 8, 8),
        "root",
        0.5,
        baseline_contract=None,
    )

    assert probe.status == "INVALID"
    assert probe.reason == "insufficient_channels"
