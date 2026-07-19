from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from apps.compress import compress, main


def test_prune_dispatch_uses_supplied_model_graph():
    model = nn.Linear(4, 2)
    graph = MagicMock(model=model)
    channel_masks = {"layer": torch.tensor([True, False])}

    with patch("apps.compress.prune_yolo_channels", return_value=model) as prune:
        result = compress(model, {"method": "prune", "graph": graph, "channel_masks": channel_masks})

    assert result is model
    prune.assert_called_once_with(graph, channel_masks)


def test_prune_dispatch_builds_graph_for_supplied_model_when_graph_is_omitted():
    model = nn.Linear(4, 2)
    example_input = torch.randn(1, 4)
    graph = MagicMock(model=model)
    channel_masks = {"layer": torch.tensor([True, False])}

    with (
        patch("infrared_detection.compression.pruning.build_yolo_dependency_graph", return_value=graph) as build_graph,
        patch("apps.compress.prune_yolo_channels", return_value=model) as prune,
    ):
        result = compress(
            model,
            {"method": "prune", "example_input": example_input, "channel_masks": channel_masks},
        )

    assert result is model
    build_graph.assert_called_once_with(model, example_input)
    prune.assert_called_once_with(graph, channel_masks)


def test_prune_dispatch_rejects_graph_bound_to_another_model():
    model = nn.Linear(4, 2)
    unrelated_model = nn.Linear(4, 2)
    graph = MagicMock(model=unrelated_model)

    with patch("apps.compress.prune_yolo_channels") as prune:
        with pytest.raises(ValueError, match="does not belong to the supplied model"):
            compress(
                model,
                {"method": "prune", "graph": graph, "channel_masks": {"layer": torch.tensor([True, False])}},
            )

    prune.assert_not_called()


def test_quantize_dispatch_returns_model_and_records_payload():
    model = nn.Linear(4, 2)
    payload = {"bit_width": 8, "quantized_state_dict": {"weight": torch.ones(2, 4, dtype=torch.int8)}}

    with patch("apps.compress.quantize_checkpoint", return_value=payload) as quantize:
        result = compress(model, {"method": "quantize", "bit_width": 8})

    assert result is model
    assert model._compression_quantization is payload
    quantize.assert_called_once_with(model, 8)


def test_distill_dispatch_returns_trained_student():
    teacher = nn.Linear(4, 2)
    student = nn.Linear(4, 2)
    dataloader = [torch.randn(1, 4)]
    trained_student = nn.Linear(4, 2)
    config = {"method": "distill", "teacher": teacher, "dataloader": dataloader, "epochs": 3}

    with patch("apps.compress.train_student_with_kd", return_value=trained_student) as distill:
        result = compress(student, config)

    assert result is trained_student
    distill.assert_called_once_with(teacher, student, dataloader, config)


def test_matrix_dispatch_runs_matrix_and_returns_original_model(tmp_path):
    model = nn.Linear(4, 2)
    config_path = tmp_path / "matrix.yaml"

    with patch("apps.compress.run_matrix") as run_matrix:
        result = compress(model, {"method": "matrix", "config_path": config_path, "dry_run": True})

    assert result is model
    run_matrix.assert_called_once_with(config_path, dry_run=True)


def test_compress_rejects_unknown_method():
    with pytest.raises(ValueError, match="prune, quantize, distill, matrix"):
        compress(nn.Linear(4, 2), {"method": "unknown"})


@pytest.mark.parametrize(
    ("arguments", "method"),
    [
        (["matrix", "--config", "matrix.yaml", "--dry-run"], "matrix"),
        (["quantize", "--model", "student.pt", "--bit-width", "4"], "quantize"),
        (
            [
                "prune",
                "--model",
                "student.pt",
                "--example-input",
                "input.pt",
                "--channel-masks",
                "masks.json",
            ],
            "prune",
        ),
        (
            ["distill", "--model", "student.pt", "--teacher", "teacher.pt", "--dataloader", "data.pt"],
            "distill",
        ),
    ],
)
def test_cli_dispatches_every_advertised_method(monkeypatch, tmp_path, arguments, method):
    model = nn.Linear(4, 2)
    masks_path = tmp_path / "masks.json"
    masks_path.write_text('{"layer": [true, false]}', encoding="utf-8")
    arguments = [str(masks_path) if value == "masks.json" else value for value in arguments]

    def load_module(path):
        return nn.Linear(4, 2) if str(path) == "teacher.pt" else model

    monkeypatch.setattr("sys.argv", ["compress.py", *arguments])
    with (
        patch("apps.compress._load_serialized_module", side_effect=load_module, create=True),
        patch("apps.compress._load_serialized_object", return_value=torch.randn(1, 4), create=True),
        patch("apps.compress.compress", return_value=model) as dispatch,
    ):
        main()

    assert dispatch.call_args.args[1]["method"] == method
