from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml
import types

import pytest

import infrared_detection.evaluation.faster_rcnn_single_layer_screening as backend
import infrared_detection.evaluation.single_layer_performance_screening as screening


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.seen_images = None

    def forward(self, images):
        self.seen_images = images
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=image.device),
                "labels": torch.tensor([1], device=image.device),
                "scores": torch.tensor([0.9], device=image.device),
            }
            for image in images
        ]


def _write_config(tmp_path: Path, *, precision="fp32"):
    checkpoint_path = tmp_path / "best.pt"
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump({"model": {"num_classes": 2}, "data": {"num_classes": 2}}),
        encoding="utf-8",
    )
    config = {
        "model": {"backend": "faster_rcnn", "checkpoint": str(checkpoint_path), "config": str(model_config)},
        "data": {"dataset_root": str(tmp_path), "split": "val"},
        "experiment": {"image_size": 8, "batch_size": 1},
        "runtime": {"device": "cpu", "precision": precision, "conf": 0.25, "iou": 0.5},
        "screening": {"warmup": 0, "iterations": 1},
    }
    return config, checkpoint_path


def test_load_model_applies_ema_shadow_parameters_and_checkpoint_round_trips(tmp_path, monkeypatch):
    config, checkpoint_path = _write_config(tmp_path)
    monkeypatch.setattr(backend, "build_faster_rcnn_model", lambda _count, _yaml: TinyDetector())
    initial = TinyDetector()
    state = {name: torch.full_like(value, 1) for name, value in initial.state_dict().items()}
    torch.save(
        {
            "model_state_dict": state,
            "ema_state_dict": {"shadow": {"conv.weight": torch.full_like(state["conv.weight"], 3)}},
        },
        checkpoint_path,
    )
    adapters = backend.make_adapters(config)

    loaded = adapters.load_model(checkpoint_path)
    save_path = tmp_path / "resume.pending.pt"
    adapters.save_checkpoint(loaded, save_path)
    reloaded = adapters.load_model(save_path)

    assert loaded.model.conv.weight.detach().eq(3).all()
    assert reloaded.model.conv.weight.detach().eq(3).all()


def test_pending_checkpoint_round_trips_pruned_model_topology(tmp_path, monkeypatch):
    config, checkpoint_path = _write_config(tmp_path)
    monkeypatch.setattr(backend, "build_faster_rcnn_model", lambda _count, _yaml: TinyDetector())
    torch.save({"model_state_dict": TinyDetector().state_dict()}, checkpoint_path)
    adapters = backend.make_adapters(config)
    pruned = adapters.load_model(checkpoint_path)
    pruned.model.conv = nn.Conv2d(3, 1, 1)
    pending_path = tmp_path / "resume.pending.pt"

    adapters.save_checkpoint(pruned, pending_path)
    restored = adapters.load_model(pending_path)

    assert restored.model.conv.out_channels == 1


def test_evaluate_uses_cameldataset_and_returns_shared_detection_metrics(tmp_path, monkeypatch):
    config, checkpoint_path = _write_config(tmp_path)
    monkeypatch.setattr(backend, "build_faster_rcnn_model", lambda _count, _yaml: TinyDetector())
    torch.save({"model_state_dict": TinyDetector().state_dict()}, checkpoint_path)
    images_seen = []
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "class_ids": torch.tensor([1])}]

    class TinyDataset:
        def __init__(self, root, *, split, augment):
            assert root == config["data"]["dataset_root"]
            assert split == "val" and augment is False

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return torch.zeros(3, 8, 8), targets[0]

    def compute_metrics(predictions, observed_targets, **kwargs):
        images_seen.append((predictions, observed_targets, kwargs))
        return {"mAP50_95": 0.8, "mAP50": 0.9, "precision": 0.7, "recall": 0.6}

    monkeypatch.setattr(backend, "CAMELDataset", TinyDataset)
    monkeypatch.setattr(backend, "compute_detection_metrics", compute_metrics)
    adapters = backend.make_adapters(config)
    model = adapters.load_model(checkpoint_path)

    metrics = adapters.evaluate(model, config)

    assert metrics == {"map50_95": 0.8, "map50": 0.9, "precision": 0.7, "recall": 0.6}
    assert len(images_seen) == 1
    assert model.model.seen_images and model.model.seen_images[0].shape == (3, 8, 8)
    assert images_seen[0][2] == {"iou_threshold": 0.5, "conf_threshold": 0.25, "num_classes": 2}


def _loaded_adapters(tmp_path, monkeypatch, *, precision="fp32"):
    config, checkpoint_path = _write_config(tmp_path, precision=precision)
    monkeypatch.setattr(backend, "build_faster_rcnn_model", lambda _count, _yaml: TinyDetector())
    torch.save({"model_state_dict": TinyDetector().state_dict()}, checkpoint_path)
    adapters = backend.make_adapters(config)
    return config, adapters, adapters.load_model(checkpoint_path)


def test_prune_filter_removes_only_requested_output_index_with_detector_example(monkeypatch, tmp_path):
    config, adapters, wrapper = _loaded_adapters(tmp_path, monkeypatch)
    target = wrapper.model.conv
    trace_shapes = []

    def prune_conv_out_channels(module, indices):
        keep = [index for index in range(module.out_channels) if index not in indices]
        module.weight = nn.Parameter(module.weight.detach()[keep].clone())
        module.bias = nn.Parameter(module.bias.detach()[keep].clone())
        module.out_channels = len(keep)

    class Group:
        def __init__(self, operations):
            self.operations = operations
            self.pruned = False

        def __iter__(self):
            return iter(self.operations)

        def prune(self):
            self.pruned = True
            prune_conv_out_channels(target, [1])

    def operation(module, indices):
        return types.SimpleNamespace(
            dep=types.SimpleNamespace(
                target=types.SimpleNamespace(module=module), handler=prune_conv_out_channels
            ),
            idxs=indices,
        )

    group = Group([operation(target, [1])])

    class Dependency:
        def get_pruning_group(self, module, function, *, idxs):
            assert module is target and function is prune_conv_out_channels and idxs == [1]
            return group

        def check_pruning_group(self, _group):
            return True

    monkeypatch.setitem(__import__("sys").modules, "torch_pruning", types.SimpleNamespace(prune_conv_out_channels=prune_conv_out_channels))

    def graph_builder(model, example):
        trace_shapes.append(tuple(example.shape))
        return types.SimpleNamespace(graph=Dependency())

    monkeypatch.setattr(backend, "build_faster_rcnn_dependency_graph", graph_builder)

    adapters.prune_filter(wrapper, "conv", 1, config)

    assert target.out_channels == 1
    assert group.pruned is True
    assert trace_shapes == [(3, 8, 8)]


def test_coupled_output_pruning_is_unscorable_without_mutating_model(monkeypatch, tmp_path):
    config, adapters, wrapper = _loaded_adapters(tmp_path, monkeypatch)
    target = wrapper.model.conv
    coupled = nn.Conv2d(2, 2, 1)

    def prune_conv_out_channels(_module, _indices):
        pass

    class Group:
        def __iter__(self):
            return iter([
                types.SimpleNamespace(dep=types.SimpleNamespace(target=types.SimpleNamespace(module=target), handler=prune_conv_out_channels), idxs=[0]),
                types.SimpleNamespace(dep=types.SimpleNamespace(target=types.SimpleNamespace(module=coupled), handler=prune_conv_out_channels), idxs=[0]),
            ])

        def prune(self):
            raise AssertionError("coupled groups must be rejected before mutation")

    class Dependency:
        def get_pruning_group(self, *_args, **_kwargs):
            return Group()

        def check_pruning_group(self, _group):
            return True

    monkeypatch.setitem(__import__("sys").modules, "torch_pruning", types.SimpleNamespace(prune_conv_out_channels=prune_conv_out_channels))
    monkeypatch.setattr(backend, "build_faster_rcnn_dependency_graph", lambda *_args: types.SimpleNamespace(graph=Dependency()))

    with pytest.raises(screening.UnscorableLayerError, match="another module's output channels"):
        adapters.prune_filter(wrapper, "conv", 0, config)

    assert target.out_channels == 2


def test_profile_wraps_tensor_batch_as_detector_image_list(monkeypatch, tmp_path):
    config, adapters, wrapper = _loaded_adapters(tmp_path, monkeypatch)
    benchmark_calls = []

    def benchmark(model, example, **kwargs):
        benchmark_calls.append((model, example, kwargs))
        model(example)
        return {"latency_mean_ms": 2.0, "latency_p50_ms": 1.8, "latency_p95_ms": 2.5, "fps": 500.0}

    monkeypatch.setattr(screening, "benchmark_pytorch_cuda_forward", benchmark, raising=False)

    result = adapters.profile(wrapper, config)

    assert result["latency_p50_ms"] == 1.8
    model, example, kwargs = benchmark_calls[0]
    assert tuple(example.shape) == (1, 3, 8, 8)
    assert kwargs == {"warmup": 0, "iterations": 1}
    assert wrapper.model.seen_images and wrapper.model.seen_images[0].shape == (3, 8, 8)


def test_profile_honors_fp16_model_and_input_dtype(monkeypatch, tmp_path):
    config, adapters, wrapper = _loaded_adapters(tmp_path, monkeypatch, precision="fp16")
    observed = []

    def benchmark(model, example, **kwargs):
        observed.append((next(model.parameters()).dtype, example.dtype, kwargs))
        return {"latency_mean_ms": 1.0, "latency_p50_ms": 1.0, "latency_p95_ms": 1.0, "fps": 1000.0}

    monkeypatch.setattr(screening, "benchmark_pytorch_cuda_forward", benchmark, raising=False)

    adapters.profile(wrapper, config)

    assert observed == [(torch.float16, torch.float16, {"warmup": 0, "iterations": 1})]


def test_existing_screening_command_dispatches_faster_rcnn_config(tmp_path, monkeypatch):
    import sys

    from apps import single_layer_performance_screening as app

    config_path = tmp_path / "screen.yaml"
    config_path.write_text("model:\n  backend: faster_rcnn\n", encoding="utf-8")
    expected = object()
    dispatched = []
    invoked = []
    module_name = "infrared_detection.evaluation.faster_rcnn_single_layer_screening"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(make_adapters=lambda config: dispatched.append(config) or expected),
    )
    monkeypatch.setattr(app, "run_single_layer_performance_screening", lambda path, **kwargs: invoked.append((path, kwargs)))
    monkeypatch.setattr("sys.argv", ["single_layer_performance_screening.py", "--config", str(config_path)])

    app.main()

    assert dispatched == [{"model": {"backend": "faster_rcnn"}}]
    assert invoked[0][0] == str(config_path)
    assert invoked[0][1]["adapters"] is expected
