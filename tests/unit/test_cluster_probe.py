import pytest
import torch
from torch import nn

from infrared_detection.compression.pruning import DependencyGraph
from infrared_detection.compression.pruning.cluster_selection import ClusterSpec
from infrared_detection.compression.pruning import cluster_probe
from infrared_detection.compression.pruning.cluster_probe import make_keep_mask, run_structural_probe


class _RejectingGraph:
    def get_pruning_group(self, module, pruning_fn, idxs):
        del module, pruning_fn, idxs
        return object()

    def check_pruning_group(self, group):
        del group
        return False


@pytest.fixture
def tiny_model():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, inputs):
            return self.conv(inputs)

    return TinyModel()


def rejecting_graph(model, example_input):
    del example_input
    return DependencyGraph(model=model, graph=_RejectingGraph(), modules={"conv": model.conv})


def test_make_keep_mask_removes_only_the_cluster_indices():
    assert torch.equal(
        make_keep_mask(4, (1, 3)),
        torch.tensor([True, False, True, False]),
    )


def test_probe_surfaces_dependency_rejection(monkeypatch, tiny_model):
    monkeypatch.setattr(cluster_probe, "build_yolo_dependency_graph", rejecting_graph)

    with pytest.raises(ValueError, match="rejected pruning group"):
        run_structural_probe(
            tiny_model, torch.randn(1, 3, 32, 32), ClusterSpec("conv", 2, (0, 1))
        )


def test_probe_rejects_known_detection_head_before_building_graph(monkeypatch, tiny_model):
    def unexpected_graph(*args, **kwargs):
        pytest.fail("detection-head probe must be rejected before graph construction")

    monkeypatch.setattr(cluster_probe, "build_yolo_dependency_graph", unexpected_graph)

    with pytest.raises(ValueError, match="detection head"):
        run_structural_probe(
            tiny_model, torch.randn(1, 3, 32, 32), ClusterSpec("model.22.cv3", 2, (0, 1))
        )


def test_probe_rejects_removing_every_output_channel(tiny_model):
    with pytest.raises(ValueError, match="removes every output channel"):
        run_structural_probe(
            tiny_model, torch.randn(1, 3, 32, 32), ClusterSpec("conv", 4, (0, 1, 2, 3))
        )
