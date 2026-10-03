from collections import OrderedDict

import torch

from infrared_detection.models.faster_rcnn.backbones import ResNet34FPNBackbone
from infrared_detection.models.faster_rcnn.training import build_faster_rcnn_model


def model_config(backbone: str, pretrained: bool = False) -> dict:
    return {
        "model": {
            "backbone": backbone,
            "pretrained": pretrained,
            "min_size": 64,
            "max_size": 64,
            "trainable_layers": 5,
        }
    }


def test_resnet34_fpn_replaces_detector_head_and_exposes_fpn_channels():
    model = build_faster_rcnn_model(4, model_config("resnet34"))

    assert isinstance(model.backbone, ResNet34FPNBackbone)
    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert model.backbone.out_channels == 256

    features = model.backbone(torch.zeros(1, 3, 64, 64))
    assert isinstance(features, OrderedDict)
    assert list(features) == ["0", "1", "2", "3"]
    assert all(feature.shape[1] == 256 for feature in features.values())


def test_resnet50_uses_native_faster_rcnn_fpn_with_custom_head():
    model = build_faster_rcnn_model(4, model_config("resnet50"))

    assert hasattr(model.backbone.body, "layer4")
    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert model.backbone.out_channels == 256


def test_resnet_variants_run_a_cpu_training_loss_pass_without_pretrained_weights():
    image = torch.zeros(3, 64, 64)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
        "labels": torch.tensor([1]),
    }

    for backbone in ("resnet34", "resnet50"):
        model = build_faster_rcnn_model(4, model_config(backbone))
        model.train()
        losses = model([image], [target])

        assert losses
        assert all(torch.isfinite(value) for value in losses.values())
