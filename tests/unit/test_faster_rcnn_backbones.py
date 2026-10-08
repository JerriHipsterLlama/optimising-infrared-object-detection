from collections import OrderedDict

import torch

from infrared_detection.models.faster_rcnn.backbones import ResNet34FPNBackbone
from infrared_detection.models.faster_rcnn.training import (
    build_faster_rcnn_model,
    collate_fn,
    export_onnx,
)


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


def test_faster_rcnn_collate_adapts_grayscale_images_to_three_channels():
    images, _ = collate_fn(
        [
            (
                torch.zeros(1, 64, 64),
                {
                    "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
                    "class_ids": torch.tensor([1]),
                },
            )
        ]
    )

    assert images[0].shape == (3, 64, 64)


def test_faster_rcnn_onnx_export_uses_three_channel_legacy_exporter(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").write_bytes(b"checkpoint")

    captured = {}

    class Logger:
        def info(self, message):
            pass

        def warning(self, message):
            pass

        def error(self, message):
            captured["error"] = message

    class Model:
        def load_state_dict(self, state_dict):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {}},
    )

    def fake_export(model, args, output_path, **kwargs):
        captured["input"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    export_onnx(
        Model(),
        {"model": {"img_size": 64}, "export": {"onnx_opset": 14}},
        weights_dir,
        torch.device("cpu"),
        Logger(),
    )

    assert captured["input"].shape == (1, 3, 64, 64)
    assert captured["kwargs"]["dynamo"] is False
    assert "error" not in captured
