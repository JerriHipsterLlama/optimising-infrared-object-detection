import torch
import cv2
import numpy as np
from torch import nn

from infrared_detection.compression.pruning.fcpts import CAMELCalibrationDataset, yolo_feature_adapter


class _FakeYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


def test_yolo_adapter_applies_parameter_override():
    model = _FakeYolo().eval()
    batch = torch.ones(1, 1, 4, 4)
    params = dict(model.named_parameters())
    masked = {name: torch.zeros_like(value) for name, value in params.items()}

    dense_output = yolo_feature_adapter(model, batch)
    masked_output = yolo_feature_adapter(model, batch, params_override=masked)

    assert not torch.allclose(dense_output[0], masked_output[0])


def test_calibration_dataset_uses_configured_image_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "sample.jpg"), np.full((8, 8), 127, dtype=np.uint8))

    dataset = CAMELCalibrationDataset(image_dir, limit=1, image_size=32)

    assert len(dataset) == 1
    assert dataset[0].shape == (1, 32, 32)
