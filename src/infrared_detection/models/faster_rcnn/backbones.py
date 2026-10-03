"""CUDA-compatible torchvision backbones for Faster R-CNN."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.ops import FeaturePyramidNetwork


class ResNet34FPNBackbone(nn.Module):
    """ResNet-34 C2-C5 features adapted to a four-level FPN."""

    out_channels = 256

    def __init__(
        self,
        weights: ResNet34_Weights | None = None,
        trainable_layers: int = 5,
    ) -> None:
        super().__init__()
        backbone = resnet34(weights=weights)
        self.body = nn.ModuleDict(
            {
                "stem": nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                ),
                "layer1": backbone.layer1,
                "layer2": backbone.layer2,
                "layer3": backbone.layer3,
                "layer4": backbone.layer4,
            }
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=self.out_channels,
        )
        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers: int) -> None:
        layers = list(self.body.values())
        trainable_layers = max(0, min(int(trainable_layers), len(layers)))
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        for layer in layers[-trainable_layers:] if trainable_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.fpn.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.body["stem"](x)
        features = OrderedDict()
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = self.body[name](x)
            features[str(len(features))] = x
        return self.fpn(features)
