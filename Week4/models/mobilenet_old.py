from typing import List

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (groups=in_ch) + pointwise conv (1x1)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):
    """MobileNetV1 for ImageNet-style inputs (default 224x224)."""

    def __init__(
            self,
            num_classes: int,
            out_channels: List[List[int]],
            alpha: float = 1.0
    ) -> None:
        super().__init__()
        assert num_classes > 0
        assert alpha > 0

        def c(ch: int) -> int:
            # width multiplier (alpha) applied to channel counts
            return max(1, int(ch * alpha))

        model_layers = []

        # conv_bn_relu: 3x3
        output_channels, stride = out_channels[0]
        model_layers.append(nn.Sequential(
            nn.Conv2d(3, c(output_channels), kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c(output_channels)),
            nn.ReLU(inplace=True)))

        # Depthwise-separable blocks
        input_channels = output_channels
        dw_channels = out_channels[1:]
        for output_channels, stride in dw_channels:
            model_layers.append(
                DepthwiseSeparableConv(
                    c(input_channels),  c(output_channels), stride=stride))
            input_channels = output_channels

        self.features = nn.Sequential(
            *model_layers
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c(output_channels), num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
