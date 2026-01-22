from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """MobileNetV1 block: depthwise 3x3 + pointwise 1x1."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch,
                bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 block: expand 1x1 -> depthwise 3x3 -> project 1x1 (linear)."""
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            stride: int, 
            expand_ratio: int
        ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        if expand_ratio < 1:
            raise ValueError("expand_ratio must be >= 1")

        hidden_ch = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = []

        # expand (optional)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ]
        else:
            hidden_ch = in_ch

        # depthwise
        layers += [
            nn.Conv2d(
                hidden_ch, hidden_ch, 3, stride=stride, padding=1,
                groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
        ]

        # project (linear bottleneck)
        layers += [
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_res else out


class MobileNet(nn.Module):
    """
    MobileNet with selectable version.

    V1 spec format:
      v1_spec = [(stem_out, stem_stride), (out_ch, stride), (out_ch, stride), ...]

    V2 spec format (flat, no repeats):
      v2_spec = [(stem_out, stem_stride), (out_ch, stride, expand), 
                 (out_ch, stride, expand), ...]

    """
    def __init__(
        self,
        num_classes: int,
        version: Literal["v1", "v2"] = "v1",
        alpha: float = 1.0,
        v1_spec: Optional[Sequence[Tuple[int, int]]] = None,
        v2_spec: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        def c(ch: int) -> int:
            return max(1, int(ch * alpha))

        layers: List[nn.Module] = []

        if version == "v1":
            if not v1_spec:
                raise ValueError("v1_spec must be provided when version='v1'")

            stem_out, stem_stride = v1_spec[0]
            cur_ch = c(stem_out)

            layers.append(nn.Sequential(
                nn.Conv2d(
                    3, cur_ch, 3, stride=stem_stride, padding=1, bias=False),
                nn.BatchNorm2d(cur_ch),
                nn.ReLU(inplace=True),
            ))

            for out_ch, stride in v1_spec[1:]:
                next_ch = c(out_ch)
                layers.append(DepthwiseSeparableConv(
                    cur_ch, next_ch, stride=stride))
                cur_ch = next_ch

        elif version == "v2":
            if not v2_spec:
                raise ValueError("v2_spec must be provided when version='v2'")

            stem_out, stem_stride = v2_spec[0]
            cur_ch = c(stem_out)

            layers.append(nn.Sequential(
                nn.Conv2d(3, cur_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cur_ch),
                nn.ReLU6(inplace=True),
            ))

            # Flat list of inverted residual blocks
            for spec_item in v2_spec[1:]:
                if len(spec_item) == 2:
                    out_ch, stride = spec_item
                    expand = 1
                elif len(spec_item) == 3:
                    out_ch, stride, expand = spec_item
                else:
                    raise ValueError(
                        "v2_spec must be contain elements of length 2 of 3, "
                        f"instead, element of length {len(spec_item)} was provided")
                
                next_ch = c(out_ch)
                layers.append(InvertedResidual(
                    cur_ch, next_ch, stride=stride, expand_ratio=expand))
                cur_ch = next_ch

        else:
            raise ValueError("version must be 'v1' or 'v2'")

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(cur_ch, num_classes)

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
        return self.classifier(x)
