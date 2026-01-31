import torch
import torch.nn as nn
from typing import List, Optional


class Fire(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ):
        super(Fire, self).__init__()

        # Squeeze layer: reduces channels
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
        )

        # Expand 1x1 branch
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU(inplace=True),
        )

        # Expand 3x3 branch (with padding to maintain spatial dimensions)
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU(inplace=True),
        )

        # Output channels = expand1x1 + expand3x3
        self.out_channels = expand1x1_channels + expand3x3_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = self.squeeze(x)
        return torch.cat([self.expand1x1(squeezed), self.expand3x3(squeezed)], dim=1)


class FireWithBypass(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ):
        super(FireWithBypass, self).__init__()

        self.fire = Fire(
            in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
        )
        self.out_channels = self.fire.out_channels
        self.use_bypass = in_channels == self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fire(x)
        if self.use_bypass:
            out = out + x
        return out


class SqueezeNetMini(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        fire_config: Optional[List[tuple]] = None,
        use_bypass: bool = True,
        dropout: float = 0.5,
        stem_channels: int = 32,
    ):
        super(SqueezeNetMini, self).__init__()

        # Format: (squeeze_channels, expand1x1_channels, expand3x3_channels)
        if fire_config is None:
            # Scaled down version of original SqueezeNet
            fire_config = [
                (base_channels, base_channels, base_channels),  # Fire1
                (base_channels, base_channels, base_channels),  # Fire2
                (base_channels * 2, base_channels * 2, base_channels * 2),  # Fire3
                (base_channels * 2, base_channels * 2, base_channels * 2),  # Fire4
                (base_channels * 3, base_channels * 3, base_channels * 3),  # Fire5
                (base_channels * 3, base_channels * 3, base_channels * 3),  # Fire6
                (base_channels * 4, base_channels * 4, base_channels * 4),  # Fire7
                (base_channels * 4, base_channels * 4, base_channels * 4),  # Fire8
            ]

        self.use_bypass = use_bypass
        FireModule = FireWithBypass if use_bypass else Fire

        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build Fire modules
        fire_modules = []
        in_channels = stem_channels
        pool_indices = [2, 4]  # Pool after Fire3 and Fire5 (0-indexed: 2, 4)

        for i, (squeeze, expand1x1, expand3x3) in enumerate(fire_config):
            fire = FireModule(in_channels, squeeze, expand1x1, expand3x3)
            fire_modules.append(fire)
            in_channels = fire.out_channels

            # Add pooling after certain fire modules
            if i in pool_indices and i < len(fire_config) - 1:
                fire_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.features = nn.Sequential(*fire_modules)

        # Final convolution (acts like a 1x1 conv classifier)
        self.final_conv = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone = nn.Sequential(self.stem, self.features)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize model weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        # Extract feature maps for GradCAM visualization
        x = self.stem(x)
        x = self.features(x)
        return x


SQUEEZENET_CONFIGS = {
    # Extremely tiny models (for parameter efficiency experiments)
    "yocto": {
        "stem_channels": 8,
        "fire_config": [
            (2, 4, 4),  # 8 out
            (2, 4, 4),  # 8 out
            (4, 8, 8),  # 16 out
            (4, 8, 8),  # 16 out
        ],
        "use_bypass": True,
        "dropout": 0.3,
    },
    "zepto": {
        "stem_channels": 12,
        "fire_config": [
            (4, 6, 6),  # 12 out
            (4, 6, 6),  # 12 out
            (6, 12, 12),  # 24 out
            (6, 12, 12),  # 24 out
            (8, 16, 16),  # 32 out
        ],
        "use_bypass": True,
        "dropout": 0.3,
    },
    "atto": {
        "stem_channels": 16,
        "fire_config": [
            (4, 8, 8),  # 16 out
            (4, 8, 8),  # 16 out
            (8, 16, 16),  # 32 out
            (8, 16, 16),  # 32 out
            (12, 24, 24),  # 48 out
            (12, 24, 24),  # 48 out
        ],
        "use_bypass": True,
        "dropout": 0.4,
    },
    "femto": {
        "stem_channels": 24,
        "fire_config": [
            (6, 12, 12),  # 24 out
            (6, 12, 12),  # 24 out
            (12, 24, 24),  # 48 out
            (12, 24, 24),  # 48 out
            (16, 32, 32),  # 64 out
            (16, 32, 32),  # 64 out
        ],
        "use_bypass": True,
        "dropout": 0.4,
    },
    "pico": {
        "stem_channels": 32,
        "fire_config": [
            (8, 16, 16),  # 32 out
            (8, 16, 16),  # 32 out
            (16, 32, 32),  # 64 out
            (16, 32, 32),  # 64 out
            (24, 48, 48),  # 96 out
            (24, 48, 48),  # 96 out
        ],
        "use_bypass": True,
        "dropout": 0.5,
    },
    "nano": {
        "stem_channels": 48,
        "fire_config": [
            (12, 24, 24),  # 48 out
            (12, 24, 24),  # 48 out
            (24, 48, 48),  # 96 out
            (24, 48, 48),  # 96 out
            (32, 64, 64),  # 128 out
            (32, 64, 64),  # 128 out
            (48, 96, 96),  # 192 out
        ],
        "use_bypass": True,
        "dropout": 0.5,
    },
    "micro": {
        "stem_channels": 64,
        "fire_config": [
            (16, 32, 32),  # 64 out
            (16, 32, 32),  # 64 out
            (32, 64, 64),  # 128 out
            (32, 64, 64),  # 128 out
            (48, 96, 96),  # 192 out
            (48, 96, 96),  # 192 out
            (64, 128, 128),  # 256 out
            (64, 128, 128),  # 256 out
        ],
        "use_bypass": True,
        "dropout": 0.5,
    },
    "mini": {
        "stem_channels": 96,
        "fire_config": [
            (16, 64, 64),  # 128 out
            (16, 64, 64),  # 128 out
            (32, 128, 128),  # 256 out
            (32, 128, 128),  # 256 out
            (48, 192, 192),  # 384 out
            (48, 192, 192),  # 384 out
            (64, 256, 256),  # 512 out
            (64, 256, 256),  # 512 out
        ],
        "use_bypass": True,
        "dropout": 0.5,
    },
    # Standard SqueezeNet 1.1 configuration (scaled down for small datasets)
    "standard": {
        "stem_channels": 64,
        "fire_config": [
            (16, 64, 64),  # Fire2: 128 out
            (16, 64, 64),  # Fire3: 128 out
            (32, 128, 128),  # Fire4: 256 out
            (32, 128, 128),  # Fire5: 256 out
            (48, 192, 192),  # Fire6: 384 out
            (48, 192, 192),  # Fire7: 384 out
            (64, 256, 256),  # Fire8: 512 out
            (64, 256, 256),  # Fire9: 512 out
        ],
        "use_bypass": True,
        "dropout": 0.5,
    },
}


def create_squeezenet_preset(preset: str, num_classes: int) -> SqueezeNetMini:
    if preset not in SQUEEZENET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(SQUEEZENET_CONFIGS.keys())}"
        )

    config = SQUEEZENET_CONFIGS[preset]
    return SqueezeNetMini(num_classes=num_classes, **config)


SqueezeNet = SqueezeNetMini
