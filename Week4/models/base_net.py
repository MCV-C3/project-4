import torch.nn as nn
import torch

from typing import *


class BasicCNN(nn.Module):
    def __init__(self, num_classes, img_size, kernel_size, channels, fc_units, use_bn=False, dropout=0.0):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential()
        input_channels = 3

        # Dynamically build convolutional layers based on config list
        for i, out_channels in enumerate(channels):
            self.features.add_module(f"conv{i}", nn.Conv2d(
                input_channels, out_channels, kernel_size=kernel_size, padding=1))

            if use_bn:
                self.features.add_module(
                    f"bn{i}", nn.BatchNorm2d(out_channels))

            self.features.add_module(f"relu{i}", nn.ReLU(inplace=True))
            self.features.add_module(
                f"pool{i}", nn.MaxPool2d(kernel_size=2, stride=2))

            self.features.add_module(f"dropout{i}", nn.Dropout2d(p=dropout))

            input_channels = out_channels

        # Calculate flattened size automatically
        # img_size is reduced by factor of 2 for each pooling layer
        num_pools = len(channels)
        final_dim = img_size // (2 ** num_pools)
        self.flatten_dim = input_channels * final_dim * final_dim

        # Fully Connected Blocks
        layers = [nn.Flatten()]
        input_dim = self.flatten_dim

        # Handle multiple dense layers (e.g., [4096, 4096])
        if isinstance(fc_units, int):
            fc_units = [fc_units]

        for i, units in enumerate(fc_units):
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU(inplace=True))

            if dropout > 0:
                layers.append(nn.Dropout(p=0.5))
            input_dim = units

        # Final Output Layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Save backbone for GradCAM
        self.backbone = self.features

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # Hook for GradCAM
    def extract_feature_maps(self, x):
        return self.features(x)
