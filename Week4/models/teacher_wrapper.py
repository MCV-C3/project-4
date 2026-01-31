import torch
import torch.nn as nn
from torchvision import models
from typing import List, Callable, Type


class TeacherWrapperModel(nn.Module):
    def __init__(self, num_classes: int, truncation_level: int = 4):
        super(TeacherWrapperModel, self).__init__()

        self.num_classes = num_classes
        self.truncation_level = truncation_level

        # Load pretrained ResNeXt101
        # We assume weights are cached or available
        full_model = models.resnext101_32x8d(weights="IMAGENET1K_V1")

        # Extract Stem (Initial layers)
        stem = [full_model.conv1, full_model.bn1,
                full_model.relu, full_model.maxpool]

        # Build Truncated Backbone
        # Stage references and their output channels
        stages = [
            ("layer1", full_model.layer1, 256),
            ("layer2", full_model.layer2, 512),
            ("layer3", full_model.layer3, 1024),
            ("layer4", full_model.layer4, 2048)
        ]

        selected_modules = stem
        current_channels = 64  # output of stem

        self.stage_names = []
        for i in range(truncation_level):
            name, module, channels = stages[i]
            selected_modules.append(module)
            self.stage_names.append(name)
            current_channels = channels

        self.backbone = nn.Sequential(*selected_modules)
        self.avgpool = full_model.avgpool
        self.classifier_in_features = current_channels

        # Initialize Classifier
        self.backbone_fc = nn.Linear(self.classifier_in_features, num_classes)

        # Track configuration
        self.head_config = {"type": "linear", "hidden_dims": None}
        self.unfrozen_blocks = 0

    def modify_classifier_head(
        self,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        normalization: str = "none"
    ):
        # Get activation function
        act_fn = self._get_activation(activation)

        def get_norm_layer(name, channels):
            if name == 'batch':
                return nn.BatchNorm1d(channels)
            elif name == 'layer':
                return nn.LayerNorm(channels)
            elif name == 'instance':
                return nn.InstanceNorm1d(channels)
            elif name == 'group':
                groups = 32 if channels % 32 == 0 else max(1, channels // 2)
                return nn.GroupNorm(groups, channels)
            return None

        if hidden_dims is None or len(hidden_dims) == 0:
            layers = []
            norm_layer = get_norm_layer(
                normalization, self.classifier_in_features)
            if norm_layer:
                layers.append(norm_layer)

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            layers.append(
                nn.Linear(self.classifier_in_features, self.num_classes))

            if len(layers) == 1:
                new_head = layers[0]
            else:
                new_head = nn.Sequential(*layers)

            self.head_config = {"type": "linear", "hidden_dims": None,
                                "dropout": dropout, "normalization": normalization}
        else:
            layers = []
            current_dim = self.classifier_in_features

            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_dim, hidden_dim))

                norm_layer = get_norm_layer(normalization, hidden_dim)
                if norm_layer:
                    layers.append(norm_layer)

                layers.append(act_fn)

                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

                current_dim = hidden_dim

            layers.append(nn.Linear(current_dim, self.num_classes))

            new_head = nn.Sequential(*layers)
            self.head_config = {
                "type": "sequential",
                "hidden_dims": hidden_dims,
                "activation": activation,
                "dropout": dropout
            }

        self.backbone_fc = new_head
        return self

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone_fc(x)

    def set_parameter_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad
