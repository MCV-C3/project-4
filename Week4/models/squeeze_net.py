import torch.nn as nn
from torchvision import models


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int, feature_extraction: bool = True):
        super(SqueezeNet, self).__init__()

        self.model = models.squeezenet1_1(weights='IMAGENET1K_V1')

        if feature_extraction:
            self.set_parameter_requires_grad(
                feature_extracting=feature_extraction)

        # Get the input channels of the existing classifier conv layer
        in_channels = self.model.classifier[1].in_channels

        # Replace with a new Conv2d for our specific num_classes
        self.model.classifier[1] = nn.Conv2d(
            in_channels, num_classes, kernel_size=(1, 1))

        # Initialize the new layer weights
        nn.init.normal_(self.model.classifier[1].weight, mean=0.0, std=0.01)

        # Compatibility attribute for your GradCAM code in main.py
        # Your main.py expects 'model.backbone' to exist for target layers
        self.backbone = self.model.features

    def forward(self, x):
        return self.model(x)

    def extract_feature_maps(self, x):
        """Hook for GradCAM or visualization"""
        return self.model.features(x)

    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the 
        training process.
        """
        if feature_extracting:
            for param in self.model.features.parameters():
                param.requires_grad = False
