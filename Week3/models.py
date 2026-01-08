import torch
import torch.nn as nn
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
import matplotlib.pyplot as plt

from typing import *
from torchview import draw_graph
from graphviz import Source

from PIL import Image
import torchvision.transforms.v2 as F
import numpy as np

import pdb


class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d

        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)

        return x


class WraperModel(nn.Module):
    def __init__(self, num_classes: int, truncation_level: int = 4):
        super(WraperModel, self).__init__()

        self.num_classes = num_classes
        self.truncation_level = truncation_level

        # Load pretrained ResNeXt101
        full_model = models.resnext101_32x8d(weights="IMAGENET1K_V1", progress=True)

        # Extract Stem (Initial layers)
        stem = [full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool]
        
        # Build Truncated Backbone
        # Stage references and their output channels
        stages = [
            ("layer1", full_model.layer1, 256),
            ("layer2", full_model.layer2, 512),
            ("layer3", full_model.layer3, 1024),
            ("layer4", full_model.layer4, 2048)
        ]
        
        selected_modules = stem
        current_channels = 64 # output of stem
        
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
        
        # Freeze everything initially
        self.set_parameter_requires_grad()

        # Track configuration
        self.head_config = {"type": "linear", "hidden_dims": None}
        self.unfrozen_blocks = 0

    def modify_classifier_head(
        self, hidden_dims: List[int] = None, activation: str = "relu", dropout: float = 0.0, normalization: str = "none"
    ):
        """
        Examples:
            # Original architecture (no hidden layers)
            model.modify_classifier_head(hidden_dims=None)

            # Add one hidden layer
            model.modify_classifier_head(hidden_dims=[512])

            # With Normalization
            model.modify_classifier_head(hidden_dims=[512], normalization='batch')
        """
        print("*" * 25)
        print(
            f"Modifying Classifier Head Configuration:\nInput features: {self.classifier_in_features}"
        )
        print(f"Output classes: {self.num_classes}")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Activation: {activation}")
        print(f"Dropout: {dropout}")
        print(f"Normalization: {normalization}")

        # Get activation function
        act_fn = self._get_activation(activation)

        # Helper to get 1D norm layer
        def get_norm_layer(name, channels):
            if name == 'batch':
                return nn.BatchNorm1d(channels)
            elif name == 'layer':
                return nn.LayerNorm(channels)
            elif name == 'instance':
                return nn.InstanceNorm1d(channels)
            elif name == 'group':
                 # GroupNorm requires num_groups. Defaulting to 32 or channels/2
                 groups = 32 if channels % 32 == 0 else max(1, channels // 2)
                 return nn.GroupNorm(groups, channels)
            return None

        # Build the new classification head
        if hidden_dims is None or len(hidden_dims) == 0:
            # Direct classification (no hidden layers)
            # Standard practice: Dropout -> Linear (last layer)
            # Normalization on the input features? Usually done by the backbone's last bn.
            # But we can add it if requested.
            layers = []
            
            # Optional: Norm before final layer?
            # If we are strictly following "Head" config, maybe not.
            # But let's support it if explicitly asked.
            norm_layer = get_norm_layer(normalization, self.classifier_in_features)
            if norm_layer:
                layers.append(norm_layer)

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            layers.append(nn.Linear(self.classifier_in_features, self.num_classes))
            
            if len(layers) == 1:
                new_head = layers[0]
            else:
                new_head = nn.Sequential(*layers)
            
            self.head_config = {"type": "linear", "hidden_dims": None, "dropout": dropout, "normalization": normalization}
        else:
            # Build sequential head with hidden layers
            layers = []  # [nn.Flatten()]
            current_dim = self.classifier_in_features

            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_dim, hidden_dim))
                
                # Norm -> Activation OR Activation -> Norm?
                # Standard ResNet: Conv -> BN -> ReLU
                # Standard Transformer: Norm -> Attention ...
                # Let's stick to Linear -> Norm -> Activation -> Dropout
                norm_layer = get_norm_layer(normalization, hidden_dim)
                if norm_layer:
                    layers.append(norm_layer)
                
                layers.append(act_fn)
                
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                
                current_dim = hidden_dim
                print(
                    f"Layer {i+1}: Linear({layers[0 if i==0 else -1].in_features if hasattr(layers[0 if i==0 else -1], 'in_features') else 'prev'} -> {hidden_dim}) + Norm({normalization}) + {activation} + Dropout({dropout})"
                )

            # Final classification layer
            layers.append(nn.Linear(current_dim, self.num_classes))
            print(f"Output: Linear({current_dim} -> {self.num_classes})")

            new_head = nn.Sequential(*layers)
            self.head_config = {
                "type": "sequential",
                "hidden_dims": hidden_dims,
                "activation": activation,
                "dropout": dropout
            }

        # Replace the classifier
        self.backbone_fc = new_head

        print(f"\nNew head structure:")
        print(f"{self.backbone_fc}")
        print("*" * 25)

        return self

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),  # Swish
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def fine_tuning(self, unfreeze_blocks: int = 0):
        """

        Sets the trainability of layers based on a desired depth.
        This function implements progressive unfreezing from deep to shallow layers.

        Args:
            unfreeze_blocks (int): The number of convolutional blocks to unfreeze starting from
                the end of the network. MAX = 4 -> [0-4] (0: fc, 1: layer4, 2: layer3...)

        """
        assert (
            0 <= unfreeze_blocks <= 4
        ), "Unfreeze_blocks variable must be an int between 0 and 4."

        self.unfrozen_blocks = unfreeze_blocks
        print(f"\n--- Setting Fine-Tuning (Unfreezing {unfreeze_blocks} blocks) ---")
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Classification head will be always trainable
        # FOR THE MOMENT (see slides Transfer Learning)
        for param in self.backbone_fc.parameters():
            param.requires_grad = True
        print("-> Classifier head (fc) is defrosted.")

        # Unfreeze remaining stages from back to front
        # Note: Since backbone is nn.Sequential, we access from the end
        trainable_stages = 0
        for i in range(unfreeze_blocks):
            idx = -1 - i # Last element, second to last...
            # Ensure we don't try to unfreeze the 'stem' if unfreeze_blocks is high
            if abs(idx) <= len(self.stage_names):
                for param in self.backbone[idx].parameters():
                    param.requires_grad = True
                trainable_stages += 1
        
        self.unfrozen_blocks = trainable_stages

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone_fc(x)

    def get_trainable_parameters(self) -> int:
        # Get the number of trainable parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        # Get the total number of parameters
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        # Print a summary of the model configuration
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Backbone: ResNeXt101_32x8d")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classifier input features: {self.classifier_in_features}")
        print(f"Head configuration: {self.head_config}")
        print(f"Unfrozen blocks: {self.unfrozen_blocks}")
        print(f"Total parameters: {self.get_total_parameters():,}")
        print(f"Trainable parameters: {self.get_trainable_parameters():,}")
        print(
            f"Frozen parameters: {self.get_total_parameters() - self.get_trainable_parameters():,}"
        )
        print(f"\nClassification head:")
        print(f"  {self.backbone_fc}")
        print("=" * 60 + "\n")

    def extract_feature_maps(self, input_image: torch.Tensor):
        conv_layers = []
        layer_names = []

        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
                layer_names.append(name)

        print("TOTAL CONV LAYERS: ", len(conv_layers))

        feature_maps = []
        x = torch.clone(input=input_image)

        # Note: Sequential extraction in ResNets is complex due to skip connections.
        # For a simple sequential test, we'll just store the activations.
        for layer in conv_layers:
            # This only works if layers are strictly sequential, which ResNet is NOT.
            # It is better to use the 'extract_features_from_hooks' method for ResNet.
            pass

        return feature_maps, layer_names

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output

            return hook

        # Register hooks for specified layers
        model_modules = dict(self.backbone.named_modules())

        for layer_name in layers:
            if layer_name in model_modules:
                layer = model_modules[layer_name]
                hooks.append(layer.register_forward_hook(get_activation(layer_name)))
            else:
                print(f"Warning: Layer {layer_name} not found in ResNeXt.")

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.backbone = modify_fn(self.backbone)

    def set_parameter_requires_grad(self):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone_fc.parameters():
            param.requires_grad = False

    def extract_grad_cam(
        self,
        input_image: torch.Tensor,
        target_layer: List[Type[nn.Module]],
        targets: List[Type[ClassifierOutputTarget]],
    ) -> Type[GradCAMPlusPlus]:
        # Ensure gradients are tracked for the input
        input_image.requires_grad = True

        # Ensure gradients are enabled globally for this specific call
        with torch.set_grad_enabled(True):
            with GradCAMPlusPlus(
                model=self, target_layers=target_layer
            ) as cam:
                grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam


# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Default model (original architecture)")
    print("=" * 70)
    model1 = WraperModel(num_classes=8)
    model1.fine_tuning(unfreeze_blocks=0)

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Model with one hidden layer")
    print("=" * 70)
    model2 = WraperModel(num_classes=8)
    model2.modify_classifier_head(hidden_dims=[512], activation="relu")
    model2.fine_tuning(unfreeze_blocks=2)

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Model with two hidden layers (deeper head)")
    print("=" * 70)
    model3 = WraperModel(num_classes=8)
    model3.modify_classifier_head(hidden_dims=[1024, 512], activation="relu")
    model3.fine_tuning(unfreeze_blocks=2)

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Model with smaller head (less deep)")
    print("=" * 70)
    model4 = WraperModel(num_classes=8)
    model4.modify_classifier_head(hidden_dims=[256], activation="relu")
    model4.fine_tuning(unfreeze_blocks=1)

    print("\n" + "=" * 70)
    print("EXAMPLE 5: Test forward pass")
    print("=" * 70)
    dummy_input = torch.randn(2, 3, 224, 224)
    for i, model in enumerate([model1, model2, model3, model4], 1):
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Model {i} output shape: {output.shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

    # GradCAM and Feature Maps on all models
    print("\n" + "=" * 70)
    print("EXAMPLE 6: GradCAM and Feature Maps")
    print("=" * 70)

    transformation = F.Compose(
        [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomHorizontalFlip(p=1.0),
            F.Resize(size=(256, 256)),
        ]
    )

    # Load image
    dummy_input = Image.open(
        "/ghome/group04/mcv/datasets/C3/2425/MIT_small_train_1/test/highway/art803.jpg"
    )
    input_image = transformation(dummy_input).unsqueeze(0)

    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    # Apply GradCAM on all models
    model_names = [
        "Default",
        "One Hidden [512]",
        "Two Hidden [1024,512]",
        "Small [256]",
    ]
    models = [model1, model2, model3, model4]

    for model, name in zip(models, model_names):
        print(f"\n--- GradCAM for {name} ---")

        # For Grad-CAM, target the last conv layer
        target_layers = [model.backbone.layer4[-1].conv3]
        targets = [ClassifierOutputTarget(6)]

        ## Visualize the activation map from Grad Cam
        grad_cams = model.extract_grad_cam(
            input_image=input_image, target_layer=target_layers, targets=targets
        )
        visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

        # Plot the result
        plt.imshow(visualization)
        plt.axis("off")
        plt.title(f"GradCAM: {name}")
        plt.show()

        # Display processed feature maps shapes
        feature_maps, layer_names = model.extract_feature_maps(input_image)

        # Process and visualize feature maps
        processed_feature_maps = []
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)
            min_feature_map, min_index = torch.min(feature_map, 0)
            processed_feature_maps.append(min_feature_map.data.cpu().numpy())

        # Plot All the convolution feature maps separately
        if len(processed_feature_maps) > 0:
            fig = plt.figure(figsize=(30, 50))
            for i in range(len(processed_feature_maps)):
                ax = fig.add_subplot(5, 4, i + 1)
                ax.imshow(
                    processed_feature_maps[i], cmap="hot", interpolation="nearest"
                )
                ax.axis("off")
                ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)
            plt.suptitle(f"Feature Maps: {name}")
            plt.show()

        ## Plot a concrete layer feature map
        with torch.no_grad():
            feature_map = (
                model.extract_features_from_hooks(x=input_image, layers=["layer4.2"])
            )["layer4.2"]
            feature_map = feature_map.squeeze(0)
            print(f"Feature map shape: {feature_map.shape}")
            processed_feature_map, _ = torch.min(feature_map, 0)

        # Plot the result
        plt.imshow(processed_feature_map, cmap="gray")
        plt.axis("off")
        plt.title(f"Layer4.2 Feature Map: {name}")
        plt.show()

        ## Draw the model
        model_graph = draw_graph(
            model,
            input_size=(1, 3, 224, 224),
            device="meta",
            expand_nested=True,
            roll=True,
        )
        model_graph.visual_graph.render(
            filename=f"model_{name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')}",
            format="png",
            directory="../Week3",
        )
