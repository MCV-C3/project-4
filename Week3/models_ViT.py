
import torch.nn as nn
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

from transformers import ViTForImageClassification

from typing import *
from torchview import draw_graph
from graphviz import Source

from PIL import Image
import torchvision.transforms.v2  as F
import numpy as np 


class WraperModel(nn.Module):
    def __init__(self, num_classes: int, model_name):
        super(WraperModel, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained VGG16 model
        self.backbone = ViTForImageClassification.from_pretrained(
            self.model_name, ignore_mismatched_sizes=True, use_safetensors=True)
        
        # freeze all model layers
        self.set_parameter_requires_grad()

        self.classifier_in_features = self.backbone.classifier.in_features

        # Modify the classifier for the number of classes
        # by default requires_grad=True
        self.backbone.classifier = nn.Linear(
            self.classifier_in_features, self.num_classes)

    def modify_classifier_head(
        self, 
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        normalization: str = "none"
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
        self.backbone.classifier = new_head

        print(f"\nNew head structure:")
        print(f"{self.backbone.classifier}")
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


    def forward(self, x):
        outputs = self.backbone.vit(x)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        return self.backbone.classifier(pooled_output)
    

    def extract_feature_maps(self, input_image:torch.Tensor):

        conv_weights =[]
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.features.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)


        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        x= torch.clone(input=input_image)
        for layer in conv_layers:
            x = layer(x)
            feature_maps.append(x)
            layer_names.append(str(layer))

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
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs


    def set_parameter_requires_grad(self):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False



    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam


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
        print(f"Backbone: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        #print(f"Classifier input features: {self.classifier_in_features}")
        print(f"Total parameters: {self.get_total_parameters():,}")
        print(f"Trainable parameters: {self.get_trainable_parameters():,}")
        print(
            f"Frozen parameters: {self.get_total_parameters() - self.get_trainable_parameters():,}"
        )
        #print(f"\nClassification head:")
        #print(f"  {self.backbone_fc}")
        print("=" * 60 + "\n")




# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = WraperModel(num_classes=8, feature_extraction=False)
    #model.load_state_dict(torch.load("saved_model.pt"))
    #model = model

    """
        features.0
        features.2
        features.5
        features.7
        features.10
        features.12
        features.14
        features.17
        features.19
        features.21
        features.24
        features.26
        features.28
    """

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.RandomHorizontalFlip(p=1.),
                                    F.Resize(size=(256, 256)),
                                ])
    # Example GradCAM usage
    dummy_input = Image.open("/home/cboned/data/Master/MIT_split/test/highway/art803.jpg")#torch.randn(1, 3, 224, 224)
    input_image = transformation(dummy_input).unsqueeze(0)



    target_layers = [model.backbone.features[26]]
    targets = [ClassifierOutputTarget(6)]
    
    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min()) 
    ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    
    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)

    visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    # Display processed feature maps shapes
    feature_maps, layer_names = model.extract_feature_maps(input_image)

                                                                 ### Aggregate the feature maps
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        min_feature_map, min_index = torch.min(feature_map, 0) # Get the min across channels
        processed_feature_maps.append(min_feature_map.data.cpu().numpy())
    
    
    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i], cmap="hot", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)


    plt.show()

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=input_image, layers=["features.28"]))["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0) 

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()



    ## Draw the model
    model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta', expand_nested=True, roll=True)
    model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")