
import torch.nn as nn
import torch
from typing import *

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, output_d: int, hidden_layers: List[int] = [300]):
        super(SimpleModel, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        
        # Build hidden layers
        current_dim = input_d
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(current_dim, output_d)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            
        x = self.output_layer(x)
        return x

    def get_features(self, x, layer_index: int = 0):
        """
        Extracts features from a specific hidden layer.
        Args:
            x: Input tensor
            layer_index: Index of the hidden layer to extract from (0-indexed)
        """
        x = x.reshape(x.shape[0], -1)
        
        if layer_index >= len(self.layers):
            raise ValueError(f"Layer index {layer_index} out of bounds. Model has {len(self.layers)} hidden layers.")

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            if i == layer_index:
                return x
        
        # If we reach here, it shouldn't happen due to the check above
        return x