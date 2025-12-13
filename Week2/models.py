
import torch.nn as nn
import torch
from typing import *

class MLP(nn.Module):

    def __init__(self, input_d: int, output_d: int, hidden_layers: List[int] = [300], dropout: float = 0.0, activation: str = "leaky_relu"):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        
        # Select activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Activation {activation} not supported.")
        
        # Build hidden layers
        current_dim = input_d
        for hidden_dim in hidden_layers:
            # Block: Linear -> BatchNorm -> Activation -> Dropout
            block = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn,
                nn.Dropout(p=dropout)
            )
            self.layers.append(block)
            current_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(current_dim, output_d)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        
        for layer in self.layers:
            x = layer(x)
            
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
            if i == layer_index:
                return x
        
        return x