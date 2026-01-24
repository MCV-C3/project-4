import torch
import torch.nn as nn
from .attention_modules import CBAM


class SmartBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dropout, use_bn=False, 
                 use_bottleneck=False, use_attention=False, use_residual=False):
        super(SmartBlock, self).__init__()
        self.use_residual = use_residual
        
        layers = []
        
        # A. 1x1 Bottleneck (Reduces dim before 3x3 conv to save params)
        if use_bottleneck:
            mid_c = out_c // 4 # Compress by factor of 4
            layers.append(nn.Conv2d(in_c, mid_c, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            current_in = mid_c
        else:
            current_in = in_c

        # B. Main Convolution
        layers.append(nn.Conv2d(current_in, out_c, kernel_size=kernel_size, padding=kernel_size//2))
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        
        layers.append(nn.ReLU(inplace=True))
        
        # C. Attention (CBAM)
        if use_attention:
            layers.append(CBAM(out_c))
            
        # D. Pooling & Dropout (Standard components)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
            
        self.body = nn.Sequential(*layers)
        
        # E. Residual Shortcut Handling
        # If dimensions change (channels or size), we need a 1x1 conv on the shortcut
        self.shortcut = nn.Sequential()
        if use_residual and (in_c != out_c):
            # Note: We use stride=2 in shortcut because the main path has MaxPool(2)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=2, bias=False)
            )
        elif use_residual:
             # Just subsample spatial dim if channels match
             self.shortcut = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.body(x)
        if self.use_residual:
            out += self.shortcut(x)
        return out


class ExperimentalCNN(nn.Module):
    def __init__(self, num_classes, img_size, channels, 
                 use_gap=False, use_attention=False, use_bottleneck=False, 
                 use_residual=False, use_bn=False, dropout=0.0):
        super(ExperimentalCNN, self).__init__()
        
        self.features = nn.Sequential()
        input_channels = 3
        
        # Create the feature extractor blocks
        for i, out_channels in enumerate(channels):
            block = SmartBlock(
                in_c=input_channels, 
                out_c=out_channels, 
                kernel_size=3, 
                dropout=dropout,
                use_bottleneck=use_bottleneck,
                use_attention=use_attention,
                use_residual=use_residual,
                use_bn=use_bn
            )
            self.features.add_module(f"block{i}", block)
            input_channels = out_channels

        # Create the classifier
        if use_gap:
            # GAP Strategy: Average the spatial dims -> Linear Layer
            # Greatly reduces parameters
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(input_channels, num_classes)
            )
        else:
            # Baseline Strategy: Flatten -> Linear Layer
            num_pools = len(channels)
            final_dim = img_size // (2 ** num_pools)
            flatten_dim = input_channels * final_dim * final_dim
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flatten_dim, num_classes)
            )

        self.backbone = self.features

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x