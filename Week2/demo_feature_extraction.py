import torch
import torch.nn as nn
from models import MLP
import torchvision.transforms.v2 as F
from PIL import Image
import os
import json

# 1. Load Configuration
with open("configs/config.json", 'r') as f:
    config = json.load(f)

# 2. Initialize Model
# We need to calculate input dimensions exactly like in main.py
# For this demo, I'll assume standard 224x224 RGB images
C, H, W = 3, 224, 224
num_classes = 8 # MIT 8 Scene dataset
hidden_layers = config["model_params"]["hidden_layers"]

model = MLP(input_d=C*H*W, output_d=num_classes, hidden_layers=hidden_layers)
# Load weights if you have trained them (optional for demo)
# model.load_state_dict(torch.load("experiments/mlp_enhanced_run/mlp_enhanced_run.pth"))
model.eval()

# 3. Load and Preprocess an Image
# Pick a random image from the train set
dataset_path = os.path.expanduser(config["dataset_path"])
image_path = "/ghome/group04/mcv/datasets/C3/2526/places_reduced/train/industrial and construction/junkyard_00004500.jpg"

if not os.path.exists(image_path):
    # Fallback if specific file doesn't exist, finding first jpg
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                break
        if image_path: break

print(f"Processing image: {image_path}")

# Same transforms as training
transformation = F.Compose([
    F.ToImage(),
    F.ToDtype(torch.float32, scale=True),
    F.Resize(size=(224, 224)),
])

img = Image.open(image_path).convert('RGB')
img_tensor = transformation(img)
img_tensor = img_tensor.unsqueeze(0) # Add batch dimension -> (1, 3, 224, 224)

# 4. Extract Features
# Get features from the first hidden layer (index 0)
features_layer0 = model.get_features(img_tensor, layer_index=0)
print(f"\nFeature extraction from Hidden Layer 0 ({hidden_layers[0]} units):")
print(f"Shape: {features_layer0.shape}") # Should be (1, units)
print(f"First 10 values: {features_layer0[0][:10].detach().numpy()}")

# If you had more layers, you could do:
# features_layer1 = model.get_features(img_tensor, layer_index=1)
