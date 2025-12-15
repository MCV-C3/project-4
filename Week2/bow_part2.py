import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
import wandb
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import MLP
from utils import get_patches


class DeepBoW:
    def __init__(self, model, device, layer_index, codebook_size=200, patch_size=28, stride=28):
        self.model = model
        self.device = device
        self.layer_index = layer_index
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.stride = stride
        
        # Initialize KMeans (Codebook)
        self.kmeans = MiniBatchKMeans(
            n_clusters=codebook_size, 
            batch_size=1000,
            random_state=42,
            n_init='auto'
        )
        self.is_fitted = False

    def extract_patch_descriptors(self, dataloader, desc="Extracting Descriptors"):
        """
        Runs the MLP on patches and returns a list of descriptor arrays.
        Returns: List of numpy arrays, where each array is (Num_Patches, Feature_Dim)
        """
        self.model.eval()
        all_descriptors = [] # List of (N_patches, Dim) for each image
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(dataloader, desc=desc):
                inputs = inputs.to(self.device)
                
                # 1. Extract Patches from the batch
                # Shape: (Batch * N_Patches, 3, 64, 64)
                patches = get_patches(inputs, self.patch_size, self.stride)
                
                # 2. Feed Patches through MLP to get dense descriptors
                # Shape: (Batch * N_Patches, Hidden_Dim)
                # Note: We must ensure MLP is init with correct input_d corresponding to patch size
                features = self.model.get_features(patches, layer_index=self.layer_index)
                
                # 3. Reshape back to (Batch, N_Patches, Hidden_Dim)
                # We need to know N_Patches per image
                n_patches = features.shape[0] // inputs.shape[0]
                features = features.view(inputs.shape[0], n_patches, -1)
                
                # 4. Store as numpy
                features_np = features.cpu().numpy()
                
                for i in range(inputs.shape[0]):
                    all_descriptors.append(features_np[i]) # Store (N_Patches, Dim)
                    all_labels.append(labels[i].item())
                    
        return all_descriptors, np.array(all_labels)

    def fit_codebook(self, descriptors_list):
        """
        Fits K-Means on a subset of all extracted patch descriptors.
        """
        print("Stacking descriptors for Codebook training...")
        # Stack all patches from all images: (Total_Patches, Dim)
        # To save RAM, we can sample if dataset is huge, but usually fine for this size
        all_features = np.vstack(descriptors_list)
        
        print(f"Fitting KMeans (k={self.codebook_size}) on {all_features.shape[0]} descriptors...")
        self.kmeans.fit(all_features)
        self.is_fitted = True
        print("Codebook fitted.")

    def build_histograms(self, descriptors_list):
        """
        Converts list of patch descriptors into Bag-of-Words histograms.
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted!")
            
        histograms = []
        
        for img_descs in tqdm.tqdm(descriptors_list, desc="Building Histograms"):
            # img_descs shape: (N_Patches, Dim)
            
            # 1. Map each patch to visual word
            visual_words = self.kmeans.predict(img_descs)
            
            # 2. Compute Histogram
            hist, _ = np.histogram(visual_words, bins=range(self.codebook_size + 1), density=False)
            
            histograms.append(hist)
            
        return np.array(histograms, dtype=np.float32)

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    
    # --- CONFIGURATION (Based on your request) ---
    CONFIG = {
        "dataset_path": "/ghome/group04/mcv/datasets/C3/2526/places_reduced", 
        "model_path": "experiments/patching_aggregation_study/size_28_mean/size_28_mean.pth",
        "img_size": 224, 
        "batch_size": 256, # Lower batch size because patches multiply memory usage
        
        # BoW Params
        "patch_size": 28, # Small patches
        "stride": 28,     # Overlap
        "layer_index": 3, # Which MLP layer to use as descriptor? (0-indexed)
        "codebook_size": 200, # Per your best config
        
        # Classifier
        "C": 1.0 # Logistic Regression C
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Data...")
    transform = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(CONFIG["img_size"], CONFIG["img_size"])),
        # Normalization is important for MLP features
        # F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageFolder(os.path.join(CONFIG["dataset_path"], "train"), transform=transform)
    test_dataset = ImageFolder(os.path.join(CONFIG["dataset_path"], "val"), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], pin_memory=True, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], pin_memory=True, shuffle=False, num_workers=4)

    # 2. Load Model
    # CRITICAL: Initialize MLP with the PATCH dimensions, not Image dimensions
    # The MLP processes a 64x64 patch, so input_d must match that.
    C, H, W = 3, CONFIG["patch_size"], CONFIG["patch_size"]
    input_d = C * H * W 
    output_d = len(train_dataset.classes)
    
    # Assuming standard architecture from your config [300] or similar
    # You must match the architecture of the saved .pth
    model = MLP(input_d=input_d, output_d=output_d, hidden_layers=[1024, 512, 256, 128]) 
    
    print(f"Loading weights from {CONFIG['model_path']}")
    # Note: If your saved model was trained on 224x224, this might fail unless 
    # you trained it on patches originally. 
    # If the model expects 224x224 input, you CANNOT use it for 64x64 patches without resizing patches 
    # (which is slow and bad) or using a Fully Convolutional approach.
    # ASSUMPTION: You are using the model trained on PATCHES (Week 2 Part 2).
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.to(device)

    # 3. Initialize Pipeline
    deep_bow = DeepBoW(
        model=model, 
        device=device, 
        layer_index=CONFIG["layer_index"],
        codebook_size=CONFIG["codebook_size"],
        patch_size=CONFIG["patch_size"],
        stride=CONFIG["stride"]
    )

    # 4. Extract Raw Descriptors
    print("\n--- Phase 1: Feature Extraction ---")
    train_descs, y_train = deep_bow.extract_patch_descriptors(train_loader, desc="Extracting Train Features")
    test_descs, y_test = deep_bow.extract_patch_descriptors(test_loader, desc="Extracting Test Features")
    
    # 5. Build Codebook (Train only)
    print("\n--- Phase 2: Codebook Generation ---")
    deep_bow.fit_codebook(train_descs)
    
    # 6. Build Histograms
    print("\n--- Phase 3: Histogram Construction ---")
    X_train_bow = deep_bow.build_histograms(train_descs)
    X_test_bow = deep_bow.build_histograms(test_descs)
    
    print(f"BoW Train Shape: {X_train_bow.shape}")
    print(f"BoW Test Shape: {X_test_bow.shape}")

    # 7. Post-Processing (Your Best Config: L2 + MinMax)
    print("\n--- Phase 4: Normalization (L2 + MinMax) ---")
    
    # L2 Normalization
    l2_norm = Normalizer(norm='l2')
    X_train_bow = l2_norm.transform(X_train_bow)
    X_test_bow = l2_norm.transform(X_test_bow)
    
    # MinMax Scaling
    mm_scaler = MinMaxScaler()
    X_train_bow = mm_scaler.fit_transform(X_train_bow)
    X_test_bow = mm_scaler.transform(X_test_bow)

    # 8. Classification (Logistic Regression)
    print(f"\n--- Phase 5: Classification (LogReg C={CONFIG['C']}) ---")
    clf = LogisticRegression(C=CONFIG["C"], max_iter=1000, random_state=42)
    clf.fit(X_train_bow, y_train)
    
    y_pred = clf.predict(X_test_bow)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
    
    # Optional: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"BoW MLP (Acc: {acc:.2f})")
    plt.savefig("bow_mlp_confusion_matrix.png")
    print("Saved confusion matrix.")