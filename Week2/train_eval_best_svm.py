import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm
from models import MLP

# ==========================================
# CONFIG DEFAULTS
# ==========================================
DEFAULT_CONFIG = {
    "img_size": 32,
    "svm_c": 1,
    "svm_kernel": 'linear',
    "layer_index": "output",
    "output_dir": "results_best_svm",
    "data_config_path": "configs/config_best_ImgSize_32.json",
    "model_weights_path": "best_models/ImgSize_32/ImgSize_32.pth",
    "svm_model_filename": "svm_model.joblib"
}

# ==========================================
# UTILS
# ==========================================

def get_transforms(img_size):
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(img_size, img_size)),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features(model, dataloader, device, layer_index):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            feats = model.get_features(inputs, layer_index=layer_index)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def load_datasets(data_config_path=None, img_size=32):
    """
    Loads training and validation (test) datasets.
    """
    if data_config_path is None:
        data_config_path = DEFAULT_CONFIG["data_config_path"]
        
    print(f"Loading data config from {data_config_path}")
    
    # Path resolution logic
    real_path = data_config_path
    if not os.path.exists(real_path):
        if os.path.exists(os.path.join("Week2", real_path)):
            real_path = os.path.join("Week2", real_path)
    
    with open(real_path, 'r') as f:
        data_config = json.load(f)
    
    dataset_path = data_config.get("dataset_path", "/ghome/group04/C3/Benet/project-4/data/places_reduced")
    model_params = data_config["model_params"]
    
    transform = get_transforms(img_size)
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
    
    return train_dataset, test_dataset, model_params

def prepare_data(train_dataset, test_dataset, model_params, model_weights_path=None, layer_index="output"):
    """
    Loads MLP, extracts features, and preprocesses them.
    (This step could be slow, so ideally we'd cache features too, but for now we focus on SVM caching)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if model_weights_path is None:
        model_weights_path = DEFAULT_CONFIG["model_weights_path"]

    # Resolution for model path
    if not os.path.exists(model_weights_path):
         if os.path.exists(os.path.join("Week2", model_weights_path)):
             model_weights_path = os.path.join("Week2", model_weights_path)

    # 1. Load MLP
    dummy_input = train_dataset[0][0]
    input_d = dummy_input.numel()
    n_classes = len(train_dataset.classes)
    
    model = MLP(input_d=input_d, output_d=n_classes, **model_params)
    print(f"Loading MLP weights from {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    
    # 2. Extract
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    print(f"Extracting features (Layer: {layer_index})...")
    X_train, y_train = extract_features(model, train_loader, device, layer_index)
    X_test, y_test = extract_features(model, test_loader, device, layer_index)
    
    # 3. Preprocess
    print("Normalizing (L2) and Scaling (MinMax)...")
    normalizer = Normalizer(norm='l2')
    scaler = MinMaxScaler()
    
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def train_or_load_svm(X_train, y_train, output_dir=None, force_retrain=False, C=5, kernel='linear'):
    """
    Trains SVM or loads if exists.
    """
    if output_dir is None:
        output_dir = DEFAULT_CONFIG["output_dir"]
        
    os.makedirs(output_dir, exist_ok=True)
    
    # improved naming to avoid collisions
    model_filename = f"svm_{kernel}_C{C}.joblib"
    model_path = os.path.join(output_dir, model_filename)
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing SVM model from {model_path}...")
        clf = joblib.load(model_path)
    else:
        print(f"Training {kernel} SVM (C={C})...")
        if kernel == 'histogram_intersection':
             # Define HI kernel locally or import
             def histogram_intersection_kernel(X, Y):
                # Basic implementation or import from svm_part1 if available
                # For now using simple broadcasting
                K = np.zeros((X.shape[0], Y.shape[0]))
                for i in range(X.shape[0]):
                    K[i, :] = np.sum(np.minimum(X[i][None, :], Y), axis=1)
                return K
             clf = SVC(kernel=histogram_intersection_kernel, C=C, probability=True, random_state=42)
        else:
             clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
             
        clf.fit(X_train, y_train)
        print(f"Saving SVM model to {model_path}...")
        joblib.dump(clf, model_path)
        
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Predicts and calculates metrics.
    """
    print("Predicting...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    
    return metrics, y_pred, y_prob

    if show:
        plt.show()
    # else: plt.close() # Don't close if we want to return it active, or let caller handle it.
    # Actually, to be safe for notebooks, if show=False was requested, we might want to keep it open or just return it.
    # If we return it, the user can display it. 
    # Let's clean up: If show=False, we don't show. We return fig regardless.
    if not show:
        plt.close(fig) # We close it from pyplot manager to avoid double plotting in notebooks if they return it? 
        # Wait, if I close it, can I return it? Yes, the object exists.
        # But `plt.show()` clears it.
    
    return fig

def plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, show=True):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8)) # Create figure explicitly
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        print(f"Saved confusion_matrix.png to {output_dir}")
        
    if show:
        plt.show()
    
    # If we return the figure, we generally want it to be 'live' if we want to manipulate it, 
    # but for simple display return, closing from pyplot state is good practice to avoid memory leaks in loops.
    # However, if the user receives it and wants to `fig.show()`, it should work.
    
    return fig

def plot_roc_curve(y_true, y_prob, classes, output_dir=None, show=True):
    y_true_bin = pd.get_dummies(y_true).values
    n_classes = len(classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=(12, 10))
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
             
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = plt.cm.get_cmap('tab20', n_classes)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'Class {classes[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Multi-class ROC', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc="lower right", fontsize=12, ncol=2)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"))
        print(f"Saved roc_auc_curve.png to {output_dir}")

    if show:
        plt.show()
        
    return fig

def evaluate_mlp(model, dataloader, device, classes, output_dir=None):
    """
    Evaluates the MLP model directly.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    print("Evaluating MLP...")
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(dataloader, desc="MLP Eval"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    
    print("\n" + "="*30)
    print(" MLP RESULTS ")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    fig_cm = None
    fig_roc = None
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Plotting
        fig_cm = plot_confusion_matrix(y_true, y_pred, classes, output_dir=output_dir, show=False)
        fig_roc = plot_roc_curve(y_true, y_prob, classes, output_dir=output_dir, show=False)
        
        # Save Metrics
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        # Generate plots even if no output dir, just don't save.
        fig_cm = plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, show=False)
        fig_roc = plot_roc_curve(y_true, y_prob, classes, output_dir=None, show=False)
            
    return metrics, y_pred, y_prob, fig_cm, fig_roc


def run_pipeline(C=1, kernel='linear', layer_index='output', img_size=32, force_retrain=False,  
                 data_config_path=None, model_weights_path=None, output_dir=None):
    """
    Main orchestrator function with customizable parameters.
    """
    # Defaults
    if output_dir is None: output_dir = DEFAULT_CONFIG["output_dir"]
    if data_config_path is None: data_config_path = DEFAULT_CONFIG["data_config_path"]
    if model_weights_path is None: model_weights_path = DEFAULT_CONFIG["model_weights_path"]

    # 3. Prepare Data (Extract Features) - MLP Loaded here
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data Info
    train_ds, test_ds, model_params = load_datasets(data_config_path=data_config_path, img_size=img_size)
    
    # Load Model
    if model_weights_path is None:
        model_weights_path = DEFAULT_CONFIG["model_weights_path"]
        
    # Resolution for model path
    if not os.path.exists(model_weights_path):
         if os.path.exists(os.path.join("Week2", model_weights_path)):
             model_weights_path = os.path.join("Week2", model_weights_path)
             
    dummy_input = train_ds[0][0]
    input_d = dummy_input.numel()
    n_classes = len(train_ds.classes)
    
    model = MLP(input_d=input_d, output_d=n_classes, **model_params)
    print(f"Loading MLP weights from {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)

    # Evaluate MLP
    # Create dataloaders for evaluation
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    mlp_output_dir = os.path.join(output_dir, f"mlp_ImgSize_{img_size}")
    mlp_metrics, _, _, mlp_fig_cm, mlp_fig_roc = evaluate_mlp(model, test_loader, device, train_ds.classes, output_dir=mlp_output_dir)

    # Continue with SVM part
    # We need features now.
    print(f"Extracting features for SVM (Layer: {layer_index})...")
    # reusing the loaded model
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    
    X_train, y_train = extract_features(model, train_loader, device, layer_index)
    X_test, y_test = extract_features(model, test_loader, device, layer_index)
    
    # Preprocess for SVM
    print("Normalizing (L2) and Scaling (MinMax) for SVM...")
    normalizer = Normalizer(norm='l2')
    scaler = MinMaxScaler()
    
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Train or Load SVM
    clf = train_or_load_svm(
        X_train, y_train, 
        output_dir=output_dir, 
        force_retrain=force_retrain, 
        C=C, 
        kernel=kernel
    )
    
    # 4. Evaluate SVM
    metrics, y_pred, y_prob = evaluate_model(clf, X_test, y_test)
    
    print("\n" + "="*30)
    print(" SVM RESULTS ")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    # 5. Plot SVM
    plot_subdir = os.path.join(output_dir, f"{kernel}_C{C}_L{layer_index}")
    svm_fig_cm = plot_confusion_matrix(y_test, y_pred, train_ds.classes, output_dir=plot_subdir, show=False)
    svm_fig_roc = plot_roc_curve(y_test, y_prob, train_ds.classes, output_dir=plot_subdir, show=False)
    
    # Save SVM Metrics
    with open(os.path.join(plot_subdir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics, mlp_metrics, (mlp_fig_cm, mlp_fig_roc, svm_fig_cm, svm_fig_roc)


def load_resources(img_size=32, data_config_path=None, model_weights_path=None):
    """
    Loads Data and Model once.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Defaults
    if data_config_path is None: data_config_path = DEFAULT_CONFIG["data_config_path"]
    if model_weights_path is None: model_weights_path = DEFAULT_CONFIG["model_weights_path"]

    # 1. Load Data Info
    train_ds, test_ds, model_params = load_datasets(data_config_path=data_config_path, img_size=img_size)
    
    # Resolution for model path
    if not os.path.exists(model_weights_path):
         if os.path.exists(os.path.join("Week2", model_weights_path)):
             model_weights_path = os.path.join("Week2", model_weights_path)
             
    dummy_input = train_ds[0][0]
    input_d = dummy_input.numel()
    n_classes = len(train_ds.classes)
    
    model = MLP(input_d=input_d, output_d=n_classes, **model_params)
    print(f"Loading MLP weights from {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    
    return train_ds, test_ds, model

def run_mlp_eval(test_ds, model, output_dir=None, classes=None):
    """
    Wrapper for MLP evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if classes is None:
        classes = test_ds.classes
        
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    if output_dir is None:
        output_dir = DEFAULT_CONFIG["output_dir"]
    
    mlp_output_dir = os.path.join(output_dir, f"mlp_ImgSize_32") # hardcoded 32 for now based on context or use param
    
    # Call internal evaluate
    metrics, y_pred, y_prob, fig_cm, fig_roc = evaluate_mlp(model, test_loader, device, classes, output_dir=mlp_output_dir)
    return metrics, (fig_cm, fig_roc)

def run_svm_eval(train_ds, test_ds, model, C=1, kernel='linear', layer_index='output', output_dir=None, force_retrain=False):
    """
    Wrapper for SVM evaluation (Includes feature extraction).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None: 
        output_dir = DEFAULT_CONFIG["output_dir"]
        
    # We need features now.
    print(f"Extracting features for SVM (Layer: {layer_index})...")
    # reusing the loaded model
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    X_train, y_train = extract_features(model, train_loader, device, layer_index)
    X_test, y_test = extract_features(model, test_loader, device, layer_index)
    
    # Preprocess for SVM
    print("Normalizing (L2) and Scaling (MinMax) for SVM...")
    normalizer = Normalizer(norm='l2')
    scaler = MinMaxScaler()
    
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train or Load SVM
    clf = train_or_load_svm(
        X_train, y_train, 
        output_dir=output_dir, 
        force_retrain=force_retrain, 
        C=C, 
        kernel=kernel
    )
    
    # Evaluate SVM
    metrics, y_pred, y_prob = evaluate_model(clf, X_test, y_test)
    
    print("\n" + "="*30)
    print(" SVM RESULTS ")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    # Plot SVM
    plot_subdir = os.path.join(output_dir, f"{kernel}_C{C}_L{layer_index}")
    svm_fig_cm = plot_confusion_matrix(y_test, y_pred, train_ds.classes, output_dir=plot_subdir, show=False)
    svm_fig_roc = plot_roc_curve(y_test, y_prob, train_ds.classes, output_dir=plot_subdir, show=False)
    
    # Save SVM Metrics
    with open(os.path.join(plot_subdir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics, (svm_fig_cm, svm_fig_roc)

if __name__ == "__main__":
    run_pipeline()
