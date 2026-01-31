import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, label_binarize, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm
import wandb
from models import MLP

# ==========================================
# KERNEL FUNCTIONS
# ==========================================

def histogram_intersection_kernel(X, Y):
    """
    Compute the histogram intersection kernel between X and Y using batching to avoid OOM.
    K(x, y) = sum(min(xi, yi))
    """
    N_x = X.shape[0]
    K = np.zeros((N_x, Y.shape[0]))
    
    # Batch size for rows of X. 
    # Logic: (Batch, N_y, D) float64 tensor.
    # 10 * 8700 * 2000 * 8 bytes ~= 1.4 GB. 
    # With 8 workers, this is 11 GB total peak. well within 48 GB.
    # To be safer let's use 5.
    BATCH_SIZE = 5
    
    for i in range(0, N_x, BATCH_SIZE):
        x_batch = X[i:i+BATCH_SIZE] # (B, D)
        # Broadcast: (B, 1, D) vs (1, Ny, D) -> (B, Ny, D)
        # Sum over D -> (B, Ny)
        res = np.sum(np.minimum(x_batch[:, None, :], Y[None, :, :]), axis=-1)
        K[i:i+BATCH_SIZE, :] = res
        
    return K

# ==========================================
# FEATURE EXTRACTION
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
            # Get features from specific layer
            # model.get_features returns shape (B, D)
            feats = model.get_features(inputs, layer_index=layer_index)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

# ==========================================
# MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 SVM Part 1: Config-Driven Feature Extraction + Grid Search")
    parser.add_argument("--svm_config", type=str, required=True, help="Path to SVM config file")
    parser.add_argument("--dry-run", action="store_true", help="Run with limited data for debugging")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs for cross-validation (default: 4)")
    
    args = parser.parse_args()

    # 1. Load SVM Configuration
    print(f"Loading SVM config from {args.svm_config}")
    with open(args.svm_config, 'r') as f:
        svm_config = json.load(f)
        
    train_config_path = svm_config["train_config_path"]
    model_path = svm_config["model_path"]
    experiment_name = svm_config["experiment_name"]
    param_grid = svm_config["param_grid"] # List of explicit configs
    
    # "layers_to_eval": [0, 1, 2, "concat"]
    # If not present, fallback to "layer_index" for backward compatibility or single layer behavior
    if "layers_to_eval" in svm_config:
        layers_to_eval = svm_config["layers_to_eval"]
    else:
        layers_to_eval = [svm_config.get("layer_index", 0)]
    
    # 2. Load Train Config (for Data/Model params)
    print(f"Loading Train config from {train_config_path}")
    if not os.path.exists(train_config_path):
        # Initial check failed, try prepending Week2/ if we are in root
        if os.path.exists(os.path.join("Week2", train_config_path)):
            train_config_path = os.path.join("Week2", train_config_path)
            print(f"Adjusted train_config_path to {train_config_path}")
        else:
            print(f"Warning: {train_config_path} not found.")

    with open(train_config_path, 'r') as f:
        train_config = json.load(f)

    # Use parameters from train config or override if needed
    IMG_SIZE = train_config.get("img_size", 224)
    # Ensure datasets use train_config path
    DATASET_PATH = train_config["dataset_path"]
    # Model architecture params
    MODEL_PARAMS = train_config["model_params"]
    
    # Batch size for extraction can be anything, defaulting to 256
    BATCH_SIZE = train_config.get("batch_size", 256)

    # 3. Setup WandB
    if not args.dry_run:
        wandb.init(
            project=svm_config.get("wandb_project", "week2-svm"),
            entity=svm_config.get("wandb_entity", None),
            mode=svm_config.get("wandb_mode", "online"),
            name=experiment_name,
            config=svm_config # Log the full SVM config
        )
    else:
        print("Dry Run: WandB disabled.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Load Data
    transform = get_transforms(IMG_SIZE)
    
    train_dataset = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=transform)
    test_dataset = ImageFolder(os.path.join(DATASET_PATH, "val"), transform=transform)
    
    if args.dry_run:
        print("Dry Run: Using subset of data")
        indices = np.random.choice(len(train_dataset), 100, replace=False)
        train_dataset = Subset(train_dataset, indices)
        indices_test = np.random.choice(len(test_dataset), 50, replace=False)
        test_dataset = Subset(test_dataset, indices_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 5. Load Model
    dummy_input = train_dataset[0][0] # (3, H, W)
    input_d = dummy_input.numel()
    output_d = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(train_dataset.dataset.classes)
    
    model = MLP(input_d=input_d, output_d=output_d, **MODEL_PARAMS)
    
    if not os.path.exists(model_path):
        # Fallback check relative to script execution?
        if os.path.exists(os.path.join("Week2", model_path)):
             model_path = os.path.join("Week2", model_path)
             print(f"Adjusted model_path to {model_path}")
        else:
             raise FileNotFoundError(f"Model weights not found at {model_path}")

    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # ==========================
    # Global SWEEP Logic
    # ==========================
    
    global_results_list = []
    global_best_score = -1.0
    global_best_params = None
    global_best_layer = None
    global_best_config_name = None
    global_best_clf = None # To store fitted model if needed, or we refit at end
    
    # Store features in memory to avoid re-extracting if we loop differently? 
    # But for "concat" we need specific extraction. 
    # Let's loop over layers.
    
    for layer_id in layers_to_eval:
        print(f"\n{'='*40}")
        print(f"Processing Layer: {layer_id}")
        print(f"{'='*40}")
        
        # 6. Extract Features for this Layer Config
        if layer_id == "concat":
            # Extract ALL layers defined in model (0 to len(model.layers)-1) + Output
            n_layers = len(model.layers)
            print(f"Extracting and Concatenating features from all {n_layers} hidden layers + Output...")
            
            X_train_list = []
            X_test_list = []
            y_train = None
            y_test = None
            
            # Hidden Layers
            for i in range(n_layers):
                print(f"  > Extracting Hidden Layer {i}...")
                xt, yt = extract_features(model, train_loader, device, layer_index=i)
                xv, yv = extract_features(model, test_loader, device, layer_index=i)
                X_train_list.append(xt)
                X_test_list.append(xv)
                if i == 0:
                    y_train = yt
                    y_test = yv
            
            # Output Layer
            print(f"  > Extracting Output Layer...")
            xt, yt = extract_features(model, train_loader, device, layer_index="output")
            xv, yv = extract_features(model, test_loader, device, layer_index="output")
            X_train_list.append(xt)
            X_test_list.append(xv)

            # Concatenate
            X_train = np.hstack(X_train_list)
            X_test = np.hstack(X_test_list)
        
        else:
            # Single Layer
            # Check if integer or "output"
            if str(layer_id).lower() == "output":
                 extract_idx = "output"
                 print(f"Extracting features from Output Layer...")
            else:
                 extract_idx = int(layer_id)
                 print(f"Extracting features from layer index {extract_idx}...")
            
            X_train, y_train = extract_features(model, train_loader, device, layer_index=extract_idx)
            X_test, y_test = extract_features(model, test_loader, device, layer_index=extract_idx)

        print(f"Train Features Shape: {X_train.shape}")
        
        # Scaling (Per Layer Choice)
        # Normalization (L2)
        print("Normalizing features (L2)...")
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

        # Scaling (MinMax)
        print("Scaling features (MinMaxScaler)...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        # Fix: handle case where minmax scaler might complain about range if test has outliers, 
        # but usually fine.
        X_test = scaler.transform(X_test)
        
        
        # 7. Run Parameter Search for this Layer
        layer_best_score = -1.0
        n_features = X_train.shape[1]
        
        for params in tqdm.tqdm(param_grid, desc=f"Grid Search (Layer {layer_id})"):
            base_config_name = params.get("name", "unnamed")
            # Create a unique config name including layer
            full_config_name = f"L{layer_id}_{base_config_name}"
            
            svc_args = {k: v for k, v in params.items() if k != "name"}
            
            if svc_args.get("kernel") == "histogram_intersection":
                 svc_args["kernel"] = histogram_intersection_kernel
                 
            try:
                clf = SVC(probability=False, **svc_args)
                
                # Cross Validation (return_train_score=True)
                # n_jobs=-1 to use all CPUs (CAUTION: Can cause OOM)
                # Using args.n_jobs instead
                cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=args.n_jobs, return_train_score=True)
                
                mean_val_score = cv_results['test_score'].mean()
                std_val_score = cv_results['test_score'].std()
                mean_train_score = cv_results['train_score'].mean()
                std_train_score = cv_results['train_score'].std()
                mean_fit_time = cv_results['fit_time'].mean()
                
                log_params = params.copy()
                log_params['layer'] = str(layer_id)
                # log_params['samples_used'] = len(y_resampled)
                
                result_entry = {
                    "config_name": full_config_name,
                    "layer": layer_id,
                    "base_config": base_config_name,
                    "params": log_params,
                    "cv_mean_test_accuracy": mean_val_score,
                    "cv_std_test_accuracy": std_val_score,
                    "cv_mean_train_accuracy": mean_train_score,
                    "cv_std_train_accuracy": std_train_score,
                    "mean_execution_time": mean_fit_time,
                    "n_features": n_features
                }
                global_results_list.append(result_entry)
                
                print(f"[{full_config_name}] Val Acc: {mean_val_score:.4f} | Train Acc: {mean_train_score:.4f} | Feats: {n_features}")
                
                if not args.dry_run:
                    wandb.log({
                        "sweep/layer": str(layer_id),
                        "sweep/n_features": n_features,
                        "sweep/config_name": full_config_name,
                        "sweep/val_accuracy": mean_val_score,
                        "sweep/train_accuracy": mean_train_score,
                        "sweep/fit_time": mean_fit_time,
                        **{f"sweep/{k}": v for k, v in log_params.items() if isinstance(v, (int, float, str))}
                    })

                # Check for Global Best
                if mean_val_score > global_best_score:
                    global_best_score = mean_val_score
                    global_best_params = svc_args
                    global_best_layer = layer_id
                    global_best_config_name = full_config_name
                    
                    # Temporarily save features/labels of best layer to retrain later?
                    # Or just refit everything at end?
                    # Since we overwrite X_train in loop, we need to handle "Retraining best model".
                    # Option: Save X_train_best = X_train.copy()
                    global_X_train_best = X_train.copy()
                    global_X_test_best = X_test.copy()
                    global_y_train_best = y_train.copy()
                    global_y_test_best = y_test.copy()
                    
            except Exception as e:
                print(f"Error config {full_config_name}: {e}")

    # End of Loop
    print(f"\n{'='*40}")
    print(f"Global Best Config: {global_best_config_name}")
    print(f"Layer: {global_best_layer}")
    print(f"CV Acc: {global_best_score:.4f}")
    print(f"{'='*40}")

    if global_best_params is None:
        print("No successful experiments.")
        exit(1)

    # 8. Train Best Model (using cached Best Features)
    print("Retraining best model on full training set (using best layer features)...")
    final_clf = SVC(probability=True, **global_best_params)
    final_clf.fit(global_X_train_best, global_y_train_best)
    
    y_pred = final_clf.predict(global_X_test_best)
    # y_prob = final_clf.predict_proba(global_X_test_best)

    # Calculate detailed metrics
    test_acc = accuracy_score(global_y_test_best, y_pred)
    precision = precision_score(global_y_test_best, y_pred, average='macro', zero_division=0)
    recall = recall_score(global_y_test_best, y_pred, average='macro', zero_division=0)
    f1 = f1_score(global_y_test_best, y_pred, average='macro', zero_division=0)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if not args.dry_run:
        wandb.log({
            "best_config": global_best_config_name,
            "best_layer": str(global_best_layer),
            "best_cv_accuracy": global_best_score,
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })

    # 9. Save Results
    output_dir = os.path.join("svm_experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Validation Plots
    cm = confusion_matrix(global_y_test_best, y_pred)
    plt.figure(figsize=(10, 8))
    classes = sorted(list(set(global_y_test_best)))
    # Reuse valid class names logic if possible, or stringify
    class_names = [str(c) for c in classes]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({global_best_config_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Save JSON/CSV
    final_results = {
        "global_best_config_name": global_best_config_name,
        "global_best_layer": global_best_layer,
        "global_best_cv_accuracy": global_best_score,
        "test_accuracy": test_acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "all_results": global_results_list,
        "svm_config_used": args.svm_config
    }
    
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
        
    df_results = pd.DataFrame(global_results_list)
    csv_path = os.path.join(output_dir, "results.csv")
    df_results.to_csv(csv_path, index=False)
        
    print(f"Results saved to {output_path} and {csv_path}")
    
    if not args.dry_run:
        wandb.finish()
