import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from itertools import cycle
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, MinMaxScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc)
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader

# Import your custom modules
from models import MLP 
from utils import get_patches

# ==========================================
# 1. HELPER CLASSES (BoW Logic)
# ==========================================

class DeepBoWEngine:
    """
    Handles the complexity of Extracting Patches -> Fitting K-Means -> Building Histograms.
    """
    def __init__(self, model, device, layer_id, codebook_size, patch_size, stride):
        self.model = model
        self.device = device
        self.layer_id = layer_id
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.stride = stride
        self.kmeans = None

    def _get_descriptors_from_loader(self, dataloader, desc_text):
        """Extracts raw patch descriptors from the MLP."""
        self.model.eval()
        all_descs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(dataloader, desc=desc_text, leave=False):
                inputs = inputs.to(self.device)
                
                # 1. Patchify (B, 3, H, W) -> (B*N, 3, P, P)
                patches = get_patches(inputs, self.patch_size, self.stride)
                
                # 2. Extract Features
                if self.layer_id == "concat":
                    feats_list = [self.model.get_features(patches, layer_index=l) for l in [0, 1, 2, 3]]
                    features = torch.cat(feats_list, dim=1)
                elif str(self.layer_id) == "output":
                    features = self.model.get_features(patches, layer_index="output")
                else:
                    features = self.model.get_features(patches, layer_index=int(self.layer_id))
                
                # 3. Reshape (B*N, D) -> (B, N, D)
                n_patches = features.shape[0] // inputs.shape[0]
                features = features.view(inputs.shape[0], n_patches, -1)
                
                feats_np = features.cpu().numpy()
                for i in range(len(labels)):
                    all_descs.append(feats_np[i]) 
                    all_labels.append(labels[i].item())
                    
        return all_descs, np.array(all_labels)

    def prepare_data(self, train_loader, test_loader):
        """Full pipeline: Extract -> Fit Codebook -> Build Histograms."""
        print(f"   > Extracting Descriptors for Layer {self.layer_id}...")
        train_descs, y_train = self._get_descriptors_from_loader(train_loader, "Extract Train")
        test_descs, y_test = self._get_descriptors_from_loader(test_loader, "Extract Val")
        
        # 2. Fit Codebook
        print(f"   > Fitting K-Means (k={self.codebook_size})...")
        all_feats_stack = np.vstack(train_descs)
        # Downsample for speed if needed
        if len(all_feats_stack) > 100000:
            idx = np.random.choice(len(all_feats_stack), 100000, replace=False)
            all_feats_stack = all_feats_stack[idx]
            
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.codebook_size, 
            batch_size=1000, 
            n_init='auto', 
            random_state=42
        ).fit(all_feats_stack)
        
        # 3. Build Histograms
        print(f"   > Building Histograms...")
        X_train = self._build_hists(train_descs)
        X_test = self._build_hists(test_descs)
        
        return X_train, y_train, X_test, y_test
    
    def _build_hists(self, descs_list):
        hists = []
        for d in descs_list:
            words = self.kmeans.predict(d)
            h, _ = np.histogram(words, bins=range(self.codebook_size + 1))
            hists.append(h)
        return np.array(hists, dtype=np.float32)

# ==========================================
# 2. PLOTTING UTILS (UPDATED)
# ==========================================

def save_qualitative_examples(dataset, y_true, y_pred, classes, target_class_idx, 
                              output_path, title_prefix=""):
    """
    Saves a grid of CORRECT and INCORRECT examples for a specific class.
    """
    # Find indices for the target class
    class_indices = np.where(y_true == target_class_idx)[0]
    
    # Split into correct and incorrect
    correct_indices = [i for i in class_indices if y_pred[i] == y_true[i]]
    incorrect_indices = [i for i in class_indices if y_pred[i] != y_true[i]]
    
    # Select up to 5 examples of each
    n_show = 5
    show_correct = correct_indices[:n_show]
    show_incorrect = incorrect_indices[:n_show]
    
    total_imgs = len(show_correct) + len(show_incorrect)
    if total_imgs == 0:
        print(f"No examples found for class {target_class_idx}")
        return

    # Create Plot
    fig, axes = plt.subplots(2, n_show, figsize=(15, 6))
    plt.suptitle(f"{title_prefix} - Analysis for Class: {classes[target_class_idx]}", fontsize=16)
    
    # Plot Correct (Top Row)
    for i, idx in enumerate(show_correct):
        img_path = dataset.samples[idx][0]
        img = mpimg.imread(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"CORRECT\nPred: {classes[y_pred[idx]]}", color='green', fontweight='bold')
        axes[0, i].axis('off')
        
    # Plot Incorrect (Bottom Row)
    for i, idx in enumerate(show_incorrect):
        img_path = dataset.samples[idx][0]
        img = mpimg.imread(img_path)
        axes[1, i].imshow(img)
        # Show what it was confused with
        wrong_label = classes[y_pred[idx]]
        axes[1, i].set_title(f"WRONG\nPred: {wrong_label}", color='red', fontweight='bold')
        axes[1, i].axis('off')

    # Hide empty subplots if fewer than 5 examples
    for i in range(len(show_correct), n_show): axes[0, i].axis('off')
    for i in range(len(show_incorrect), n_show): axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved qualitative plot to {output_path}")

def plot_heatmap(df, value_col, title, output_path):
    pivot = df.pivot(index="Layer", columns="C", values=value_col)
    
    order = [0, 1, 2, 3, "concat", "output"]
    existing = [o for o in order if o in pivot.index]
    pivot = pivot.reindex(existing)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", linewidths=.5)
    plt.title(title, fontsize=16)
    plt.ylabel("Layer", fontsize=12)
    plt.xlabel("C Value", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_multiclass_roc(y_true, y_score, classes, title, output_path):
    """
    Generates a detailed multi-class ROC plot similar to the reference image.
    Includes Micro, Macro, and per-class curves.
    """
    # Binarize the output
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Macro Average (Dotted Navy)
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)

    # Micro Average (Dotted Pink)
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)

    # Individual Classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'gray'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                 label='Class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, ncol=2) # Adjust ncol if many classes
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Best BoW Config)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def main(args):
    # --- 1. Load Configs ---
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not args.dry_run:
        wandb.init(project=cfg.get("wandb_project", "week2-bovw"), 
                   name=cfg.get("experiment_name", "bovw_lr_run"),
                   config=cfg)

    # --- 2. Data & Model Loading ---
    print(f"Loading Data from {cfg['dataset_path']}...")
    transform = F.Compose([
        F.ToImage(), F.ToDtype(torch.float32, scale=True), 
        F.Resize((cfg['img_size'], cfg['img_size']))
    ])
    
    train_ds = ImageFolder(os.path.join(cfg['dataset_path'], "train"), transform=transform)
    test_ds = ImageFolder(os.path.join(cfg['dataset_path'], "val"), transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Loading MLP from {cfg['model_path']}...")
    input_d = 3 * cfg['patch_size'] * cfg['patch_size'] 
    model = MLP(input_d=input_d, output_d=len(train_ds.classes), hidden_layers=[1024, 512, 256, 128])
    model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
    model.to(device)

    # --- 3. End-to-End Evaluation (Baseline) ---
    print("\n--- Phase 1: End-to-End Baseline Evaluation ---")
    model.eval()
    e2e_probs_all = []
    e2e_labels_all = []
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="E2E Inference"):
            inputs = inputs.to(device)
            # Patchify & Forward
            patches = get_patches(inputs, cfg['patch_size'], cfg['stride'])
            logits = model(patches)
            probs = torch.softmax(logits, dim=1)
            
            # Aggregate (Mean Pooling)
            n_patches = probs.shape[0] // inputs.shape[0]
            probs = probs.view(inputs.shape[0], n_patches, -1)
            img_probs = torch.mean(probs, dim=1)
            
            e2e_probs_all.append(img_probs.cpu().numpy())
            e2e_labels_all.append(labels.numpy())
            
    e2e_probs_final = np.concatenate(e2e_probs_all)
    e2e_labels_final = np.concatenate(e2e_labels_all)
    e2e_acc = accuracy_score(e2e_labels_final, np.argmax(e2e_probs_final, axis=1))
    print(f"End-to-End Accuracy: {e2e_acc:.4f}")

    # --- 4. BoVW Grid Search Loop ---
    print("\n--- Phase 2: BoVW Grid Search ---")
    
    results_list = []
    best_val_acc = 0.0
    best_model_artifacts = {} 
    
    for layer in cfg['layers_to_eval']:
        print(f"\n[Processing Layer: {layer}]")
        
        # A. Feature Extraction
        engine = DeepBoWEngine(model, device, layer, cfg['codebook_size'], cfg['patch_size'], cfg['stride'])
        X_train, y_train, X_test, y_test = engine.prepare_data(train_loader, test_loader)
        
        # B. Preprocessing
        norm = Normalizer(norm='l2')
        scaler = MinMaxScaler()
        X_train_proc = scaler.fit_transform(norm.transform(X_train))
        X_test_proc = scaler.transform(norm.transform(X_test))
        
        # C. Inner Loop: Logistic Regression
        for C_val in cfg['c_values']:
            clf = LogisticRegression(C=C_val, max_iter=2000, random_state=42)
            clf.fit(X_train_proc, y_train)
            
            y_val_pred = clf.predict(X_test_proc)
            y_val_prob = clf.predict_proba(X_test_proc)
            
            train_acc = accuracy_score(y_train, clf.predict(X_train_proc))
            val_acc = accuracy_score(y_test, y_val_pred)
            
            print(f"  C={C_val} | Train: {train_acc:.3f}, Val: {val_acc:.3f}")
            
            results_list.append({
                "Layer": layer,
                "C": C_val,
                "Train Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })
            
            if not args.dry_run:
                wandb.log({"layer": layer, "C": C_val, "val_acc": val_acc})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_artifacts = {
                    "y_true": y_test,
                    "y_pred": y_val_pred,
                    "y_prob": y_val_prob,
                    "config": f"L{layer}_C{C_val}"
                }

    # --- 5. Save & Plot Results ---
    print("\n--- Phase 3: Saving Results & Generating Plots ---")
    output_dir = os.path.join("results", cfg['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)
    
    # 1. Heatmaps
    plot_heatmap(df, "Train Accuracy", "Heatmap: Train Accuracy", os.path.join(output_dir, "heatmap_train.png"))
    plot_heatmap(df, "Validation Accuracy", "Heatmap: Validation Accuracy", os.path.join(output_dir, "heatmap_val.png"))
                 
    # 2. Confusion Matrix (Best BoVW)
    plot_confusion_matrix(best_model_artifacts["y_true"], 
                          best_model_artifacts["y_pred"], 
                          range(len(train_ds.classes)),  #   train_ds.classes,
                          os.path.join(output_dir, "confusion_matrix.png"))
                          
    # 3. DETAILED ROC CURVES (The Request)
    print("Generating Detailed ROC Curves...")
    
    # A. End-to-End ROC
    plot_multiclass_roc(
        e2e_labels_final, 
        e2e_probs_final, 
        train_ds.classes,
        title="End-To-End ROC",
        output_path=os.path.join(output_dir, "roc_curve_e2e.png")
    )
    
    # B. BoVW ROC (Best Config)
    plot_multiclass_roc(
        best_model_artifacts["y_true"],
        best_model_artifacts["y_prob"],
        train_ds.classes,
        title=f"BoVW ROC ({best_model_artifacts['config']})",
        output_path=os.path.join(output_dir, "roc_curve_bovw.png")
    )

    print("\n--- Phase 4: Qualitative Visualization ---")
    
    classes_to_viz = [6, 2, 9, 5] # The ones we discussed
    
    # 1. Visualize for End-to-End Model
    e2e_preds = np.argmax(e2e_probs_final, axis=1)
    
    for cls_idx in classes_to_viz:
        if cls_idx < len(train_ds.classes):
            save_qualitative_examples(
                dataset=test_ds, 
                y_true=e2e_labels_final, 
                y_pred=e2e_preds, 
                classes=train_ds.classes, 
                target_class_idx=cls_idx, 
                output_path=os.path.join(output_dir, f"qualitative_E2E_class_{cls_idx}.png"),
                title_prefix="End-to-End MLP"
            )

    # 2. Visualize for Best BoVW Model
    best_preds = best_model_artifacts["y_pred"]
    best_true = best_model_artifacts["y_true"]
    
    for cls_idx in classes_to_viz:
        if cls_idx < len(train_ds.classes):
            save_qualitative_examples(
                dataset=test_ds, 
                y_true=best_true, 
                y_pred=best_preds, 
                classes=train_ds.classes, 
                target_class_idx=cls_idx, 
                output_path=os.path.join(output_dir, f"qualitative_BoVW_class_{cls_idx}.png"),
                title_prefix="Deep BoVW"
            )
    
    print(f"Global Best BoVW Config: {best_model_artifacts['config']} (Acc: {best_val_acc:.4f})")
    print(f"End-to-End Baseline Acc: {e2e_acc:.4f}")
    print(f"All outputs saved to: {output_dir}")
    
    if not args.dry_run:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_bow.json", help="Path to JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Run fast debug mode")
    args = parser.parse_args()
    main(args)