import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from itertools import cycle

# Import the model
from models import WraperModel

# Constants
HYP_OPT_DIR = "experiments/hyp_opt"
CONFIG_PATH = "configs/hyp_opt/hyp_config.yaml"
DATASET_PATH = "/ghome/group04/mcv/datasets/C3/2425/MIT_small_train_1"
OUTPUT_FOLDER = "experiments/analyze_best_model"
IMG_SIZE = 224

def get_best_model_path(root_folder=HYP_OPT_DIR):
    """Finds the best model based on existing results.json files."""
    max_acc = -1.0
    best_folder = None
    best_config_name = None

    if not os.path.isdir(root_folder):
        print(f"Directory not found: {root_folder}")
        return None, None

    # Sort to ensure reproducibility if multiple have same score
    subdirs = sorted(os.listdir(root_folder))
    
    for subdir in subdirs:
        dir_path = os.path.join(root_folder, subdir)
        if not os.path.isdir(dir_path):
            continue
        
        results_path = os.path.join(dir_path, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                    acc = data.get("best_test_accuracy", 0)
                    if acc > max_acc:
                        max_acc = acc
                        best_folder = dir_path
                        best_config_name = subdir
            except Exception as e:
                print(f"Error reading {results_path}: {e}")
                pass
    
    return best_folder, best_config_name

def load_hyp_config(yaml_path=CONFIG_PATH):
    """Loads fixed configuration from the YAML file manually."""
    parsed_config = {}
    current_key = None
    
    with open(yaml_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Simple parser for the specific structure:
            # key:
            #   value: val
            
            # Check for top level keys (indentation 2 spaces usually)
            # Actually let's just regex or string split
            # The file structure is consistent.
            
            if ": " in line:
                part1, part2 = line.split(": ", 1)
            elif line.endswith(":"):
                part1 = line[:-1]
                part2 = ""
            else:
                continue
                
            # Heuristic: keys that are parameters
            if part1 in ["num_workers", "unfreeze_depth", "level", "head_config", 
                         "data_augmentation", "dropout", "dropout_method", "norm", 
                         "l1_reg", "l2_reg"]:
                current_key = part1
            elif current_key and part1 == "value":
                # Parse value
                val_str = part2.strip()
                # Handle types
                if val_str.startswith("{") or val_str.startswith("["):
                    # JSON-like
                    try:
                        parsed_config[current_key] = json.loads(val_str.replace("'", '"'))
                    except:
                        parsed_config[current_key] = val_str
                elif val_str.isdigit():
                    parsed_config[current_key] = int(val_str)
                elif val_str.replace('.','',1).isdigit():
                    parsed_config[current_key] = float(val_str)
                elif val_str.lower() in ["true", "false"]:
                    parsed_config[current_key] = val_str.lower() == "true"
                else:
                    # String, remove quotes if any
                    parsed_config[current_key] = val_str.strip('"').strip("'")
                current_key = None # Reset
                
    return parsed_config

def parse_folder_config(folder_name):
    """Parses hyperparams from the folder name."""
    # Expected format: Batch_{}_Epochs_{}_LR_{}_Opt_{}_Momentum_{}_Scheduler_{}
    # Example: Batch_10_Epochs_50_LR_0.0001_Opt_SGD_Momentum_0.0_Scheduler_StepLR
    
    config = {}
    parts = folder_name.split('_')
    
    try:
        if 'Batch' in parts:
            idx = parts.index('Batch') + 1
            config['batch_size'] = int(parts[idx])
        
        if 'Epochs' in parts:
            idx = parts.index('Epochs') + 1
            config['epochs'] = int(parts[idx])
            
        if 'LR' in parts:
            idx = parts.index('LR') + 1
            config['lr'] = float(parts[idx])
            
        if 'Opt' in parts:
            idx = parts.index('Opt') + 1
            config['optimizer'] = parts[idx]
            
        if 'Momentum' in parts:
            idx = parts.index('Momentum') + 1
            config['momentum'] = float(parts[idx])
            
        if 'Scheduler' in parts:
            idx = parts.index('Scheduler') + 1
            config['scheduler'] = parts[idx]
            
    except Exception as e:
        print(f"Error parsing folder name {folder_name}: {e}")
        
    return config

def load_best_model(best_folder, best_config_name, device):
    """Loads the best model."""
    
    # 1. Load configurations
    base_config = load_hyp_config()
    folder_config = parse_folder_config(best_config_name)
    
    # Merge configs (folder config overwrites base if collision, though keys should be disjoint)
    merged_config = {**base_config, **folder_config}
    
    print("Reconstructed Configuration:")
    for k, v in merged_config.items():
        print(f"  {k}: {v}")
    
    # 2. Initialize Model
    num_classes = 8
    level = merged_config.get("level", 3)
    
    model = WraperModel(num_classes=num_classes, truncation_level=level)
    
    # Normalize Backbone
    norm_type = merged_config.get("norm", "batch")
    if norm_type != "batch":
        print(f"Applying {norm_type} normalization to backbone...")
        from main_hyp import replace_normalization
        replace_normalization(model.backbone, norm_type)
        
    # Dropout (Feature Extractor)
    dropout_rate = merged_config.get("dropout", 0.0)
    dropout_method = merged_config.get("dropout_method", "classifier")
    
    if dropout_method == "feature_extractor" and dropout_rate > 0:
        print(f"Applying Dropout ({dropout_rate}) to Feature Extractor")
        from main_hyp import inject_dropout_after_activation
        # Assuming ReLU for now as standard
        inject_dropout_after_activation(model.backbone, dropout_rate, torch.nn.ReLU)

    # Classifier Head
    head_dropout = dropout_rate if dropout_method == "classifier" else 0.0
    head_config = merged_config.get("head_config", {"hidden_dims": [512], "activation": "relu"})
    
    if head_config:
        hidden_dims = head_config.get("hidden_dims", [512])
        activation = head_config.get("activation", "relu")
        print(f"Building Head: {hidden_dims}, {activation}, Drop: {head_dropout}, Norm: {norm_type}")
        model.modify_classifier_head(
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=head_dropout,
            normalization=norm_type
        )
        
    # Unfreeze
    unfreeze_depth = merged_config.get("unfreeze_depth", 1)
    model.fine_tuning(unfreeze_blocks=unfreeze_depth)
    
    model = model.to(device)
    
    # 3. Load Weights
    weight_path = os.path.join(best_folder, f"{best_config_name}.pth")
    if not os.path.exists(weight_path):
        print(f"Weight file not found at {weight_path}. Checking for any .pth file...")
        files = [f for f in os.listdir(best_folder) if f.endswith('.pth')]
        if len(files) > 0:
            weight_path = os.path.join(best_folder, files[0])
            print(f"Found {weight_path}")
        else:
            raise FileNotFoundError("Model weights not found!")

    print(f"Loading weights from {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    return model, merged_config

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(y_true, y_score, classes, output_dir):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

def analyze_qualitative(model, dataset, y_true, y_pred, output_dir, device):
    correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    
    # Visualize top 5 correct and incorrect
    classes = dataset.classes
    
    def visualize_samples(indices, title, filename, num_samples=5):
        if not indices:
             return
        sample_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        fig, axes = plt.subplots(1, len(sample_indices), figsize=(15, 4))
        if len(sample_indices) == 1: axes = [axes]
        
        for ax, idx in zip(axes, sample_indices):
            img, label = dataset[idx]
            prediction = y_pred[idx]
            
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.set_title(f"True: {classes[label]}\nPred: {classes[prediction]}")
            ax.axis('off')
            
        plt.suptitle(title)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        return sample_indices

    print("Generating Qualitative Analysis...")
    correct_samples = visualize_samples(correct_indices, "Correctly Classified Examples", "correct_examples.png")
    incorrect_samples = visualize_samples(incorrect_indices, "Incorrectly Classified Examples", "incorrect_examples.png")
    
    return correct_samples, incorrect_samples

def generate_activations_and_gradcam(model, dataset, indices, output_dir, device, prefix="sample"):
    if indices is None: return
    
    model.eval()
    
    # Target Layer for GradCAM (Last ResNet block)
    # ResNeXt 101: dynamic selection of the last convolutional layer
    target_layer_candidate = model.backbone[-1]
    
    if hasattr(target_layer_candidate, 'conv3'): 
         # Likely a BottleNeck
         target_layers = [target_layer_candidate]
    else:
        # Fallback: use the last item in backbone
        target_layers = [model.backbone[-1]]

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.to(device)
        
        # 1. GradCAM
        targets = [ClassifierOutputTarget(label)]
        cam = model.extract_grad_cam(img_tensor.unsqueeze(0), target_layers, targets)
        
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        visualization = show_cam_on_image(img_np, cam, use_rgb=True)
        plt.imsave(os.path.join(output_dir, f"{prefix}_{idx}_gradcam.png"), visualization)
        
        # 2. Activation Maps (Feature Maps) - Evolution across layers
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                # Only save if it's a spatial feature map (4D: B, C, H, W)
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    activations[name] = output.detach()
            return hook
            
        # Register hooks for each layer in the backbone
        layer_names_map = {
            0: "Conv 2D",
            1: "Conv 2D BN",
            2: "Conv 2D ReLU",
            3: "Conv 2D MaxPool",
            4: "Layer 1",
            5: "Layer 2",
            6: "Layer 3",
            7: "Layer 4"
        }
        
        print(f"Hooking {len(model.backbone)} layers in backbone...")
        for i, layer in enumerate(model.backbone):
            layer_type = type(layer).__name__
            name = layer_names_map.get(i, f"Layer {i}")
            hooks.append(layer.register_forward_hook(get_activation(f"{i}: {name}")))
            
        with torch.no_grad():
            model(img_tensor.unsqueeze(0))
            
        print(f"Captured activations for: {list(activations.keys())}")
            
        # Remove hooks
        for h in hooks:
            h.remove()
        
        if activations:
            num_layers = len(activations)
            fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 3, 3))
            if num_layers == 1: axes = [axes]
            
            for ax, (name, act) in zip(axes, activations.items()):
                # act is [1, C, H, W]
                mean_act = torch.mean(act[0], dim=0).cpu().numpy()
                
                # Normalize for better visualization
                mean_act = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
                
                ax.imshow(mean_act, cmap='viridis')
                ax.set_title(name)
                ax.axis('off')
                
            plt.suptitle(f"Activation Evolution - Class {dataset.classes[label]}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_{idx}_activation_evolution.png"))
            plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Find and Load Best Model
    print("Searching for best model...")
    best_folder, best_config_name = get_best_model_path()
    
    if not best_folder:
        print("No best model found.")
        return
        
    print(f"Best Model Found: {best_config_name}")
    print(f"Path: {best_folder}")
    
    model, config = load_best_model(best_folder, best_config_name, device)
    
    # 2. Prepare Data
    transform_test = transforms.Compose([
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Loading test data from {DATASET_PATH}")
    test_dataset = ImageFolder(os.path.join(DATASET_PATH, "test"), transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 32), shuffle=False, num_workers=4)
    
    # 3. Predict
    print("Running Predictions...")
    all_preds = []
    all_labels = []
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 4. Analysis output
    output_dir = os.path.join(OUTPUT_FOLDER, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis to {output_dir}")
    
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("Classification Report:")
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Plots
    plot_confusion_matrix(all_labels, all_preds, test_dataset.classes, output_dir)
    plot_roc_curve(all_labels, all_scores, test_dataset.classes, output_dir)
    
    # Qualitative
    correct_idx, incorrect_idx = analyze_qualitative(model, test_dataset, all_labels, all_preds, output_dir, device)
    
    # Activations & GradCAM
    print("Generating Visualizations for Correct Examples...")
    generate_activations_and_gradcam(model, test_dataset, correct_idx, output_dir, device, prefix="correct")
    print("Generating Visualizations for Incorrect Examples...")
    generate_activations_and_gradcam(model, test_dataset, incorrect_idx, output_dir, device, prefix="incorrect")
    
    print("Analysis Complete!")

if __name__ == "__main__":
    main()
