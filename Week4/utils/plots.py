import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import wandb
from typing import Dict, List
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Imports for Computational Graph and GradCAM
from torchviz import make_dot
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def save_and_log(fig, name, output_dir, wandb_log=True):
    """Saves a matplotlib figure locally and logs it to WandB."""
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path)
    if wandb_log:
        wandb.log({name: wandb.Image(path)})
    plt.close(fig)

def plot_metrics(
    train_metrics: Dict[str, List[float]],
    test_metrics: Dict[str, List[float]],
    metric_name: str,
    output_dir: str,
):
    """
    Plots a specific metric (e.g., 'loss' or 'accuracy') for both Train and Test.
    Replaces the previous 'plot_learning_curves' to allow separate, flexible plots.
    """
    epochs = range(1, len(train_metrics[metric_name]) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_metrics[metric_name], label=f"Train {metric_name.capitalize()}")
    if metric_name in test_metrics:
        ax.plot(epochs, test_metrics[metric_name], label=f"Test {metric_name.capitalize()}")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} Over Epochs")
    ax.legend()
    ax.grid(True)
    
    # Save using the helper (automatically handles filename and wandb logging)
    save_and_log(fig, f"{metric_name}_curve", output_dir)

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    save_and_log(fig, "confusion_matrix", output_dir)

def plot_roc_curve(y_true, y_probs, classes, output_dir):
    """
    Plots Multiclass ROC Curve using One-vs-Rest strategy.
    y_probs: Array of probabilities (softmax outputs), shape (N_samples, N_classes)
    """
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_bin.shape[1]
    y_probs = np.array(y_probs)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (One-vs-Rest)')
    ax.legend(loc="lower right")
    save_and_log(fig, "roc_curve", output_dir)

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, output_dir: str):
    """
    Generates and saves a visual representation of the model's computational graph.
    Requires 'torchviz' installed.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
    
    # render() creates a file with the given name (e.g., 'model_graph.png')
    # It creates a hidden 'model_graph' file + 'model_graph.png'
    graph_filename = os.path.join(output_dir, "model_graph")
    
    try:
        dot = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True)
        dot.format = 'png'
        dot.render(graph_filename)
        
        # Log to WandB (we must manually point to the .png file created by render)
        wandb.log({"computational_graph": wandb.Image(f"{graph_filename}.png")})
        print(f"Computational graph saved to {graph_filename}.png")
    except Exception as e:
        print(f"Failed to generate computational graph: {e}")

def plot_grad_cam_samples(model, dataset, device, output_dir, num_samples=3):
    """
    Generates GradCAM visualizations for random samples from the dataset.
    Requires 'pytorch_grad_cam' installed.
    """
    model.eval()
    
    # NOTE: This assumes 'model.backbone' exists, as defined in your BasicCNN.
    # If using other architectures, this target layer might need to change.
    try:
        target_layers = [model.backbone[-1]]
    except AttributeError:
        print("Warning: Model does not have a 'backbone' attribute. Skipping GradCAM.")
        return

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img_tensor, label = dataset[idx]
        
        # Prepare targets
        targets = [ClassifierOutputTarget(label)]
        
        # Generate CAM
        # Note: 'model' here must implement 'extract_feature_maps' or be compatible with GradCAM
        # If using standard GradCAM wrapper, you might need to wrap 'model' inside the main script.
        # This function assumes 'model' is the wrapper itself or has the method.
        # Ideally, pass the GradCAM object or ensure model compatibility.
        
        # For this snippet, assuming 'model' has the .extract_grad_cam method 
        # OR you wrap it here if you prefer using the library directly on the standard model:
        from pytorch_grad_cam import GradCAM
        
        # We need to reconstruct the CAM object here or pass it in. 
        # To keep it simple, let's instantiate it locally:
        cam = GradCAM(model=model, target_layers=target_layers)

        input_tensor = img_tensor.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Un-normalize for visualization
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # Save manually (using plt.imsave for simple image arrays)
        save_path = os.path.join(output_dir, f"gradcam_sample_{i}.png")
        plt.imsave(save_path, visualization)

        # Log to WandB
        wandb.log({f"grad_cam_{i}": wandb.Image(save_path, caption=f"Class: {dataset.classes[label]}")})

