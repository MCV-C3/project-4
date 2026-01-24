
import models
from main_distill import build_student_model, get_transforms
import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"

# Import from local codebase
sys.path.append('.')


def inverse_normalize(tensor):
    """Reverses the ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np


def visualize_samples(dataset, indices, y_pred, classes, title, filepath):
    """Plots a grid of samples with True/Pred labels."""
    if indices is None or len(indices) == 0:
        return None

    num_samples = min(5, len(indices))
    selected_indices = np.random.choice(indices, num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    if num_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected_indices):
        img_tensor, label = dataset[idx]
        pred_label = y_pred[idx]

        img_np = inverse_normalize(img_tensor)

        ax.imshow(img_np)
        ax.set_title(f"True: {classes[label]}\nPred: {classes[pred_label]}")
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {filepath}")
    return selected_indices


def get_target_layers(model, model_name):
    """Determines target layers for GradCAM based on architecture."""
    # Logic adapted for MobileNet and ShuffleNet
    if "MobileNet" in model_name:
        # Target the last block of features
        return [model.features[-1]]
    elif "ShuffleNet" in model_name:
        # Target conv5 (last conv before pooling)
        return [model.conv5]
    else:
        # Fallback to last leaf module that is a Conv2d
        print(f"Unknown model structure for {model_name}, trying heuristic...")
        layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        return [layers[-1]] if layers else []


def register_activation_hooks(model, model_name):
    """Registers hooks to capture feature maps from key layers."""
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                activations[name] = output.detach()
        return hook

    if "MobileNet" in model_name:
        # MobileNet: features is a Sequential list of blocks
        # We'll sample a few evenly spaced blocks
        num_blocks = len(model.features)
        indices = np.linspace(0, num_blocks - 1, num=5, dtype=int)

        for i in indices:
            layer = model.features[i]
            # Try to get a nice name
            name = f"Block_{i}"
            hooks.append(layer.register_forward_hook(get_hook(name)))

    elif "ShuffleNet" in model_name:
        # ShuffleNetMini: conv1, stage2, stage3, stage4, conv5
        layers_to_hook = [
            ('Conv1', model.conv1),
            ('Stage2', model.stage2),
            ('Stage3', model.stage3),
            ('Stage4', model.stage4),
            ('Conv5', model.conv5)
        ]
        for name, layer in layers_to_hook:
            hooks.append(layer.register_forward_hook(get_hook(name)))

    return activations, hooks


def generate_visualizations(model, dataset, indices, output_dir, device, model_name, prefix):
    """Generates GradCAM and Activation Evolution plots for specific samples."""
    if indices is None or len(indices) == 0:
        return

    target_layers = get_target_layers(model, model_name)
    if not target_layers:
        print("Could not find target layers for GradCAM.")
        return

    # Process each sample
    for idx in indices:
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # 1. GradCAM
        targets = [ClassifierOutputTarget(label)]
        try:
            with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor, targets=targets)[0, :]

            img_np = inverse_normalize(img_tensor)
            visualization = show_cam_on_image(
                img_np, grayscale_cam, use_rgb=True)

            save_path = os.path.join(
                output_dir, f"{prefix}_sample_{idx}_gradcam.png")
            plt.imsave(save_path, visualization)

        except Exception as e:
            print(f"GradCAM failed for {model_name}: {e}")

        # 2. Activation Evolution
        activations, hooks = register_activation_hooks(model, model_name)

        # Forward pass to trigger hooks
        with torch.no_grad():
            model(input_tensor)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Plot activations
        if activations:
            num_layers = len(activations)
            fig, axes = plt.subplots(
                1, num_layers, figsize=(num_layers * 3, 3))
            if num_layers == 1:
                axes = [axes]

            sorted_keys = sorted(activations.keys(), key=lambda x: int(
                x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
            # If not sortable by simple digit, just use default insertion order (Python dicts preserve order)
            keys_to_use = list(activations.keys())

            for ax, name in zip(axes, keys_to_use):
                act = activations[name]  # [1, C, H, W]
                # Mean across channels
                mean_act = torch.mean(act[0], dim=0).cpu().numpy()

                # Normalize [0,1]
                mean_act = (mean_act - mean_act.min()) / \
                    (mean_act.max() - mean_act.min() + 1e-8)

                ax.imshow(mean_act, cmap='viridis')
                ax.set_title(name)
                ax.axis('off')

            true_label = dataset.classes[label]
            plt.suptitle(f"Activation Evolution - {true_label} ({prefix})")
            plt.tight_layout()
            save_path = os.path.join(
                output_dir, f"{prefix}_sample_{idx}_activations.png")
            plt.savefig(save_path)
            plt.close()


def main():
    base_dir = "../results/best_hypopt_models"
    target_dirs = [
        "MobileNet_Distill_Batch_16_LR_0.1_Opt_SGD_Sched_CosineAnnealingLR_r6zen7t9",
        "ShuffleNet_Distill_Batch_16_LR_0.01_Opt_SGD_Sched_CosineAnnealingLR_vqbx5ztp"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_dir in target_dirs:
        full_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(full_path):
            continue

        print(f"\nAnalyzing {model_dir}...")

        # Output directory for analysis
        analysis_dir = os.path.join(full_path, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Load Config
        config_path = os.path.join(full_path, "experiment_results.json")
        with open(config_path, 'r') as f:
            cfg = json.load(f).get('configuration', {})

        # Load Data
        img_size = cfg['data']['img_size']
        test_dir = cfg['data']['test_dir']
        batch_size = cfg['data']['batch_size']
        num_workers = cfg['data']['num_workers']

        test_transforms = get_transforms(img_size, is_train=False)
        test_dataset = ImageFolder(test_dir, transform=test_transforms)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        classes = test_dataset.classes

        # Load Model
        print(f"Building {cfg['student_model']['name']}...")
        student = build_student_model(cfg, len(classes))
        student.to(device)

        weights_path = os.path.join(full_path, "best_model.pth")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(full_path, "best_student_model.pth")

        if not os.path.exists(weights_path):
            print("Weights not found, skipping.")
            continue

        student.load_state_dict(torch.load(weights_path, map_location=device))
        student.eval()

        # Run Inference
        print("Running Inference...")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = student(inputs)
                _, preds = outputs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Identify Correct/Incorrect
        correct_indices = np.where(all_preds == all_labels)[0]
        incorrect_indices = np.where(all_preds != all_labels)[0]

        print(
            f"Found {len(correct_indices)} correct and {len(incorrect_indices)} incorrect samples.")

        # Qualitative Plots
        print("Generating Qualitative Plots...")
        best_samples = visualize_samples(test_dataset, correct_indices, all_preds, classes,
                                         "Correctly Classified Examples",
                                         os.path.join(analysis_dir, "correct_examples.png"))

        worst_samples = visualize_samples(test_dataset, incorrect_indices, all_preds, classes,
                                          "Incorrectly Classified Examples",
                                          os.path.join(analysis_dir, "incorrect_examples.png"))

        # GradCAM and Activations
        if best_samples is not None:
            print("Generating Visualizations for Correct Samples...")
            generate_visualizations(student, test_dataset, best_samples,
                                    analysis_dir, device, cfg['student_model']['name'], "correct")

        if worst_samples is not None:
            print("Generating Visualizations for Incorrect Samples...")
            generate_visualizations(student, test_dataset, worst_samples,
                                    analysis_dir, device, cfg['student_model']['name'], "incorrect")

        print(f"Analysis complete for {model_dir}")


if __name__ == "__main__":
    main()
