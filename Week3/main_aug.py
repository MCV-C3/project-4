import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn as nn
import torch.optim as optim

import os
import tqdm
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import WraperModel

# Train function


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, exp_name: str, output_dir: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name],
             label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name],
             label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = os.path.join(output_dir, f"{exp_name}_{metric_name}.png")
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory


def get_train_transform(
        augmentation_transform: Optional[str | list],
        combination_mode: Optional[str],
        img_size: int
    ):
    """Build the training transform pipeline with optional data augmentation.

    This function creates a torchvision transform pipeline that converts images
    to tensors, optionally applies data augmentation, resizes them, and normalizes
    them for model training.

    Args:
        augmentation_transform (Optional[str | list]): Specification of which
            augmentations to apply. Can be one of:
            - None or "none": no data augmentation.
            - "all": apply all available augmentations sequentially.
            - str: the name of a single augmentation.
            - list or tuple of str: names of multiple augmentations to apply.
        combination_mode (Optional[str]): Specification of the way multiple 
            transforms are going to be combined. Can be one of:
            - None or "none" or "all": apply all the transforms for an image sample.
            - "choice": randomly chooses only one transform for an image sample.
        img_size (int): Target height and width of the output image after resizing.

    Returns:
        torchvision.transforms.Compose: A composed transform that can be passed
        to a PyTorch Dataset for training.

    Raises:
        ValueError: If an unknown augmentation name is provided.
        TypeError: If `augmentation_transform` has an invalid type.
    """

    augmentation_transform_map = {
        "crop": F.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        "horizontal_flip": F.RandomHorizontalFlip(p=.5),
        "gaussian_noise": F.RandomApply([F.GaussianNoise()], p=0.5),
        "rotation": F.RandomApply([F.RandomRotation(degrees=[-45, 45])], p=0.5),
        "photometric_distort": F.RandomPhotometricDistort(),
        "perspective": F.RandomPerspective(),
    }

    # decide which augmentation ops to apply
    aug_ops = []

    if augmentation_transform is None or augmentation_transform == "none":
        aug_ops = []

    elif augmentation_transform == "all":
        aug_ops = list(augmentation_transform_map.values())

    elif isinstance(augmentation_transform, str):
        if augmentation_transform not in augmentation_transform_map:
            raise ValueError(f"Unknown augmentation '{augmentation_transform}'. "
                             f"Valid: {list(augmentation_transform_map.keys())} + ['all', None]")
        aug_ops = [augmentation_transform_map[augmentation_transform]]

    elif isinstance(augmentation_transform, (list, tuple)):
        unknown = [
            k for k in augmentation_transform if k not in augmentation_transform_map]
        if unknown:
            raise ValueError(f"Unknown augmentations {unknown}. "
                             f"Valid: {list(augmentation_transform_map.keys())}")
        aug_ops = [augmentation_transform_map[k]
                   for k in augmentation_transform]

    else:
        raise TypeError(
            "augmentation_transform must be None, 'all', str, list, or tuple.")

    # compose pipeline
    if combination_mode == "choice":
        return F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomChoice(aug_ops),
            F.Resize(size=(img_size, img_size)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif combination_mode is None or combination_mode == "none" or combination_mode == "all":
        return F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            *aug_ops,
            F.Resize(size=(img_size, img_size)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise TypeError(
            "combination_mode must be None, 'none', or 'all'.")


def plot_confusion_matrix(model, dataloader, device, classes, plot_name, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation='vertical', ax=ax)
    plt.title(f'Confusion Matrix: {plot_name}')

    # Save locally
    save_path = os.path.join(output_dir, f"{plot_name}_cm.png")
    plt.savefig(save_path)

    # Log to WandB
    wandb.log({"confusion_matrix": wandb.Image(save_path)})

    plt.close()


def plot_grad_cam_samples(model, dataset, device, plot_name, output_dir, num_samples=3):
    model.eval()

    # Ensure gradients are enabled for Grad-CAM even though we are in eval mode
    target_layers = [model.backbone[-1]]

    for i in range(num_samples):
        # Pick random images from the test set
        idx = np.random.randint(0, len(dataset))
        img_tensor, label = dataset[idx]

        # Targets for the specific ground truth class
        targets = [ClassifierOutputTarget(label)]

        # Generate CAM
        grayscale_cam = model.extract_grad_cam(
            img_tensor.unsqueeze(0).to(device), target_layers, targets)

        # Convert tensor back to image for visualization
        # Reverse normalization for visualization
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])
                  ) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        save_path = os.path.join(output_dir, f"{plot_name}_gradcam_{i}.png")
        plt.imsave(save_path, visualization)

        # Log to WandB
        wandb.log({f"grad_cam_{i}": wandb.Image(
            save_path, caption=f"Class: {dataset.classes[label]}")})


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set up shared parameters
    DATASET_PATH = os.path.expanduser(config["dataset_path"])
    BATCH_SIZE = config.get("batch_size", 256)
    EPOCHS = config.get("epochs", 20)
    LR = config.get("lr", 0.001)
    NUM_WORKERS = config.get("num_workers", 4)
    IMG_SIZE = config.get("img_size", 224)
    
    # Fine-tuning parameters
    UNFREEZE_DEPTH = config.get("unfreeze_depth", 0)
    HEAD_CONFIG = config.get("head_config", None)
    LEVEL = config.get("level", 4)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if str(device) == 'cpu':
        print("WARNING: Training on CPU. This might be slow.")

    # Loop over variations ----
    variations = config.get("variations", [])
    if not variations:
        # Backwards compatible: run single experiment
        variations = [{"name": str(config.get("data_augmentation", "none")),
                       "data_augmentation": config.get("data_augmentation", "none"),
                       "combination_mode": config.get("combination_mode", "none")}]

    for var in variations:
        DATA_AUGMENTATION = var.get("data_augmentation", "none")
        COMBINATION_MODE = var.get("combination_mode", "none")
        EXPERIMENT_NAME = var.get("name", str(DATA_AUGMENTATION))

        # Create Output Directory (per variation)
        OUTPUT_DIR = os.path.join("data_augmentation_experiments", EXPERIMENT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\n=== Running variation: {EXPERIMENT_NAME} | aug={DATA_AUGMENTATION} | comb={COMBINATION_MODE} ===")
        print(f"Output directory: {OUTPUT_DIR}")

        # Initialize WandB (per variation)
        wandb.init(
            project=config.get("wandb_project", "week3"),
            entity=config.get("wandb_entity", None),
            name=EXPERIMENT_NAME,
            config={**config, **var},   # log the variation too
            dir=OUTPUT_DIR,
            mode=config.get("wandb_mode", "online"),
            reinit=True,               # IMPORTANT: allow multiple init() in one process
        )

        # Data Augmentation for Training
        train_transformation = get_train_transform(DATA_AUGMENTATION, COMBINATION_MODE, IMG_SIZE)

        # Data Augmentation for Testing -> just clean resize and normalize
        test_transformation = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize(size=(IMG_SIZE, IMG_SIZE)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loading data from {DATASET_PATH}")

        data_train = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transformation)
        data_test = ImageFolder(os.path.join(DATASET_PATH, "test"), transform=test_transformation)

        train_loader = DataLoader(
            data_train, batch_size=BATCH_SIZE,
            pin_memory=True, shuffle=True, num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            data_test, batch_size=BATCH_SIZE,
            pin_memory=True, shuffle=False, num_workers=NUM_WORKERS
        )

        model = WraperModel(num_classes=8, truncation_level=LEVEL).to(device)
        
        if HEAD_CONFIG is not None:
            model.modify_classifier_head(
                hidden_dims=HEAD_CONFIG.get("hidden_dims", None),
                activation=HEAD_CONFIG.get("activation", "relu"),
            )
        
        model.fine_tuning(unfreeze_blocks=UNFREEZE_DEPTH)

        model.summary()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        wandb.watch(model, log="all")

        for epoch in tqdm.tqdm(range(EPOCHS), desc=f"TRAINING ({EXPERIMENT_NAME})"):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = test(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"[{EXPERIMENT_NAME}] Epoch {epoch + 1}/{EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            })

        plot_name = EXPERIMENT_NAME  # keep it simple and consistent

        # Save Model
        save_path = os.path.join(OUTPUT_DIR, f"{plot_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Plot results
        print("Generating Loss and Accuracy plots...")
        plot_metrics({"loss": train_losses, "accuracy": train_accuracies},
                     {"loss": test_losses, "accuracy": test_accuracies},
                     "loss", plot_name, OUTPUT_DIR)
        plot_metrics({"loss": train_losses, "accuracy": train_accuracies},
                     {"loss": test_losses, "accuracy": test_accuracies},
                     "accuracy", plot_name, OUTPUT_DIR)

        print("Generating Confusion Matrix...")
        plot_confusion_matrix(model, test_loader, device, data_test.classes, plot_name, OUTPUT_DIR)

        print("Generating Grad-CAM samples...")
        plot_grad_cam_samples(model, data_test, device, plot_name, OUTPUT_DIR)

        # Save results to JSON
        results = {
            "experiment_name": EXPERIMENT_NAME,
            "data_augmentation": DATA_AUGMENTATION,
            "best_test_accuracy": max(test_accuracies),
            "best_test_loss": min(test_losses),
            "final_test_accuracy": test_accuracies[-1],
            "final_test_loss": test_losses[-1],
            "train_accuracy": train_accuracies[-1],
            "train_loss": train_losses[-1],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "img_size": IMG_SIZE,
            "unfreeze_depth": UNFREEZE_DEPTH,
            "level": LEVEL,
            "head_config": HEAD_CONFIG,
            "trainable_params": model.get_trainable_parameters(),
            "total_params": model.get_total_parameters(),
        }

        with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {os.path.join(OUTPUT_DIR, 'results.json')}")

        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3 Main")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON configuration file")
    args = parser.parse_args()

    main(args)
