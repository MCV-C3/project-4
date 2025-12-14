from typing import *
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import MLP
from utils import get_patches
import torchvision.transforms.v2 as F
from torchviz import make_dot
import tqdm
import argparse
import json
import os
import wandb


class EarlyStopping:
    """Early stopping"""

    def __init__(self, patience=7, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train(
    model, dataloader, criterion, optimizer, device, patch_enabled=False, patch_size=112
):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Store original batch size to calculate repetition factor later
        original_batch_size = inputs.size(0)

        if patch_enabled:
            # Patches shape: (4096, 3, 64, 64)
            inputs = get_patches(inputs, patch_size=patch_size, stride=patch_size)

            # Expand Labels
            num_patches_per_img = inputs.size(0) // original_batch_size
            labels = labels.repeat_interleave(num_patches_per_img)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability (especially important with patches)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(
    model,
    dataloader,
    criterion,
    device,
    patch_enabled=False,
    patch_size=112,
    aggregation="mean",
):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Store original batch size to calculate repetition factor later
            original_batch_size = inputs.size(0)

            if patch_enabled:
                # Extract patches
                inputs = get_patches(inputs, patch_size=patch_size, stride=patch_size)

                # Forward pass on all patches
                outputs = model(inputs)

                # Group patches by original image
                num_patches = inputs.size(0) // original_batch_size
                outputs_grouped = outputs.view(original_batch_size, num_patches, -1)

                # AVERAGE Pooling: Average the logits/probabilities across all patches
                if aggregation == "mean":
                    outputs = outputs_grouped.mean(dim=1)
                elif aggregation == "max":
                    outputs = outputs_grouped.max(dim=1)
                elif aggregation == "median":
                    outputs = outputs_grouped.median(dim=1).values
                else:
                    # VOTING??
                    outputs = outputs_grouped.mean(dim=1)
            else:
                # Standard forward pass without patches
                outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * original_batch_size  # Use original batch size
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += original_batch_size  # Use original batch size

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_metrics(
    train_metrics: Dict,
    test_metrics: Dict,
    metric_name: str,
    exp_name: str,
    output_dir: str,
):
    """
    Plots and saves metrics for training and testing.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f"Train {metric_name.capitalize()}")
    plt.plot(test_metrics[metric_name], label=f"Test {metric_name.capitalize()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = os.path.join(output_dir, f"{exp_name}_{metric_name}.png")
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory


def plot_computational_graph(
    model: torch.nn.Module, input_size: tuple, filename: str, output_dir: str
):
    """
    Generates and saves a plot of the computational graph of the model.
    """
    model.eval()  # Set the model to evaluation mode

    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)

    # Create a graph from the model
    try:
        # torchviz appends .png automatically, but we want to control the path
        # It's easier to verify where it saves. make_dot(...).render(path) saves to path.png
        save_path = os.path.join(output_dir, filename)
        graph = make_dot(
            model(dummy_input), params=dict(model.named_parameters()), show_attrs=True
        ).render(save_path, format="png")
        print(f"Computational graph saved as {save_path}.png")
    except OSError as e:
        print(f"Could not render computational graph (Graphviz might be missing): {e}")


def get_optimizer(model, optimizer_name, lr, **kwargs):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, **kwargs
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")


def get_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **kwargs)
    elif scheduler_name == "linearlr":
        return optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    elif scheduler_name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 MLP Experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run a single epoch for testing"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Set up parameters
    EXPERIMENT_NAME = config.get("experiment_name", "experiment")
    DATASET_PATH = os.path.expanduser(config["dataset_path"])
    BATCH_SIZE = config.get("batch_size", 256)
    EPOCHS = config.get("epochs", 20)
    LR = config.get("lr", 0.001)
    NUM_WORKERS = config.get("num_workers", 4)
    MODEL_PARAMS = config.get("model_params", {"hidden_layers": [300]})
    IMG_SIZE = config.get("img_size", 224)  # New param
    DATA_AUGMENTATION = config.get("data_augmentation", True)

    # Patch parameters
    PATCH_ENABLE = config.get("patch_enable", False)
    PATCH_SIZE = config.get("patch_size", 112)  # 112, 56, 28
    AGGREGATION = config.get("aggregation", "mean")

    # Early stopping
    USE_EARLY_STOPPING = config.get("early_stopping", True)
    EARLY_STOPPING_PATIENCE = config.get("early_stopping_patience", 10)

    # Normalization
    USE_NORMALIZATION = config.get("use_normalization", True)

    # Optimizer & Scheduler
    OPTIMIZER_NAME = config.get("optimizer_name", "adam")
    OPTIMIZER_PARAMS = config.get("optimizer_params", {})
    SCHEDULER_NAME = config.get("scheduler_name", None)
    SCHEDULER_PARAMS = config.get("scheduler_params", {})

    # Create Output Directory
    OUTPUT_DIR = os.path.join("experiments", EXPERIMENT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Patches enabled: {PATCH_ENABLE}")
    if PATCH_ENABLE:
        print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
        print(f"Aggregation: {AGGREGATION}")
        num_patches_per_side = IMG_SIZE // PATCH_SIZE
        total_patches = num_patches_per_side**2
        print(f"Patches per image: {total_patches}")
    print(
        f"Early stopping: {USE_EARLY_STOPPING} (patience={EARLY_STOPPING_PATIENCE if USE_EARLY_STOPPING else 'N/A'})"
    )

    # Initialize WandB
    wandb.init(
        project=config.get("wandb_project", "week2"),
        entity=config.get("wandb_entity", None),
        name=EXPERIMENT_NAME,
        config=config,
        dir=OUTPUT_DIR,  # Save wandb metadata here too
        mode=config.get("wandb_mode", "online"),
    )

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if str(device) == "cpu":
        print("WARNING: Training on CPU. This might be slow.")

    # Data Augmentation for Training
    train_transforms_list = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(IMG_SIZE, IMG_SIZE)),  # Resize first
    ]

    if DATA_AUGMENTATION:
        train_transforms_list.extend(
            [
                F.RandomHorizontalFlip(p=0.5),
                F.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )

    if USE_NORMALIZATION:
        train_transforms_list.append(
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    train_transform = F.Compose(train_transforms_list)

    # Validation transform
    val_transforms_list = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(IMG_SIZE, IMG_SIZE)),
    ]

    if USE_NORMALIZATION:
        val_transforms_list.append(
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    val_transform = F.Compose(val_transforms_list)

    print(f"Loading data from {DATASET_PATH}")
    # Using 'train' and 'val' subfolders
    # Original 'train' folder is loaded as FULL training set
    data_train = ImageFolder(
        os.path.join(DATASET_PATH, "train"), transform=train_transform
    )

    # Validation folder used as Test/Reporting set
    data_val = ImageFolder(os.path.join(DATASET_PATH, "val"), transform=val_transform)

    print(f"Full Training Set: {len(data_train)} images")
    print(f"Test/Validation Set: {len(data_val)} images")
    print(f"Number of classes: {len(data_train.classes)}")

    train_loader = DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        data_val,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Get dimensions
    C, H, W = data_train[0][0].numpy().shape
    num_classes = len(data_train.classes)

    if PATCH_ENABLE:
        input_dim = C * PATCH_SIZE * PATCH_SIZE
        print(
            f"Creating model for patches: input_dim={input_dim} (C={C}, patch={PATCH_SIZE}x{PATCH_SIZE})"
        )
    else:
        input_dim = C * H * W
        print(
            f"Creating model for full images: input_dim={input_dim} (C={C}, H={H}, W={W})"
        )

    model = MLP(input_d=input_dim, output_d=num_classes, **MODEL_PARAMS)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Plot computational graph with correct input size
    try:
        plot_computational_graph(
            model,
            input_size=(1, input_dim),
            filename="computational_graph",
            output_dir=OUTPUT_DIR,
        )
    except Exception as e:
        print(f"Failed to plot graph: {e}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, OPTIMIZER_NAME, LR, **OPTIMIZER_PARAMS)
    scheduler = get_scheduler(optimizer, SCHEDULER_NAME, **SCHEDULER_PARAMS)

    # Early stopping
    early_stopping = None
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, mode="max", min_delta=0.001
        )
        print(f"Early stopping enabled (patience={EARLY_STOPPING_PATIENCE})\n")

    # Training metrics
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_epoch = 0

    wandb.watch(model, log="all")

    for epoch in range(EPOCHS):
        if args.dry_run and epoch > 0:
            print("Dry run mode: Stopping after 1 epoch")
            break

        # Training
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device, PATCH_ENABLE, PATCH_SIZE
        )

        # Validation
        val_loss, val_accuracy = test(
            model, val_loader, criterion, device, PATCH_ENABLE, PATCH_SIZE, AGGREGATION
        )

        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log current LR
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Track best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1

            # Save best model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "config": config,
            }
            best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(checkpoint, best_model_path)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Best: {best_val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
                "best_val_accuracy": best_val_acc,
                "overfitting_gap": train_accuracy - val_accuracy,
            }
        )

        # Early stopping check
        if early_stopping:
            if early_stopping(val_accuracy, epoch + 1):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                break

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(
        f"Overfitting gap (train-val): {train_accuracies[-1] - val_accuracies[-1]:.4f}"
    )

    # Plot metrics
    plot_name = os.path.basename(EXPERIMENT_NAME)
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": val_losses, "accuracy": val_accuracies},
        "loss",
        plot_name,
        OUTPUT_DIR,
    )
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": val_losses, "accuracy": val_accuracies},
        "accuracy",
        plot_name,
        OUTPUT_DIR,
    )

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, f"{plot_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")

    # Save detailed results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "best_val_loss": min(val_losses),
        "final_val_accuracy": val_accuracies[-1],
        "final_val_loss": val_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "final_train_loss": train_losses[-1],
        "overfitting_gap": train_accuracies[-1] - val_accuracies[-1],
        "epochs_trained": len(train_losses),
        "total_epochs": EPOCHS,
        "stopped_early": len(train_losses) < EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "img_size": IMG_SIZE,
        "patch_enable": PATCH_ENABLE,
        "model_params": MODEL_PARAMS,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    if PATCH_ENABLE:
        results["patch_size"] = PATCH_SIZE
        results["aggregation"] = AGGREGATION
        results["patches_per_image"] = (IMG_SIZE // PATCH_SIZE) ** 2

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    wandb.finish()
