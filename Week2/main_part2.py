from typing import *
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import MLP
from utils import get_patches, InMemoryDataset, aggregate_predictions
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import argparse
import json
import os
import wandb

# Train function
def train(model, dataloader, criterion, optimizer, device, patch_enabled=False, patch_size=112, aggregation="mean", evaluation_mode="patch"):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        original_batch_size = inputs.size(0)

        if patch_enabled:
            # Patches shape: (B * N_patches, C, H, W)
            inputs = get_patches(inputs, patch_size=patch_size, stride=patch_size)

            # Expand Labels 
            num_patches_per_img = inputs.size(0) // original_batch_size
            labels_expanded = labels.repeat_interleave(num_patches_per_img)
        else:
            labels_expanded = labels

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels_expanded)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        if patch_enabled and evaluation_mode == "aggregated":
            # We aggregate the patch predictions to see how well the model classifies IMAGES during training
            aggregated_outputs = aggregate_predictions(outputs, original_batch_size, aggregation)
            
            # Recalculate loss for reporting (Image vs Image Label)
            # Note: This is just for the plot, the gradients came from patch loss above
            loss_report = criterion(aggregated_outputs, labels) 
            
            train_loss += loss_report.item() * original_batch_size
            _, predicted = aggregated_outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) # Total Images
            
        else:
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels_expanded).sum().item()
            total += labels_expanded.size(0) # Total Patches

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, dataloader, criterion, device, patch_enabled, patch_size, aggregation, evaluation_mode="patch"):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            original_batch_size = inputs.size(0)

            if patch_enabled:
                inputs = get_patches(inputs, patch_size=patch_size, stride=patch_size)
                num_patches_per_img = inputs.size(0) // original_batch_size
                labels_expanded = labels.repeat_interleave(num_patches_per_img)
            else:
                labels_expanded = labels
            
            # Forward pass
            outputs = model(inputs)

            if patch_enabled and evaluation_mode == "aggregated":
                aggregated_outputs = aggregate_predictions(outputs, original_batch_size, aggregation)
                
                loss = criterion(aggregated_outputs, labels) # Compare Aggregated vs Image Label
                
                test_loss += loss.item() * original_batch_size
                _, predicted = aggregated_outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0) # Total Images
                
            else:
                loss = criterion(outputs, labels_expanded) # Compare Patch vs Patch Label
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += (predicted == labels_expanded).sum().item()
                total += labels_expanded.size(0) # Total Patches

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, exp_name: str, output_dir: str):
    """
    Plots and saves metrics for training and testing.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
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


def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str, output_dir: str):
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
        graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(save_path, format="png")
        print(f"Computational graph saved as {save_path}.png")
    except OSError as e:
        print(f"Could not render computational graph (Graphviz might be missing): {e}")

def get_optimizer(model, optimizer_name, lr, **kwargs):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, **kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

def get_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    if scheduler_name.lower() == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name.lower() == 'linearlr':
        return optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 MLP Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for testing")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set up parameters
    EXPERIMENT_NAME = config.get("experiment_name", "experiment")
    DATASET_PATH = os.path.expanduser(config["dataset_path"])
    BATCH_SIZE = config.get("batch_size", 256)
    EPOCHS = config.get("epochs", 20)
    LR = config.get("lr", 0.001)
    NUM_WORKERS = config.get("num_workers", 4)
    MODEL_PARAMS = config.get("model_params", {"hidden_layers": [300]})
    IMG_SIZE = config.get("img_size", 224) # New param
    DATA_AUGMENTATION = config.get("data_augmentation", True)
    PATCH_ENABLE = config.get("patch_enable", False)
    PATCH_SIZE = config.get("patch_size", 112) # 112, 56, 28
    AGGREGATION = config.get("aggregation", "mean")
    EVALUATION_MODE = config.get("evaluation_mode", "patch")

    # Optimizer & Scheduler config
    OPTIMIZER_NAME = config.get("optimizer_name", "adam")
    OPTIMIZER_PARAMS = config.get("optimizer_params", {})
    SCHEDULER_NAME = config.get("scheduler_name", None)
    SCHEDULER_PARAMS = config.get("scheduler_params", {})
    
    # Create Output Directory
    OUTPUT_DIR = os.path.join("experiments", EXPERIMENT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Initialize WandB
    wandb.init(
        project=config.get("wandb_project", "week2"),
        entity=config.get("wandb_entity", None),
        name=EXPERIMENT_NAME,
        config=config,
        dir=OUTPUT_DIR, # Save wandb metadata here too
        mode=config.get("wandb_mode", "online")
    )

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if str(device) == 'cpu':
        print("WARNING: Training on CPU. This might be slow.")

    # Data Augmentation for Training
    train_transforms_list = [
        # F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(IMG_SIZE, IMG_SIZE)), # Resize first
    ]

    if DATA_AUGMENTATION:
        train_transforms_list.extend([
            F.RandomHorizontalFlip(p=0.5),
            F.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    # Normalization always applied
    # train_transforms_list.append(F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = F.Compose(train_transforms_list)

    # Standard Transform for Validation/Test
    val_transform = F.Compose([
        # F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(IMG_SIZE, IMG_SIZE)),
        # F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Loading data from {DATASET_PATH}")
    # Using 'train' and 'val' subfolders
    # Original 'train' folder is loaded as FULL training set
    # data_train = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transform)
    data_train = InMemoryDataset(ImageFolder(os.path.join(DATASET_PATH, "train")), device=device, transform=train_transform)
    
    # Validation folder used as Test/Reporting set
    # data_val = ImageFolder(os.path.join(DATASET_PATH, "val"), transform=val_transform)
    data_val = InMemoryDataset(ImageFolder(os.path.join(DATASET_PATH, "val")), device=device, transform=val_transform)
    
    print(f"Full Training Set: {len(data_train)} images")
    print(f"Test/Validation Set: {len(data_val)} images")
    
    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, num_workers=0)
    val_loader = DataLoader(data_val, batch_size=BATCH_SIZE, pin_memory=False, shuffle=False, num_workers=0)
    # We won't use test_loader in the training loop loop typically, unless we want to see test perfromance too.
    # But methodology says: Tune on Val, Report on Test.
    # For now, let's just log Validation performance.

    C, H, W = data_train[0][0].shape
    num_classes = len(data_train.classes)

    if PATCH_ENABLE:
        model = MLP(input_d=C*PATCH_SIZE*PATCH_SIZE, output_d=num_classes, **MODEL_PARAMS)
    else:    
        model = MLP(input_d=C*H*W, output_d=num_classes, **MODEL_PARAMS)
    
    model = model.to(device)
    
    # Plotting graph after moving to device and setting eval to avoid BN issues with single sample
    try:
        plot_computational_graph(model, input_size=(1, C*H*W), filename="computational_graph", output_dir=OUTPUT_DIR) 
    except Exception as e:
        print(f"Failed to plot graph: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, OPTIMIZER_NAME, LR, **OPTIMIZER_PARAMS)
    scheduler = get_scheduler(optimizer, SCHEDULER_NAME, **SCHEDULER_PARAMS)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    wandb.watch(model, log="all")

    for epoch in tqdm.tqdm(range(EPOCHS), desc="TRAINING THE MODEL"):
        if args.dry_run:
            print("Dry run: Running one epoch with limited batches...")
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, PATCH_ENABLE, PATCH_SIZE, AGGREGATION, EVALUATION_MODE)
            val_loss, val_accuracy = test(model, val_loader, criterion, device, PATCH_ENABLE, PATCH_SIZE, AGGREGATION, EVALUATION_MODE)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            break
        
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, PATCH_ENABLE, PATCH_SIZE, AGGREGATION, EVALUATION_MODE)
        val_loss, val_accuracy = test(model, val_loader, criterion, device, PATCH_ENABLE, PATCH_SIZE, AGGREGATION, EVALUATION_MODE) # Using test function for validation
        
        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        # Log current LR
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

    # Plot results
    # Plot results
    plot_name = os.path.basename(EXPERIMENT_NAME)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": val_losses, "accuracy": val_accuracies}, "loss", plot_name, OUTPUT_DIR)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": val_losses, "accuracy": val_accuracies}, "accuracy", plot_name, OUTPUT_DIR)
    
    # Save Model
    save_path = os.path.join(OUTPUT_DIR, f"{plot_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save results to JSON for sweep runner
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "best_val_accuracy": max(val_accuracies),
        "best_val_loss": min(val_losses),
        "final_val_accuracy": val_accuracies[-1],
        "final_val_loss": val_losses[-1],
        "train_accuracy": train_accuracies[-1],
        "train_loss": train_losses[-1],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "img_size": IMG_SIZE,
        "model_params": MODEL_PARAMS
    }
    if PATCH_ENABLE:
        results["patch_size"] = PATCH_SIZE
        results["aggregation"] = AGGREGATION
        results["evaluation_mode"] = EVALUATION_MODE

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'results.json')}")
    
    wandb.finish()