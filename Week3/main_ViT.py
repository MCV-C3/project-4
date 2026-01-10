import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchviz import make_dot
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import ViTImageProcessor

import numpy as np
import os
import tqdm
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models_ViT import WraperModel


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


def get_train_transform(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.RandomHorizontalFlip(p=.5),
        F.Resize(size=(size, size)),
        F.Normalize(mean=image_mean, std=image_std),
    ])

def get_test_transform(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(size, size)),
        F.Normalize(mean=image_mean, std=image_std),
    ])


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
    MODEL_NAME = config.get("model", "google/vit-large-patch16-224")

    # Fine-tuning parameters
    HEAD_CONFIG = config.get("head_config", None)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if str(device) == 'cpu':
        print("WARNING: Training on CPU. This might be slow.")


    EXPERIMENT_NAME = config.get("experiment_name", "ViT")

    # Create Output Directory (per variation)
    OUTPUT_DIR = os.path.join("ViT_experiments", EXPERIMENT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


    # Initialize WandB
    wandb.init(
        project=config.get("wandb_project", "week3"),
        entity=config.get("wandb_entity", None),
        name=EXPERIMENT_NAME,
        config={**config},
        dir=OUTPUT_DIR,
        mode=config.get("wandb_mode", "online"),
        reinit=True,           # IMPORTANT: allow multiple init() in one process
    )

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Data Augmentation for Training
    train_transform = get_train_transform(processor)
    test_transform = get_test_transform(processor)

    print(f"Loading data from {DATASET_PATH}")

    data_train = ImageFolder(os.path.join(DATASET_PATH, "train"),
                             transform=train_transform)
    data_test = ImageFolder(os.path.join(DATASET_PATH, "test"),
                            transform=test_transform)


    train_loader = DataLoader(
        data_train, batch_size=BATCH_SIZE,
        pin_memory=True, shuffle=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        data_test, batch_size=BATCH_SIZE,
        pin_memory=True, shuffle=False, num_workers=NUM_WORKERS,
    )

    # fine-tune the model
    model = WraperModel(num_classes=8, model_name=MODEL_NAME)

    if HEAD_CONFIG is not None:
        model.modify_classifier_head(
            hidden_dims=HEAD_CONFIG.get("hidden_dims", None),
            activation=HEAD_CONFIG.get("activation", "relu"),
        )

    model.to(device)

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

    # Save results to JSON
    results = {
        "experiment_name": EXPERIMENT_NAME,
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single batch for testing")
    args = parser.parse_args()

    main(args)
