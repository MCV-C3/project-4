from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import argparse
import json
import os
import wandb

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
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    try:
        # torchviz appends .png automatically, but we want to control the path
        # It's easier to verify where it saves. make_dot(...).render(path) saves to path.png
        save_path = os.path.join(output_dir, filename)
        graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(save_path, format="png")
        print(f"Computational graph saved as {save_path}.png")
    except OSError as e:
        print(f"Could not render computational graph (Graphviz might be missing): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 MLP Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
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
    MODEL_PARAMS = config.get("model_params", {"hidden_d": 300})
    
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

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    # Ensure data directory exists could be added here for robustness
    
    print(f"Loading data from {DATASET_PATH}")
    # Using 'train' and 'val' subfolders
    data_train = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=transformation)
    data_test = ImageFolder(os.path.join(DATASET_PATH, "val"), transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    C, H, W = data_train[0][0].numpy().shape
    num_classes = len(data_train.classes)

    model = SimpleModel(input_d=C*H*W, output_d=num_classes, **MODEL_PARAMS)
    plot_computational_graph(model, input_size=(1, C*H*W), filename="computational_graph", output_dir=OUTPUT_DIR) 

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    wandb.watch(model, log="all")

    for epoch in tqdm.tqdm(range(EPOCHS), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", EXPERIMENT_NAME, OUTPUT_DIR)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", EXPERIMENT_NAME, OUTPUT_DIR)
    
    # Save Model
    save_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")