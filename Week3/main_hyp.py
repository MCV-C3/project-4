import torchvision.transforms.v2  as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchviz import make_dot
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
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

from models import SimpleModel, WraperModel

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

# Data augmentation example
def get_data_transforms():
    """
    Returns a Compose object with data augmentation transformations.
    """
    return Compose([
        RandomResizedCrop(size=224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")


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


# TODO: improve this function to store relevant plots
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
        grayscale_cam = model.extract_grad_cam(img_tensor.unsqueeze(0).to(device), target_layers, targets)
        
        # Convert tensor back to image for visualization [cite: 426]
        # Reverse normalization for visualization
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        save_path = os.path.join(output_dir, f"{plot_name}_gradcam_{i}.png")
        plt.imsave(save_path, visualization)
        
        # Log to WandB
        wandb.log({f"grad_cam_{i}": wandb.Image(save_path, caption=f"Class: {dataset.classes[label]}")})



# Helper functions for dynamic construction
def get_optimizer(model, optimizer_name, lr, momentum, weight_decay):
    params = model.parameters() # Or filter for requires_grad?
    # WraperModel usually sets requires_grad=False for frozen layers
    # But it is safer to pass only trainable params to some optimizers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    name = optimizer_name.lower()
    if name == 'sgd':
        return optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adagrad':
        return optim.Adagrad(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == 'adadelta':
        return optim.Adadelta(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return optim.Adamax(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        return optim.NAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}', defaulting to Adam")
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, scheduler_name, epochs):
    name = scheduler_name.lower()
    if name == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    elif name == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif name == 'linearlr':
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    elif name == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        return None  # No scheduler

def get_activation(activation_name):
    name = activation_name.lower()
    if name == 'softmax':
        return nn.Softmax(dim=1)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'softsign':
        return nn.Softsign()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'hard_sigmoid':
        return nn.Hardsigmoid()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU() # Default

def main():
    # Initialize WandB
    wandb.init()
    config = wandb.config

    # --- Hyperparameter Setup with Defaults ---
    # Core Params
    BATCH_SIZE = config.get("batch_size", 32)
    EPOCHS = config.get("epochs", 20)
    LR = config.get("lr", 0.001)
    NUM_WORKERS = config.get("num_workers", 8)
    IMG_SIZE = config.get("img_size", 224)
    
    # Model Params
    UNFREEZE_DEPTH = config.get("unfreeze_depth", 0)
    HEAD_CONFIG = config.get("head_config", None)
    LEVEL = config.get("level", 4)
    DROPOUT_RATE = config.get("dropout", 0.0)
    ACTIVATION_NAME = config.get("activation", "relu")
    
    # Optimizer Params
    OPTIMIZER_NAME = config.get("optimizer", "adam")
    MOMENTUM = config.get("momentum", 0.0) # For SGD/RMSprop
    WEIGHT_DECAY = config.get("decay", 0.0)
    
    # Scheduler Params
    SCHEDULER_NAME = config.get("scheduler", "None")

    # Data Augmentation
    USE_AUGMENTATION = config.get("data_augmentation", False)

    # Dynamic Experiment Name
    # "hyperparameter_optimization" prefix + params
    EXPERIMENT_NAME = (
        f"hyp_opt_"
        f"lr{LR}_bs{BATCH_SIZE}_"
        f"opt{OPTIMIZER_NAME}_mom{MOMENTUM}_dec{WEIGHT_DECAY}_"
        f"sch{SCHEDULER_NAME}_"
        f"ep{EPOCHS}_"
        f"aug{int(USE_AUGMENTATION)}_"
        f"do{DROPOUT_RATE}_"
        f"frz{UNFREEZE_DEPTH}_"
        f"act{ACTIVATION_NAME}"
    )

    # Update WandB run name to match (useful for UI)
    wandb.run.name = EXPERIMENT_NAME

    DATASET_PATH = config.get("dataset_path", "/ghome/group04/mcv/datasets/C3/2425/MIT_small_train_1")
    OUTPUT_DIR = os.path.join("experiments", "sweeps", EXPERIMENT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Experiment: {EXPERIMENT_NAME} ---")
    print(f"Output directory: {OUTPUT_DIR}") 
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    if USE_AUGMENTATION:
        print("Using Data Augmentation")
        transform_train = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        print("Using Basic Transforms (No Augmentation)")
        transform_train = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize(size=(IMG_SIZE, IMG_SIZE)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Test transform is always just clean resize
    transform_test = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(IMG_SIZE, IMG_SIZE)),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading data from {DATASET_PATH}")
    data_train = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=transform_train)
    data_test = ImageFolder(os.path.join(DATASET_PATH, "test"), transform=transform_test) 
    
    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    # --- Model Setup ---
    model = WraperModel(num_classes=8, truncation_level=LEVEL)

    if HEAD_CONFIG is not None:
        model.modify_classifier_head(
            hidden_dims=HEAD_CONFIG.get("hidden_dims", None),
            activation=HEAD_CONFIG.get("activation", "relu"),
        )

    model = model.to(device)
    
    # 1. Unfreeze layers
    model.fine_tuning(unfreeze_blocks=UNFREEZE_DEPTH)
    
    model.summary()

    # 2. Modify Head (for Dropout and Activation experimentation)
    # The default head is just a Linear layer. We replace it to add capability for dropout/activation
    # We create a new structure: Linear(in, in) -> Activation -> Dropout -> Linear(in, classes)
    # This preserves the feature dimension for one extra layer to apply activation/dropout
    in_features = model.backbone_fc.in_features
    num_classes = 8
    
    model.backbone_fc = nn.Sequential(
        nn.Linear(in_features, in_features),
        get_activation(ACTIVATION_NAME),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, num_classes)
    ).to(device)
    
    print(f"Modified Head: {model.backbone_fc}")

    # --- Optimizer & Scheduler ---
    optimizer = get_optimizer(model, OPTIMIZER_NAME, LR, MOMENTUM, WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, SCHEDULER_NAME, EPOCHS)

    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    wandb.watch(model, log="all")

    for epoch in tqdm.tqdm(range(EPOCHS), desc="TRAINING"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        # Step the scheduler if it exists
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
            
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr}, commit=False)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
    plot_name = os.path.basename(EXPERIMENT_NAME)

    # Save Model
    save_path = os.path.join(OUTPUT_DIR, f"{plot_name}.pth")
    torch.save(model.state_dict(), save_path)
    
    # Plot results
    print("Generating Plots...")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", plot_name, OUTPUT_DIR)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", plot_name, OUTPUT_DIR)
   
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(model, test_loader, device, data_test.classes, plot_name, OUTPUT_DIR)
    
    # Skip GradCAM to save time during sweeps unless needed? Keeping it for now.
    print("Generating Grad-CAM samples...")
    plot_grad_cam_samples(model, data_test, device, plot_name, OUTPUT_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()
