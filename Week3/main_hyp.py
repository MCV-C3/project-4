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


def get_train_transform(augmentation_transform: Optional[str | list], img_size: int):
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
        #"color": F.RandomApply([F.ColorJitter(brightness=.5, hue=.3)], p=0.5),
    }

    # decide which augmentation ops to apply
    aug_ops = []

    if augmentation_transform is None or augmentation_transform == "none":
        aug_ops = []

    elif augmentation_transform == "all":
        aug_ops = list(augmentation_transform_map.values())
            # compose pipeline
        return F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomChoice([*aug_ops]),
            F.Resize(size=(img_size, img_size)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
        
        return F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.RandomChoice([*aug_ops]),
            F.Resize(size=(img_size, img_size)),
            F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        raise TypeError(
            "augmentation_transform must be None, 'all', str, list, or tuple.")

    # compose pipeline
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        *aug_ops,
        F.Resize(size=(img_size, img_size)),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    # Dynamic target layer selection: use the last element of the backbone
    # model.backbone is nn.Sequential. We want the last block (which is a ResNet bottleneck usually)
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


# --- Architecture Modification Helpers ---
def replace_normalization(model: nn.Module, norm_type: str):
    """
    Recursively replaces BatchNorm2d layers with other normalization layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Get the number of channels
            num_features = module.num_features
            
            if norm_type == 'layer':
                # LayerNorm in 2D is tricky. GroupNorm with 1 group is mathematically equivalent 
                # to LayerNorm per channel-set, often used as a substitute in CNNs.
                new_layer = nn.GroupNorm(num_groups=1, num_channels=num_features)
            elif norm_type == 'instance':
                new_layer = nn.InstanceNorm2d(num_features, affine=True)
            elif norm_type == 'group':
                # Default to 32 groups or num_features if less
                groups = 32 if num_features % 32 == 0 else num_features // 2
                groups = max(1, groups)
                new_layer = nn.GroupNorm(num_groups=groups, num_channels=num_features)
            elif norm_type == 'batch':
                continue # Already batch
            else:
                print(f"Warning: Unknown norm type {norm_type}, keeping BatchNorm")
                continue
            
            setattr(model, name, new_layer)
        else:
            replace_normalization(module, norm_type)

def replace_activation(model: nn.Module, new_activation_name: str):
    """
    Recursively replaces nn.ReLU layers with a new activation function.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU): # Assuming Backbone uses ReLU
             # Create new activation instance
            new_layer = get_activation(new_activation_name)
            setattr(model, name, new_layer)
        else:
            replace_activation(module, new_activation_name)

def inject_dropout_after_activation(module: nn.Module, p: float, activation_type: type):
    """
    Recursively injects Dropout layers after a specific activation type.
    """
    for name, child in module.named_children():
        if isinstance(child, activation_type):
            new_layer = nn.Sequential(
                child,
                nn.Dropout(p=p)
            )
            setattr(module, name, new_layer)
        else:
            inject_dropout_after_activation(child, p, activation_type)




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
    NUM_WORKERS = config.get("num_workers", 8)
    IMG_SIZE = config.get("img_size", 224)
    
    # Model Params
    UNFREEZE_DEPTH = config.get("unfreeze_depth", 0)
    HEAD_CONFIG = config.get("head_config", {"hidden_dims": [512], "activation": "relu"})
    LEVEL = config.get("level", 3)

    # Data Augmentation
    AUGMENTATION_NAME = config.get("data_augmentation", ["gaussian_noise", "horizontal_flip", "crop", "photometric_distort"])

    # Learning Params
    DROPOUT_METHOD = config.get("dropout_method", "classifier") # classifier, feature_extractor
    DROPOUT_RATE = config.get("dropout", 0.0)
    NORMALIZATION = config.get("norm", "batch") # batch, layer, instance, group
    L1_REG = config.get("l1_reg", 0.0) # 0.0 means off
    L2_REG = config.get("l2_reg", 0.0) # Default L2 (weight decay)
    WEIGHT_DECAY = L2_REG
    
    # Hyperparameters
    BATCH_SIZE = config.get("batch_size", 32)
    EPOCHS = config.get("epochs", 50)
    LR = config.get("lr", 0.0001)
    ACTIVATION_NAME = config.get("activation", "relu")
    OPTIMIZER_NAME = config.get("optimizer", "adam")
    MOMENTUM = config.get("momentum", 0.0) # For SGD/RMSprop
    SCHEDULER_NAME = config.get("scheduler", "None")
    
    # Dynamic Experiment Name
    RUN_NAME_FORMAT = config.get("run_name_format", None)
    
    if RUN_NAME_FORMAT:
        config_dict = dict(config)
        config_dict.update({
            "DROPOUT_METHOD": DROPOUT_METHOD,
            "DROPOUT_RATE": DROPOUT_RATE,
            "NORMALIZATION": NORMALIZATION,
            "L1_REG": L1_REG,
            "L2_REG": L2_REG,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "OPTIMIZER_NAME": OPTIMIZER_NAME,
            "MOMENTUM": MOMENTUM,
            "SCHEDULER_NAME": SCHEDULER_NAME,
        })
        try:
            EXPERIMENT_NAME = RUN_NAME_FORMAT.format(**config_dict)
        except KeyError as e:
            print(f"Warning: Key {e} for run_name_format not found in config. Using default naming.")
            RUN_NAME_FORMAT = None # Fallback

    if not RUN_NAME_FORMAT:
        # Default naming convention
        EXPERIMENT_NAME = (
            f"hyp_opt_"
            f"do{DROPOUT_RATE}({DROPOUT_METHOD})_"
            f"norm{NORMALIZATION}_"
            f"l1{l1_reg}_"
            f"l2{l2_reg}_"
            f"bs{batch_size}_"
            f"ep{epochs}_"
            f"lr{lr}_"
            f"opt{optimizer}_"
            f"mom{momentum}_"
            f"sch{scheduler}_"
        )
    wandb.run.name = EXPERIMENT_NAME

    DATASET_PATH = config.get("dataset_path", "/ghome/group04/mcv/datasets/C3/2425/MIT_small_train_1")
    OUTPUT_DIR = os.path.join("experiments", "hyp_opt", EXPERIMENT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Experiment: {EXPERIMENT_NAME} ---")
    print(f"Output directory: {OUTPUT_DIR}") 
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    print(f"Using Data Augmentation: {AUGMENTATION_NAME}")
    transform_train = get_train_transform(AUGMENTATION_NAME, IMG_SIZE)

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
    print(f"Initializing model with Truncation Level: {LEVEL}")
    model = WraperModel(num_classes=8, truncation_level=LEVEL)
    
    # 1. Apply Normalization Changes (Pre-Head)
    # Replaces BatchNorm with other types if requested. 
    # Must be done before moving to device to ensure new layers are properly registered? 
    # Actually safe to do before.
    if NORMALIZATION != "batch":
        print(f"Applying {NORMALIZATION} normalization to backbone...")
        replace_normalization(model.backbone, NORMALIZATION)

    # 2. Modify Activations in Feature Extractor (Backbone)
    # Default ResNeXt uses ReLU. If we want to sweep activations for the whole model (not just head),
    # we should replace them here.
    if ACTIVATION_NAME.lower() != "relu":
         print(f"Replacing backbone activations with {ACTIVATION_NAME}...")
         replace_activation(model.backbone, ACTIVATION_NAME)

    # 3. Modify Dropout in Feature Extractor (if requested)
    # This edits the backbone layers (after activations)
    if DROPOUT_METHOD == "feature_extractor" and DROPOUT_RATE > 0:
        # We need to know which activation to target. 
        # If we replaced it, it's the new one. If not, it's ReLU.
        # Note: get_activation returns an instance, we need the type.
        act_instance = get_activation(ACTIVATION_NAME)
        act_type = type(act_instance)
        
        print(f"Applying Dropout ({DROPOUT_RATE}) to Feature Extractor (after {act_type.__name__})")
        inject_dropout_after_activation(model.backbone, DROPOUT_RATE, act_type)

    # 3. Modify Classifier Head
    # Logic: If method is classifier, we pass dropout to the head builder.
    # If method is feature_extractor, head gets 0 dropout.
    head_dropout = DROPOUT_RATE if DROPOUT_METHOD == "classifier" else 0.0
    
    # Check if we need to modify the head (either due to custom config OR dropout)
    # If HEAD_CONFIG is None, we default to no hidden layers (standard Linear head)
    if HEAD_CONFIG is not None or head_dropout > 0:
        current_config = HEAD_CONFIG if HEAD_CONFIG is not None else {}
        hidden_dims = current_config.get("hidden_dims", [512])
        activation = current_config.get("activation", "relu")
        
        print(f"Building Classifier Head. Hidden Dims: {hidden_dims}, Activation: {activation}, Dropout: {head_dropout}, Normalization: {NORMALIZATION}")
        model.modify_classifier_head(
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=head_dropout,
            normalization=NORMALIZATION
        )
    
    # 4. Unfreeze layers (Fine Tuning)
    # Should be done after replacing head, so head params are tracked correctly.
    print(f"Setting Unfreeze Depth: {UNFREEZE_DEPTH}")
    model.fine_tuning(unfreeze_blocks=UNFREEZE_DEPTH)
    
    # Move to device
    model = model.to(device)
    
    # --- Model Structure Verification ---
    print("\n" + "="*50)
    print("FINAL MODEL STRUCTURE VERIFICATION")
    print("="*50)
    model.summary()
    
    # Print samples of the backbone to verify deep changes (activations, normalization, dropout)
    print("\n--- Backbone Sample: Layer 1, Block 0 ---")
    if hasattr(model.backbone, 'layer1'):
        print(model.backbone.layer1[0])
        
    print("\n--- Backbone Sample: Layer 4, Last Block ---")
    if hasattr(model.backbone, 'layer4'):
         print(model.backbone.layer4[-1])
    print("="*50 + "\n")

    # --- Optimizer & Scheduler ---
    optimizer = get_optimizer(model, OPTIMIZER_NAME, LR, MOMENTUM, WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, SCHEDULER_NAME, EPOCHS)

    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    wandb.watch(model, log="all")

    plot_name = os.path.basename(EXPERIMENT_NAME)
    best_test_accuracy = 0.0
    best_epoch = 0

    for epoch in tqdm.tqdm(range(EPOCHS), desc="TRAINING"):

        # Custom train step for L1 regularization
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L1 Regularization
            if L1_REG > 0:
                l1_penalty = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_penalty += torch.norm(param, 1)
                loss += L1_REG * l1_penalty
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
        train_loss = train_loss / total
        train_accuracy = correct / total
        
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

        # Save best model logic
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            save_path = os.path.join(OUTPUT_DIR, f"{plot_name}.pth")
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}" + 
              (f" [BEST]" if test_accuracy == best_test_accuracy else ""))
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
    # Plot results
    print(f"Best model saved with accuracy: {best_test_accuracy:.4f} at epoch {best_epoch}")
    
    # Plot results
    print("Generating Plots...")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", plot_name, OUTPUT_DIR)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", plot_name, OUTPUT_DIR)

    plot_confusion_matrix(model, test_loader, device, data_test.classes, plot_name, OUTPUT_DIR)
    
    # Skip GradCAM to save time during sweeps unless needed? Keeping it for now.
    plot_grad_cam_samples(model, data_test, device, plot_name, OUTPUT_DIR)

    plot_grad_cam_samples(model, data_test, device, plot_name, OUTPUT_DIR)

    # Save Results JSON
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "best_test_accuracy": max(test_accuracies),
        "best_epoch": best_epoch,
        "best_test_loss": min(test_losses),
        "final_test_accuracy": test_accuracies[-1],
        "final_test_loss": test_losses[-1],
        "train_accuracy": train_accuracies[-1],
        "train_loss": train_losses[-1],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "optimizer": OPTIMIZER_NAME,
        "l2_reg": WEIGHT_DECAY,
        "l1_reg": L1_REG,
        "dropout": DROPOUT_RATE,
        "dropout_method": DROPOUT_METHOD,
        "normalization": NORMALIZATION,
        "augmentation": AUGMENTATION_NAME,
        "unfreeze_depth": UNFREEZE_DEPTH,
        "head_config": HEAD_CONFIG,
        "level": LEVEL,
        "trainable_params": model.get_trainable_parameters(),
        "total_params": model.get_total_parameters(),
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'results.json')}")

    wandb.finish()


if __name__ == "__main__":
    main()
