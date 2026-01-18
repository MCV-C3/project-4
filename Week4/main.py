import argparse
from dataclasses import dataclass
import inspect
import json
import os
from typing import Any, Dict, List

import yaml
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms

from utils.metrics import (
    get_model_summary,
    compute_efficiency_score,
    calculate_classification_metrics
)
from utils.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics,
    plot_computational_graph,
    plot_grad_cam_samples
)

from utils.sweeps import configure_sweep

import models


@dataclass
class Exp:
    """Container for experiment-level configuration and environment state."""

    cfg: Dict[str, Any]          # Parsed experiment configuration dictionary
    device: Any                  # Torch device used for training and evaluation
    output_dir: str              # Directory where results and artifacts are stored
    best_model_path: str         # Path for saving the best model weights


@dataclass
class Data:
    """Container for dataset-related objects and loaders."""

    classes: List[str]           # List of class names
    train_loader: Any            # DataLoader for the training set
    val_loader: Any              # DataLoader for the validation set
    test_loader: Any             # DataLoader for the test set


@dataclass
class Run:
    """Container for model, optimization, and training state."""

    model: Any                   # Neural network model instance
    criterion: Any               # Loss function used for training
    optimizer: Any               # Optimizer instance
    # Dictionary storing training and validation metrics
    history: Dict[str, list]
    params: int                  # Number of model parameters
    flops: int                   # Estimated floating point operations
    latency: float               # Estimated inference latency in milliseconds
    best_val_acc: float = 0.0    # Best validation accuracy achieved so far
    best_epoch: int = 0          # Epoch corresponding to best validation accuracy


@dataclass
class Eval:
    """Container for evaluation outputs."""

    all_labels: List[str]        # Ground truth labels for evaluated samples
    all_preds: List[int]         # Model predictions
    all_probs: List[float]       # Predicted class probabilities


def get_transforms(img_size: int, is_train: bool = True):
    """Create image transformation pipeline.

    Args:
        img_size: Target image size for resizing and cropping.
        is_train: Whether to apply data augmentation.

    Returns:
        A torchvision Compose object with the requested transforms.
    """
    if is_train:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.8, 1.0)),  # Slight zoom/crop
            # Rotations up to 15 degrees
            transforms.RandomRotation(degrees=15),
            # Robustness to lighting
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])


def setup_experiment(config_path: str) -> Exp:
    """Initialize experiment configuration, logging, and environment.

    Loads the YAML configuration file, initializes Weights & Biases logging,
    sets the output directory, and selects the compute device.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Exp object containing configuration and environment information.
    """

    # Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup WandB
    wandb.init(project=cfg['project'], config=cfg,
               name=cfg['experiment_name'], allow_val_change=True)

    # Change the config for the sweeps
    if wandb.run.sweep_id:
        cfg = configure_sweep(cfg)

        # If the sweep injects 'experiment_name' into model.params use it for wandb run name
        if 'experiment_name' in cfg['model'].get('params', {}):
            cfg['experiment_name'] = cfg['model']['params'].pop(
                'experiment_name')
            wandb.run.name = cfg['experiment_name']

        output_dir = os.path.join(
            "results", "sweeps", wandb.run.sweep_id, cfg['experiment_name'])
    else:
        # Standard run
        output_dir = os.path.join(
            "results", f"{cfg['experiment_name']}_{wandb.run.id}")

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(cfg['training']['seed'])

    best_model_path = os.path.join(output_dir, "best_model.pth")

    return Exp(cfg, device, output_dir, best_model_path)


def setup_data(exp: Exp) -> Data:
    """Prepare datasets and data loaders.

    Loads training and test datasets, splits the test set into validation
    and test subsets, and constructs PyTorch DataLoaders.

    Args:
        exp: Experiment configuration container.

    Returns:
        Data object containing classes and data loaders.
    """

    cfg = exp.cfg

    # Data Loading
    train_dataset = ImageFolder(
        cfg['data']['train_dir'],
        transform=get_transforms(cfg['data']['img_size'], True))

    all_test_dataset = ImageFolder(
        cfg['data']['test_dir'],
        transform=get_transforms(cfg['data']['img_size'], False))

    # Split test dataset into validation and test sets
    val_percentage = cfg['data'].get('val_split', 0.7)
    val_size = int(len(all_test_dataset) * val_percentage)
    test_size = len(all_test_dataset) - val_size

    generator = torch.Generator().manual_seed(cfg['training']['seed'])
    val_dataset, test_dataset = random_split(
        all_test_dataset, [val_size, test_size], generator=generator)

    print(f"Data Split: {len(train_dataset)} Train | {len(val_dataset)} Val | "
          f"{len(test_dataset)} Test")

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'])
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg['data']['num_workers'])

    classes = train_dataset.classes

    return Data(classes, train_loader, val_loader, test_loader)


def build_model_from_config(cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    """Instantiate a model class by name using params from the config.

    Expects:
      cfg["model"]["name"]: class name exported in models/__init__.py
      cfg["model"]["params"]: kwargs passed to the model constructor
    """
    model_name = cfg["model"]["name"]

    # Look up class by name from the models package
    try:
        model_cls = getattr(models, model_name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Make sure it's imported in models/__init__.py."
        ) from e

    # Get constructor kwargs from config
    params = dict(cfg["model"].get("params", {}))
    params["num_classes"] = num_classes  # always inject

    # Filter params to only those accepted by the constructor
    # (prevents accidental YAML keys from crashing the run)
    sig = inspect.signature(model_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in params.items() if k in accepted}

    # If constructor has **kwargs, don't filter (let it handle extras)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kwargs:
        filtered_params = params

    return model_cls(**filtered_params)


def setup_model(exp: Exp, data: Data) -> Run:
    """Initialize model, optimizer, and training state.

    Builds the selected model architecture, computes model statistics,
    and prepares the loss function and optimizer.

    Args:
        exp: Experiment configuration container.
        data: Dataset container with class information.

    Returns:
        Run object containing model and training state.
    """

    cfg = exp.cfg
    device = exp.device
    classes = data.classes

    model = build_model_from_config(cfg, num_classes=len(classes)).to(device)

    # Model Summary
    print("Model Summary:")
    img_size = cfg['data']['img_size']
    summary(model, (3, img_size, img_size))

    # Metrics Calculation (Pre-training)
    params, flops, latency = get_model_summary(
        model, (3, cfg['data']['img_size'], cfg['data']['img_size']), device)
    print(f"Model Params: {params/1e6:.2f}M | FLOPs: {flops/1e9:.2f}G | "
          f"Latency: {latency:.2f}ms")
    wandb.log({"Parameters": params, "FLOPs": flops, "Latency_ms": latency})

    # Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg['training']['learning_rate'])

    # History containers (for plotting later)
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    # Initialize Best Model Trackers
    best_val_acc = 0.0
    best_epoch = 0

    return Run(model, criterion, optimizer, history, params, flops, latency,
               best_val_acc, best_epoch)


def train(exp: Exp, data: Data, run: Run) -> Run:
    """Train the model and track best validation performance.

    Executes the training loop for the configured number of epochs,
    evaluates on the validation set, logs metrics to Weights & Biases,
    and saves the best performing model.

    Args:
        exp: Experiment configuration container.
        data: Dataset container with data loaders.
        run: Model and optimization state.

    Returns:
        Updated Run object with training history and best model information.
    """

    cfg = exp.cfg
    device = exp.device
    best_model_path = exp.best_model_path

    train_loader = data.train_loader
    val_loader = data.val_loader

    model = run.model
    criterion = run.criterion
    optimizer = run.optimizer
    history = run.history
    best_val_acc = run.best_val_acc
    best_epoch = run.best_epoch

    # Training Loop
    print("Starting training...")
    for epoch in tqdm.tqdm(range(cfg['training']['epochs']), desc="TRAINING THE MODEL"):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track loss and accuracy
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        # Log History
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model! Saved (Val Acc: {best_val_acc:.4f})")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "best_val_accuracy": best_val_acc
        })

    print(f"\nTraining finished. Loading best model from Epoch {best_epoch} "
          f"(Acc: {best_val_acc:.4f})...")
    model.load_state_dict(torch.load(best_model_path))

    return run


def evaluate(exp: Exp, data: Data, run: Run) -> Eval:
    """Evaluate the best model on the validation set.

    Computes final validation metrics using the best saved model weights,
    logs results, and saves them to a JSON file.

    Args:
        exp: Experiment configuration container.
        data: Dataset container with data loaders.
        run: Model and optimization state.

    Returns:
        Eval object containing predictions, labels, and probabilities.
    """

    cfg = exp.cfg
    device = exp.device
    output_dir = exp.output_dir

    val_loader = data.val_loader

    model = run.model
    criterion = run.criterion
    params = run.params
    best_epoch = run.best_epoch
    flops = run.flops
    latency = run.latency

    print("Evaluating Best Model on Validation Set...")
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # We need to re-generate these lists using the BEST model weights
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Calculate aggregate metrics for the best model
    final_val_acc = val_correct / val_total
    final_val_loss = val_loss / val_total

    precision, recall, f1 = calculate_classification_metrics(
        all_labels, all_preds)
    efficiency_score = compute_efficiency_score(final_val_acc, params)

    print(f"\nBest Validation Results (Epoch {best_epoch}):")
    print(f"Accuracy: {final_val_acc:.4f}")
    print(f"Efficiency Ratio: {efficiency_score:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    wandb.log({
        "final_val_accuracy": final_val_acc,
        "efficiency_score": efficiency_score,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Prepare results dictionary
    results_data = {
        "configuration": cfg,
        "metrics": {
            "best_epoch": best_epoch,
            "final_val_accuracy": final_val_acc,
            "final_val_loss": final_val_loss,
            "efficiency_score": efficiency_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model_params": params,
            "model_flops": flops,
            "inference_latency_ms": latency
        }
    }

    # Save Results to JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Results and config saved to {json_path}")

    return Eval(all_labels, all_preds, all_probs)


def generate_plots(exp: Exp, data: Data, run: Run, eval_: Eval) -> None:
    """Generate and save evaluation plots.

    Creates visualizations including training curves, confusion matrix,
    ROC curves, and computational graph diagrams.

    Args:
        exp: Experiment configuration container.
        data: Dataset container with class information.
        run: Training state container with history and model.
        eval_: Evaluation results container.
    """

    cfg = exp.cfg
    output_dir = exp.output_dir

    classes = data.classes

    history = run.history
    model = run.model

    all_labels = eval_.all_labels
    all_preds = eval_.all_preds
    all_probs = eval_.all_probs

    # Generate and Save Plots
    # (History is safe to use as-is because it tracks the whole training curve)
    train_metrics = {
        "loss": history['train_loss'],
        "accuracy": history['train_acc']
    }
    val_metrics = {
        "loss": history['val_loss'],
        "accuracy": history['val_acc']
    }

    plot_metrics(train_metrics, val_metrics, "loss", output_dir)
    plot_metrics(train_metrics, val_metrics, "accuracy", output_dir)

    # Plot Confusion Matrix and ROC Curve (Now using the correct 'all_preds'
    # from best model)
    plot_confusion_matrix(all_labels, all_preds, classes, output_dir)
    plot_roc_curve(all_labels, all_probs, classes, output_dir)

    # Plot Computational Graph (requires model input size)
    # We create a dummy input size tuple: (Batch_Size, Channels, Height, Width)
    input_dims = (1, 3, cfg['data']['img_size'], cfg['data']['img_size'])
    plot_computational_graph(model, input_dims, output_dir)

    # GradCAM not needed for the moment

    # # Save Model
    # model_save_path = os.path.join(output_dir, "model_weights.pth")
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")


def main(config_path: str) -> None:
    """Execute the full experiment pipeline.

    Runs experiment setup, data loading, model initialization,
    training, evaluation, and plot generation.

    Args:
        config_path: Path to YAML configuration file.
    """

    exp = setup_experiment(config_path)

    data = setup_data(exp)

    run = setup_model(exp, data)

    run = train(exp, data, run)

    eval_ = evaluate(exp, data, run)

    generate_plots(exp, data, run, eval_)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml",
        help="Path to config file")
    args = parser.parse_args()

    main(args.config)
