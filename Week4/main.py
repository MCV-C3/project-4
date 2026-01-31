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
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
try:
    from torchsummary import summary
except ImportError:
    print("Warning: torchsummary not found. Model summary will be skipped.")
    summary = lambda *args, **kwargs: print("Summary skipped.")
from torch.utils.data import DataLoader
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
    test_loader: Any             # DataLoader for the test set


@dataclass
class Run:
    """Container for model, optimization, and training state."""

    model: Any                   # Neural network model instance
    criterion: Any               # Loss function used for training
    optimizer: Any               # Optimizer instance
    # Dictionary storing training and test metrics
    history: Dict[str, list]
    params: int                  # Number of model parameters
    flops: int                   # Estimated floating point operations
    latency: float               # Estimated inference latency in milliseconds
    best_test_acc: float = 0.0   # Best test accuracy achieved so far
    best_epoch: int = 0          # Epoch corresponding to best test accuracy
    # Teacher model instance (optional, for distillation)
    teacher: Any = None
    scheduler: Any = None        # Optional scheduler


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


def distillation_loss_fn(student_logits, teacher_logits, labels, T, alpha, criterion):
    """Computes the distillation loss (KL Divergence + CrossEntropy)."""
    # KL Divergence Loss
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_softmax = F.log_softmax(student_logits / T, dim=1)
    kl_loss = F.kl_div(student_log_softmax, soft_targets,
                       reduction='batchmean') * (T ** 2)

    # Standard CrossEntropy
    ce_loss = criterion(student_logits, labels)

    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return total_loss, kl_loss, ce_loss


def _get_value_by_path(d: Dict[str, Any], path: str) -> Any:
    """Helper to retrieve value from nested dict using dotted path."""
    keys = path.split('.')
    val = d
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return "N/A"
    return val


def format_run_name(cfg: Dict[str, Any], fmt: str) -> str:
    """Format run name string using values from config."""
    # Find all placeholders like {data.batch_size}
    import re
    placeholders = re.findall(r'\{([^}]+)\}', fmt)

    result = fmt
    for p in placeholders:
        val = _get_value_by_path(cfg, p)
        # Replace {p} with str(val)
        result = result.replace(f"{{{p}}}", str(val))

    return result


def build_teacher_model(cfg: Dict[str, Any], num_classes: int, device: Any) -> torch.nn.Module:
    """Builds and loads the teacher model for distillation."""
    print("Loading Teacher Model...")
    # Parameters for the teacher as identified in Week 3
    teacher = models.TeacherWrapperModel(
        num_classes=num_classes, truncation_level=3)

    # Apply modifications as per Week 3 results.json
    teacher.modify_classifier_head(
        hidden_dims=[512],
        activation='relu',
        dropout=0.5,
        normalization='batch'
    )

    teacher_path = cfg['distillation']['teacher_path']
    print(f"Loading teacher weights from {teacher_path}")

    state_dict = torch.load(teacher_path, map_location=device)
    teacher.load_state_dict(state_dict)

    teacher.to(device)
    teacher.eval()
    teacher.set_parameter_requires_grad(False)

    return teacher


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
        model_key = 'model' if 'model' in cfg else 'student_model'
        if model_key in cfg and 'experiment_name' in cfg[model_key].get('params', {}):
            cfg['experiment_name'] = cfg[model_key]['params'].pop(
                'experiment_name')
            wandb.run.name = cfg['experiment_name']

        # Customize run name for distillation sweeps
        if 'distillation' in cfg:
            alpha = cfg['distillation'].get('alpha')
            temp = cfg['distillation'].get('temperature')
            if alpha is not None and temp is not None:
                new_name = f"{cfg['experiment_name']}_alpha{alpha}_T{temp}"
                wandb.run.name = new_name
                cfg['experiment_name'] = new_name
                # Update wandb.config to reflect the new name if needed
                wandb.config.update(
                    {'experiment_name': new_name}, allow_val_change=True)

        # Handle 'rerun_config' for mixed-parameter sweeps
        if 'rerun_config' in cfg:
            print(">>> Applying rerun_config overrides:")
            for key, val in cfg['rerun_config'].items():
                if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                    def recursive_update(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                                recursive_update(d[k], v)
                            else:
                                d[k] = v
                    recursive_update(cfg[key], val)
                else:
                    cfg[key] = val

            # Re-trigger naming logic because experiment_name might have changed
            model_key = 'model' if 'model' in cfg else 'student_model'
            if model_key in cfg and 'experiment_name' in cfg[model_key].get('params', {}):
                cfg['experiment_name'] = cfg[model_key]['params'].pop(
                    'experiment_name')

            # Also re-trigger distillation naming suffix if those changed
            if 'distillation' in cfg:
                alpha = cfg['distillation'].get('alpha')
                temp = cfg['distillation'].get('temperature')
                if alpha is not None and temp is not None:
                    new_name = f"{cfg['experiment_name']}_alpha{alpha}_T{temp}"
                    wandb.run.name = new_name
                    cfg['experiment_name'] = new_name
                    wandb.config.update(
                        {'experiment_name': new_name}, allow_val_change=True)

        # Apply run_name_format if present (Overrides everything else)
        if 'run_name_format' in cfg:
            try:
                formatted_name = format_run_name(cfg, cfg['run_name_format'])
                cfg['experiment_name'] = formatted_name
                wandb.run.name = formatted_name
                wandb.config.update(
                    {'experiment_name': formatted_name}, allow_val_change=True)
                print(f"Set Run Name to: {formatted_name}")
            except Exception as e:
                print(f"Error formatting run name: {e}")

        output_dir = os.path.join(
            "results", "sweeps", wandb.run.sweep_id, f"{cfg['experiment_name']}_{wandb.run.id}")
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
    """Prepare datasets and data loaders (Train and Test only).

    Loads training and test datasets and constructs PyTorch DataLoaders.

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

    test_dataset = ImageFolder(
        cfg['data']['test_dir'],
        transform=get_transforms(cfg['data']['img_size'], False))

    print(f"Data Split: {len(train_dataset)} Train | {len(test_dataset)} Test")

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'])
    test_loader = DataLoader(
        test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'])

    classes = train_dataset.classes

    return Data(classes, train_loader, test_loader)


def build_model_from_config(cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    """Instantiate a model class by name using params from the config.

    Expects:
      cfg["model"]["name"]: class name exported in models/__init__.py
      cfg["model"]["params"]: kwargs passed to the model constructor
    """
    if "model" in cfg:
        model_name = cfg["model"]["name"]
        params = dict(cfg["model"].get("params", {}))
    elif "student_model" in cfg:
        # Alias for distillation configs
        model_name = cfg["student_model"]["name"]
        params = dict(cfg["student_model"].get("params", {}))
    else:
        raise ValueError(
            "Config must contain 'model' or 'student_model' section.")

    # Look up class by name from the models package
    try:
        model_cls = getattr(models, model_name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Make sure it's imported in models/__init__.py."
        ) from e

    # Get constructor kwargs from config
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

    # Optimizer Logic
    # Default to Adam if not in top-level
    opt_name = cfg.get('optimizer', 'Adam')
    if 'training' in cfg and 'optimizer' in cfg['training']:
        opt_name = cfg['training']['optimizer']

    lr = cfg['training']['learning_rate']
    momentum = cfg.get('momentum', 0.9)
    # momentum might be in training block depending on config structure
    if 'training' in cfg and 'momentum' in cfg['training']:
        momentum = cfg['training']['momentum']

    print(f"Initializing Optimizer: {opt_name} (LR: {lr})")

    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    else:
        print(
            f"Warning: Optimizer {opt_name} not explicitly handled, defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler Logic
    scheduler_name = cfg.get('scheduler', 'None')
    scheduler = None
    if scheduler_name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs'])
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10)

    if scheduler:
        print(f"Using Scheduler: {scheduler_name}")

    # History containers (for plotting later)
    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}

    # Initialize Best Model Trackers
    best_test_acc = 0.0
    best_epoch = 0

    # Distillation Setup
    teacher = None
    if 'distillation' in cfg:
        teacher = build_teacher_model(
            cfg, num_classes=len(classes), device=device)

    return Run(
        model=model,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        # We attach scheduler to Run so we can step it
        history=history,
        params=params,
        flops=flops,
        latency=latency,
        best_test_acc=best_test_acc,
        best_epoch=best_epoch,
        scheduler=scheduler
    )


def train(exp: Exp, data: Data, run: Run) -> Run:
    """Train the model and save best model based on Test Accuracy.
    Executes the training loop for the configured number of epochs,
    evaluates on the test set, logs metrics to Weights & Biases,
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
    test_loader = data.test_loader  # Use Test loader for evaluation

    model = run.model
    teacher = run.teacher
    criterion = run.criterion
    optimizer = run.optimizer
    history = run.history
    best_test_acc = run.best_test_acc
    best_epoch = run.best_epoch

    # Distillation Params
    T = cfg.get('distillation', {}).get('temperature', 1.0)
    alpha = cfg.get('distillation', {}).get('alpha', 0.0)

    # Training Loop
    print("Starting training...")
    for epoch in tqdm.tqdm(range(cfg['training']['epochs']), desc="TRAINING THE MODEL"):
        model.train()
        train_loss = 0.0
        train_kl = 0.0
        train_ce = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if teacher:
                # Distillation Forward
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                student_outputs = model(inputs)
                loss, kl, ce = distillation_loss_fn(
                    student_outputs, teacher_outputs, labels, T, alpha, criterion)

                # Track components
                train_kl += kl.item() * inputs.size(0)
                train_ce += ce.item() * inputs.size(0)
            else:
                # Standard Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                student_outputs = outputs  # Alias for acc calc

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        # Validation Loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track loss and accuracy
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_epoch_loss = test_loss / test_total
        test_epoch_acc = test_correct / test_total

        # Log History
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_loss'].append(test_epoch_loss)
        history['test_acc'].append(test_epoch_acc)

        log_msg = (f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
                   f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                   f"Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}")

        if teacher:
            epoch_kl = train_kl / total
            epoch_ce = train_ce / total
            log_msg += f" (KL: {epoch_kl:.4f}, CE: {epoch_ce:.4f})"

        print(log_msg)

        # Step Scheduler
        if hasattr(run, 'scheduler') and run.scheduler:
            if isinstance(run.scheduler, lr_scheduler.ReduceLROnPlateau):
                run.scheduler.step(test_epoch_acc)
            else:
                run.scheduler.step()

            # Log LR
            current_lr = run.optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr}, commit=False)

        # Save Best Model based on Test Accuracy
        if test_epoch_acc > best_test_acc:
            best_test_acc = test_epoch_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  >>> New Best Model! Saved (Test Acc: {best_test_acc:.4f})")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "test_loss": test_epoch_loss,
            "test_accuracy": test_epoch_acc,
            "best_test_accuracy": best_test_acc
        })

    print(f"\nTraining finished. Loading best model from Epoch {best_epoch} "
          f"(Acc: {best_test_acc:.4f})...")
    model.load_state_dict(torch.load(best_model_path))

    # Update run object with best results
    run.best_test_acc = best_test_acc
    run.best_epoch = best_epoch

    # Clean up scheduler if attached (though not strictly necessary)
    if hasattr(run, 'scheduler'):
        del run.scheduler

    return run


def evaluate(exp: Exp, data: Data, run: Run) -> Eval:
    """Evaluate the best model on the Test set.
    Computes final metrics using the best saved model weights,
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

    test_loader = data.test_loader  # Use Test loader

    model = run.model
    criterion = run.criterion
    params = run.params
    best_epoch = run.best_epoch
    flops = run.flops
    latency = run.latency

    print("Evaluating Best Model on Test Set...")
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # We need to re-generate these lists using the BEST model weights
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Calculate aggregate metrics for the best model
    final_test_acc = test_correct / test_total
    final_test_loss = test_loss / test_total

    precision, recall, f1 = calculate_classification_metrics(
        all_labels, all_preds)
    efficiency_score = compute_efficiency_score(final_test_acc, params)

    print(f"\nBest Test Results (Epoch {best_epoch}):")
    print(f"Accuracy: {final_test_acc:.4f}")
    print(f"Efficiency Ratio: {efficiency_score:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    wandb.log({
        "final_test_accuracy": final_test_acc,
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
            "final_test_accuracy": final_test_acc,
            "final_test_loss": final_test_loss,
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
    """Generate and save plots.
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
    test_metrics = {
        "loss": history['test_loss'],
        "accuracy": history['test_acc']
    }

    plot_metrics(train_metrics, test_metrics, "loss", output_dir)
    plot_metrics(train_metrics, test_metrics, "accuracy", output_dir)

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
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")

    main(args.config)
