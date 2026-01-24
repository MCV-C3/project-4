
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import copy
import wandb

import models
from utils.metrics import count_nonzero_params, calculate_classification_metrics, compute_efficiency_score
from main import get_transforms, build_model_from_config
from utils.plots import plot_metrics, plot_confusion_matrix, plot_roc_curve


def get_prunable_layers(model, layer_types):
    """Returns a list of (module, name) tuples for pruning."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            parameters_to_prune.append((module, 'weight'))
    return parameters_to_prune


def constant_sparsity_pruning(model, amount, method='unstructured'):
    """
    Applies pruning to the model.
    For 'unstructured': uses Global Unstructured L1.
    For 'structured': uses Structured L1 (Channel-wise) on Conv2d.
    """
    if method == 'unstructured':
        layer_types = (nn.Conv2d, nn.Linear)
        parameters_to_prune = get_prunable_layers(model, layer_types)

        print(
            f"  [Pruning] Global Unstructured L1 on {len(parameters_to_prune)} layers (amount={amount:.4f})...")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    elif method == 'structured':
        layer_types = (nn.Conv2d,)
        print(
            f"  [Pruning] Structured L1 (Output Channels) on Conv2d layers (amount={amount:.4f})...")
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                # Prune output channels (dim=0)
                prune.ln_structured(module, name="weight",
                                    amount=amount, n=1, dim=0)
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def make_pruning_permanent(model):
    """Removes pruning re-parametrization to make weights permanently zero."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')


def finetune(model, train_loader, criterion, optimizer, device, epochs=1):
    """
    Fine-tunes the model for a few epochs.
    """
    model.train()
    print(f"  [Fine-tuning] Starting for {epochs} epochs...")

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

    return history


def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Capture probabilities for ROC
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    accuracy = correct / total
    loss_val = test_loss / total
    precision, recall, f1 = calculate_classification_metrics(
        all_labels, all_preds)

    return accuracy, loss_val, precision, recall, f1, all_preds, all_labels, all_probs


def main(args):
    # WandB Init
    wandb.init(project="week4_pruning", entity="c3-team4")

    # Map model names to paths for Sweep convenience
    MODEL_PATHS = {
        "MobileNetV2": "/ghome/group04/C3/Benet/project-4/Week4/results/best_hypopt_models/MobileNet_Distill_Batch_16_LR_0.1_Opt_SGD_Sched_CosineAnnealingLR_r6zen7t9",
        "ShuffleNet": "/ghome/group04/C3/Benet/project-4/Week4/results/best_hypopt_models/ShuffleNet_Distill_Batch_16_LR_0.01_Opt_SGD_Sched_CosineAnnealingLR_vqbx5ztp"
    }

    # Override args with WandB config if available (Sweep)
    if wandb.config:
        for key, value in wandb.config.items():
            if key == 'model_name' and value in MODEL_PATHS:
                args.model_dir = MODEL_PATHS[value]
                print(
                    f"WandB: Mapped model_name '{value}' to '{args.model_dir}'")
            elif hasattr(args, key):
                setattr(args, key, value)
                print(f"WandB: Overriding arg '{key}' with '{value}'")

    if args.model_dir is None:
        raise ValueError(
            "args.model_dir is not set. Please provide --model_dir or ensure model_name is mapped in wandb.config")

    # Construct descriptive experiment name
    # Format: Modelname_method_amountX_stepsY_epochsZ
    # Example: MobileNetV2_unstructured_amount0.2_steps5_epochs1
    # Check if 'model_name' is in wandb.config or args
    model_name_str = "UnknownModel"
    if args.model_name:
        model_name_str = args.model_name
    elif wandb.config and 'model_name' in wandb.config:
        model_name_str = wandb.config.model_name

    experiment_name = (f"{model_name_str}_{args.method}_amount{args.amount}_"
                       f"steps{args.iterative_steps}_epochs{args.finetune_epochs}")

    # Update WandB run name
    wandb.run.name = experiment_name
    print(f"Set WandB Run Name to: {experiment_name}")

    # 1. Load Experiment Config
    result_json_path = os.path.join(args.model_dir, "experiment_results.json")
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(f"Could not find {result_json_path}")

    with open(result_json_path, 'r') as f:
        results = json.load(f)

    cfg = results['configuration']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Rebuild Model
    num_classes = cfg['data']['num_classes']
    model = build_model_from_config(cfg, num_classes=num_classes)
    model.to(device)

    # 3. Load Best Weights
    best_model_path = os.path.join(args.model_dir, "best_model.pth")
    print(f"Loading weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 4. Prepare Data Loaders
    print("Preparing Data Loaders...")
    # Check if train_dir exists locally, might need adjustment if paths are different
    if not os.path.exists(cfg['data']['train_dir']):
        print(
            f"Warning: Train dir {cfg['data']['train_dir']} not found. Fine-tuning might fail if not fixed.")

    train_dataset = ImageFolder(
        cfg['data']['train_dir'],
        transform=get_transforms(cfg['data']['img_size'], True))
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'])

    test_dataset = ImageFolder(
        cfg['data']['test_dir'],
        transform=get_transforms(cfg['data']['img_size'], False))
    test_loader = DataLoader(
        test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'])

    # 5. Baseline Evaluation
    print("Evaluating Baseline...")
    base_acc, base_loss, _, _, _, _, _, _ = evaluate_model(
        model, test_loader, device)
    _, total_params_original = count_nonzero_params(model)
    print(
        f"Baseline Results: Acc: {base_acc:.4f} | Params: {total_params_original}")

    # 6. Iterative Pruning Loop
    final_amount = args.amount
    steps = args.iterative_steps
    finetune_epochs = args.finetune_epochs

    # Optimizer for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"\nStarting Iterative Pruning ({steps} steps, target amount={final_amount}, finetune_epochs={finetune_epochs})...")

    history = []

    # Calculate amount per step to reach cumulative target
    step_target = final_amount / steps

    for step in range(1, steps + 1):
        print(f"\n--- Step {step}/{steps} ---")

        current_amount = step_target * step

        # Prune
        constant_sparsity_pruning(model, current_amount, args.method)

        # Check sparsity
        nonzero, total = count_nonzero_params(model)
        sparsity = 1.0 - (nonzero / total)
        print(f"  Sparsity after pruning: {sparsity:.2%}")

        # Fine-tune
        if finetune_epochs > 0:
            step_history = finetune(model, train_loader, criterion,
                                    optimizer, device, epochs=finetune_epochs)
            total_history['train_loss'].extend(step_history['train_loss'])
            total_history['train_acc'].extend(step_history['train_acc'])

        # Evaluate
        acc, loss, prec, rec, f1, _, _, _ = evaluate_model(
            model, test_loader, device)
        print(f"  Result Step {step}: Acc: {acc:.4f}")

        history.append({
            "step": step,
            "sparsity": sparsity,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        # Log to WandB
        wandb.log({
            "step": step,
            "sparsity": sparsity,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "current_amount": current_amount
        })

    # 7. Final Results
    # Make temporary pruning permanent before final count and save
    make_pruning_permanent(model)

    final_nonzero, _ = count_nonzero_params(model)
    final_sparsity = 1.0 - (final_nonzero / total_params_original)

    print(f"\nFinal Results (Sparsity {final_sparsity:.2%}):")
    print(f"Accuracy: {acc:.4f}")

    results_data = {
        "method": args.method,
        "mode": "iterative" if steps > 1 else "one-shot",
        "target_amount": args.amount,
        "steps": steps,
        "finetune_epochs": finetune_epochs,
        "learning_rate": args.lr,
        "original_params": total_params_original,
        "final_nonzero_params": final_nonzero,
        "final_sparsity": final_sparsity,
        "baseline_accuracy": base_acc,
        "final_accuracy": acc,
        "history": history
    }

    # Determine Output Directory
    if wandb.run.sweep_id:
        # Create a unique directory for this run within the sweep folder
        run_name = wandb.run.name or wandb.run.id
        output_dir = os.path.join(
            "results", "sweeps", wandb.run.sweep_id, run_name)
    else:
        # Standard run: save in the user-provided model_dir (legacy behavior) or create a new one?
        # User asked for "results/sweeps/<id>" like main.py.
        # For non-sweep runs, preserving current behavior (saving in model_dir) is safer unless asked otherwise.
        # But if we want consistent "results/..." behavior:
        output_dir = args.model_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # 8. Save
    # Save Model
    output_model_name = f"pruned_{args.method}_{args.amount}_steps{steps}_ft{finetune_epochs}.pth"
    output_model_path = os.path.join(output_dir, output_model_name)
    torch.save(model.state_dict(), output_model_path)
    print(f"Saved pruned model to {output_model_path}")

    with open(output_json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved metrics to {output_json_path}")

    # 9. Generate Plots & Save Experiment Results (Standard Format)
    print("Generating plots...")

    # Get final evaluation details
    final_acc, final_loss, final_prec, final_rec, final_f1, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device)

    # Plot Metrics (Learning Curves)
    # We only have aggregated training history.
    # For test, we didn't track it per-epoch inside finetune, so we pass empty dict or just train.
    # plot_metrics expects specific keys.
    if total_history['train_loss']:  # Only if we did finetuning
        plot_metrics_data_train = {
            "loss": total_history['train_loss'], "accuracy": total_history['train_acc']}
        plot_metrics_data_test = {}  # Not available per-epoch

        plot_metrics(plot_metrics_data_train,
                     plot_metrics_data_test, "loss", output_dir)
        plot_metrics(plot_metrics_data_train,
                     plot_metrics_data_test, "accuracy", output_dir)

    # Plot Confusion Matrix & ROC
    # Need class names. `test_loader.dataset.classes`
    classes = test_loader.dataset.classes
    plot_confusion_matrix(all_labels, all_preds, classes, output_dir)
    plot_roc_curve(all_labels, all_probs, classes, output_dir)

    # Save experiment_results.json (Standard Format for consistency)
    standard_results = {
        "configuration": cfg,  # Original config
        "pruning_config": vars(args),  # Pruning args
        "metrics": {
            "final_test_accuracy": final_acc,
            "final_test_loss": final_loss,
            "precision": final_prec,
            "recall": final_rec,
            "f1_score": final_f1,
            "model_params": total_params_original,  # Original
            "final_nonzero_params": final_nonzero,
            "final_sparsity": final_sparsity
        }
    }

    std_json_path = os.path.join(output_dir, "experiment_results.json")
    with open(std_json_path, 'w') as f:
        json.dump(standard_results, f, indent=4)
    print(f"Saved standard experiment results to {std_json_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=False, default=None,
                        help="Path to the directory containing experiment_results.json and best_model.pth")
    parser.add_argument("--model_name", type=str, required=False, default=None,
                        help="Model name (used for WandB sweeps)")
    parser.add_argument("--method", type=str, choices=['structured', 'unstructured'], default='unstructured',
                        help="Pruning method")
    parser.add_argument("--amount", type=float, default=0.2,
                        help="Total amount of pruning (0.0 to 1.0)")
    parser.add_argument("--iterative_steps", type=int, default=1,
                        help="Number of pruning steps")
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Epochs to fine-tune after each pruning step")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")

    args = parser.parse_args()
    main(args)
