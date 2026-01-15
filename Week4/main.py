import argparse
import json
import yaml
import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms

from models.base_net import BasicCNN
from models.squeeze_net import SqueezeNet

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


def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)), # Slight zoom/crop
            transforms.RandomRotation(degrees=15),    # Rotations up to 15 degrees
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Robustness to lighting
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def main(config_path):
    # Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup WandB 
    wandb.init(project=cfg['project'], config=cfg, name=cfg['experiment_name'], allow_val_change=True)
    
    # Change the config for the sweeps
    if wandb.run.sweep_id:
        cfg = configure_sweep(cfg)
        
    # Create output folder
    output_dir = os.path.join("results", f"{cfg['experiment_name']}_{wandb.run.id}")
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(cfg['training']['seed'])

    # Data Loading
    train_dataset = ImageFolder(cfg['data']['train_dir'], transform=get_transforms(cfg['data']['img_size'], True))
    all_test_dataset = ImageFolder(cfg['data']['test_dir'], transform=get_transforms(cfg['data']['img_size'], False))

    # Split test dataset into validation and test sets
    val_percentage = cfg['data'].get('val_split', 0.7)
    val_size = int(len(all_test_dataset) * val_percentage)
    test_size = len(all_test_dataset) - val_size
    
    generator = torch.Generator().manual_seed(cfg['training']['seed'])
    val_dataset, test_dataset = random_split(all_test_dataset, [val_size, test_size], generator=generator)

    print(f"Data Split: {len(train_dataset)} Train | {len(val_dataset)} Val | {len(test_dataset)} Test")

    # Data Loaders    
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['data']['num_workers'])

    classes = train_dataset.classes

    # Model Setup
    # Add more statements if there are more models
    if cfg['model']['name'] == "BasicCNN":
        model = BasicCNN(num_classes=len(classes), img_size=cfg['data']['img_size'], config=cfg['model'])
    elif cfg['model']['name'] == "SqueezeNet":
        model = SqueezeNet(num_classes=len(classes), feature_extraction=cfg['model']['feature_extraction'])
    else:
        raise ValueError(f"Model {cfg['model']['name']} not implemented")
        
    model = model.to(device)
    
    # Model Summary
    print("Model Summary:")
    img_size = cfg['data']['img_size']
    summary(model, (3, img_size, img_size))

    # Metrics Calculation (Pre-training)
    params, flops, latency = get_model_summary(model, (3, cfg['data']['img_size'], cfg['data']['img_size']), device)
    print(f"Model Params: {params/1e6:.2f}M | FLOPs: {flops/1e9:.2f}G | Latency: {latency:.2f}ms")
    wandb.log({"Parameters": params, "FLOPs": flops, "Latency_ms": latency})

    # Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # History containers (for plotting later)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Log History
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc
        })

    # Post-Training Evaluation
    precision, recall, f1 = calculate_classification_metrics(all_labels, all_preds)
    efficiency_score = compute_efficiency_score(val_epoch_acc, params)
    
    print(f"\nFinal Results:")
    print(f"Efficiency Ratio (Acc/Params*100k): {efficiency_score:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    wandb.log({
        "final_val_accuracy": val_epoch_acc,
        "efficiency_score": efficiency_score,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    
    # Prepare results dictionary
    results_data = {
        "configuration": cfg,
        "metrics": {
            "final_val_accuracy": val_epoch_acc,
            "final_val_loss": val_epoch_loss,
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

    # Generate and Save Plots
    train_metrics = {
        "loss": history['train_loss'], 
        "accuracy": history['train_acc']
    }
    val_metrics = {
        "loss": history['val_loss'], 
        "accuracy": history['val_acc']
    }

    # Plot Loss and Accuracy Curves
    plot_metrics(train_metrics, val_metrics, "loss", output_dir)
    plot_metrics(train_metrics, val_metrics, "accuracy", output_dir)

    # Plot Confusion Matrix and ROC Curve
    plot_confusion_matrix(all_labels, all_preds, classes, output_dir)
    plot_roc_curve(all_labels, all_probs, classes, output_dir)
    
    # Plot Computational Graph (requires model input size)
    # We create a dummy input size tuple: (Batch_Size, Channels, Height, Width)
    input_dims = (1, 3, cfg['data']['img_size'], cfg['data']['img_size'])
    plot_computational_graph(model, input_dims, output_dir)
    
    # GradCAM not needed for the moment
    
    # Save Model
    model_save_path = os.path.join(output_dir, "model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)