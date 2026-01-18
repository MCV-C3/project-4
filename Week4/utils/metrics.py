import torch
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
try:
    from thop import profile
except ImportError:
    profile = None

def get_model_summary(model, input_size, device):
    """
    Returns: params (int), flops (float), inference_time_ms (float)
    """
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 1. Parameters
    params = sum(p.numel() for p in model.parameters()) #  if p.requires_grad
    
    # 2. FLOPs (using thop if available)
    flops = 0.0
    if profile:
        flops, _ = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # Cleanup thop hooks to prevent issues in subsequent forward passes
        for m in model.modules():
            if hasattr(m, 'total_ops'):
                del m.total_ops
            if hasattr(m, 'total_params'):
                del m.total_params
            m._forward_hooks.clear()
    
    # 3. Latency (Average of 100 runs)
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()
    
    inference_time_ms = ((end_time - start_time) / 100) * 1000
    
    return params, flops, inference_time_ms

def compute_efficiency_score(accuracy, params):
    """
    Metric: Accuracy / (Parameters / 100K)
    """
    if params == 0: return 0
    return accuracy / (params / 100000.0)

def calculate_classification_metrics(y_true, y_pred):
    """
    Returns precision, recall, f1 (weighted average for multi-class)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1