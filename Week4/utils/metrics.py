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

    # Parameters
    params = sum(p.numel() for p in model.parameters())

    # FLOPs (using thop if available)
    flops = 0.0
    if profile:
        flops, _ = profile(model, inputs=(dummy_input, ), verbose=False)

        # Cleanup thop hooks
        for m in model.modules():
            if hasattr(m, 'total_ops'):
                del m.total_ops
            if hasattr(m, 'total_params'):
                del m.total_params
            m._forward_hooks.clear()

    # Latency (Average of 100 runs)
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
    if params == 0:
        return 0
    return accuracy / (params / 100000.0)


def calculate_classification_metrics(y_true, y_pred):
    """
    Returns precision, recall, f1 (weighted average for multi-class)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1


def count_nonzero_params(model):
    """
    Counts the total number of non-zero parameters in the model.
    Handles pruned models (where weight is a computed attribute, not a parameter).
    """
    nonzero_params = 0
    total_params = 0

    # Use a set to track processed parameter IDs to handle standard parameters
    processed_params = set()

    for module in model.modules():
        # Check standard layers
        if isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
            # Check weight
            if hasattr(module, 'weight') and module.weight is not None:
                # If module is pruned, 'weight' is an attribute (tensor), 'weight_orig' is the parameter
                # We want to count non-zeros in 'weight' (effective)
                w = module.weight
                nonzero_params += torch.count_nonzero(w).item()
                total_params += w.numel()

                # Mark associated parameter as processed
                if hasattr(module, 'weight_orig'):
                    processed_params.add(id(module.weight_orig))
                else:
                    processed_params.add(id(module.weight))

            # Check bias
            if hasattr(module, 'bias') and module.bias is not None:
                b = module.bias
                nonzero_params += torch.count_nonzero(b).item()
                total_params += b.numel()
                processed_params.add(id(module.bias))

    # Iterate remaining parameters (e.g. BatchNorm, Embeddings, etc.)
    for param in model.parameters():
        if id(param) not in processed_params:
            nonzero_params += torch.count_nonzero(param).item()
            total_params += param.numel()
            processed_params.add(id(param))

    return nonzero_params, total_params
