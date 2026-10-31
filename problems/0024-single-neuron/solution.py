import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_t = torch.as_tensor(features, dtype=torch.float)
    labels_t = torch.as_tensor(labels, dtype=torch.float)
    weights_t = torch.as_tensor(weights, dtype=torch.float)
    bias_t = torch.as_tensor(bias, dtype=torch.float)

    logits = torch.matmul(features_t, weights_t) + bias_t
    logits = torch.sigmoid(logits)

    mse = torch.nn.functional.mse_loss(logits, labels_t)
    return (logits * 1000 / 1000).tolist(), round(mse.item(), 4)