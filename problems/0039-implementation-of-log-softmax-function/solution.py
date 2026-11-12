import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    scores = torch.as_tensor(scores, dtype=torch.float)
    Z = torch.log(torch.sum(torch.exp(scores)))
    return scores - Z
