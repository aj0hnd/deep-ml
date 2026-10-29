import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores_t = torch.as_tensor(scores, dtype=torch.float)
    scores_t -= torch.max(scores_t)
    return F.softmax(torch.tensor(scores_t), dim=0).tolist()
