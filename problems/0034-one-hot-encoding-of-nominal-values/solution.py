import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    if n_col is None:
        n_col = max(x) + 1

    out = torch.zeros(x.shape[0], n_col, dtype=torch.float)
    out[range(x.shape[0]), x] = 1.0
    return out
