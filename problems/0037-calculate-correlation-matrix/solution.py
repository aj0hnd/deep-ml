import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    # Your implementation here
    X = torch.as_tensor(X, dtype=torch.float)
    if Y is not None:
        Y = torch.as_tensor(Y, dtype=torch.float)
    
    n_example = X.shape[0]
    X_norm = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    if Y is not None:
        Y_norm = (Y - Y.mean(dim=0, keepdim=True)) / Y.std(dim=0, keepdim=True)
    else:
        Y_norm = X_norm

    corr = (X_norm.T @ Y_norm) / (n_example - 1)
    return corr