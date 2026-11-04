import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_iterator(X, y=None, batch_size=64):
    """
    Batch iterator for a dataset represented as PyTorch tensors.

    Args:
        X: Tensor of shape (n_samples, n_features)
        y: Optional Tensor of shape (n_samples,)
        batch_size: Number of samples per batch

    Returns:
        List of Tensor batches (or [X_batch, y_batch] pairs if y is provided)
    """
    # Hint: Use DataLoader with shuffle=False to preserve sample order.
    ds = TensorDataset(X, y) if y is not None else TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    batches = []
    for batch in dl:
        if y is not None:
            batches.append(batch)
        else:
            batches.append(*batch)
    return batches