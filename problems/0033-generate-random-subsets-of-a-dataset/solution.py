import numpy as np
import torch
from typing import List, Tuple

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    # Your code here
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)

    n_rows = X_t.shape[0]
    n_elems = n_rows if replacements else n_rows // 2
    results = []
    for _ in range(n_subsets):
        indices = np.random.choice(n_rows, n_elems, replace=replacements)

        X_subset = X_t[indices]
        y_subset = y_t[indices]
        results.append((X_subset.tolist(), y_subset.tolist()))

    return results