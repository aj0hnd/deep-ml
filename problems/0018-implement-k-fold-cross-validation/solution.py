import numpy as np
import torch
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    # Your implementation here
    samples = torch.arange(n_samples)
    if shuffle:
        np.random.shuffle(samples)

    qut, remain = n_samples // k, n_samples % k
    indices = [qut + (1 if i < remain else 0) for i in range(k)]
    
    result = []
    start = 0
    for test_size in indices:
        end = start + test_size
        test_indices = samples[start:end]
        train_indices = torch.cat([samples[:start], samples[end:]], dim=0)
        result.append((train_indices.tolist(), test_indices.tolist()))

        start = end
    return result
