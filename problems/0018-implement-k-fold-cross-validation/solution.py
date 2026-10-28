import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    samples = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(samples)

    qut = n_samples // k
    remain = n_samples % k
    indices = [qut + (1 if i < remain else 0) for i in range(k)]
    
    results = []
    for i in range(k):
        test_s = sum(elem for elem in indices[:i])
        test_e = test_s + indices[i]
        test_indices = samples[test_s:test_e]
        train_indices = np.concat([samples[:test_s], samples[test_e:n_samples]], axis=0)
        results.append((train_indices.tolist(), test_indices.tolist()))
    return results