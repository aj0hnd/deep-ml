import torch
from typing import List

def divide_on_feature(X, feature_i, threshold) -> List[torch.Tensor]:
    """
    Divide the tensor X into two subsets based on whether X[:, feature_i] is greater than or equal to (or equal to) the threshold.
    Return two tensors: one with samples that meet the condition, one with samples that do not.
    """
    X_t = torch.as_tensor(X)

    accepted, rejected = [], []
    for i in range(len(X_t)):
        if X_t[i, feature_i] >= threshold:
            accepted.append(i)
        else:
            rejected.append(i)
    return [X_t[accepted], X_t[rejected]]