import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    # Your implementation here
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)

    accur = (y_true == y_pred).sum() / y_true.shape[0]
    return accur.item()
