import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    standardized_data = torch.round((data_t - data_t.mean(dim=0, keepdim=True)) / data_t.std(dim=0, unbiased=False, keepdim=True), decimals=4)
    
    scaler = data_t.max(dim=0, keepdim=True).values - data_t.min(dim=0, keepdim=True).values
    normalized_data = torch.round((data_t - data_t.min(dim=0, keepdim=True).values) / scaler, decimals=4)
    return (standardized_data, normalized_data)
