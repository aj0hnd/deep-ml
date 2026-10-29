import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # Your implementation here
    data_t = torch.as_tensor(data, dtype=torch.float)
    normed_data = (data_t - torch.mean(data_t, dim=0, keepdim=True)) / torch.std(data_t, dim=0, keepdim=True)

    eigvals, eigvecs = torch.linalg.eigh(normed_data.T @ normed_data)
    top_k_eigvals = torch.argsort(eigvals, descending=True)[:k]
    principals = eigvecs[:, top_k_eigvals]
    
    fliper = torch.ones(k)
    fliper[principals[0, :] < 0] = -1
    
    return principals * fliper
