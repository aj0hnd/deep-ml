import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    normed_data = (data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True)
    cov_mat = normed_data.T @ normed_data
    eigvals, eigvecs = np.linalg.eigh(cov_mat)

    top_k_eigvals = np.argsort(eigvals, descending=True)[:k]
    principals = eigvecs[:, top_k_eigvals]
    convert = np.ones(k)
    convert[principals[0, :] < 0] = -1
    
    return np.round(principals * convert, decimals=4)
