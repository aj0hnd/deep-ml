import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Your code here
    n_example = X.shape[0]
    X_norm = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True, ddof=1)
    if Y is not None:
        Y_norm = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True, ddof=1)
    else:
        Y_norm = X_norm
    
    corr = (X_norm.T @ Y_norm) / (n_example - 1)
    return corr