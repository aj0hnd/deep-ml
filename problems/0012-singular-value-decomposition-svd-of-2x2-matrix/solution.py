import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    a1, a2 = A[:, 0], A[:, 1]
    
    alpha, beta, gamma = a1 @ a1, a2 @ a2, a1 @ a2
    theta = 0.5 * np.arctan2(2 * gamma, alpha - beta)
    
    V = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    UVt = A @ V
    
    S = np.linalg.norm(UVt, axis=0)
    sin_mat = np.diag(S)
    U = UVt @ np.linalg.inv(sin_mat)
    return (U, S, V.T)