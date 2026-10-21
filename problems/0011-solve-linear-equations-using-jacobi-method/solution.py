import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	D = np.diag(A.diagonal())
    LU = A - D
    
    x = np.zeros_like(b)
    for _ in range(n):
        x = np.linalg.inv(D) @ (b - LU @ x)

    return np.round(x, 4)