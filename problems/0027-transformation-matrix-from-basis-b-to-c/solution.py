import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B_n, C_n = np.array(B, dtype=np.float32), np.array(C, dtype=np.float32)
    return [[round(val, 4) for val in row] for row in np.linalg.inv(C_n) @ B_n]