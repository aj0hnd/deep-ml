import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	def determinant(matrix: list[list[int|float]]):
        return np.linalg.det(np.array(matrix))
    
    if not (determinant(T) and determinant(S)):
        return -1

    A_n, T_n, S_n = np.array(A), np.array(T), np.array(S)
    return (np.linalg.inv(T_n) @ A_n @ S_n).tolist()