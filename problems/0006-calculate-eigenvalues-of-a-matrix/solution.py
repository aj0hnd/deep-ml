def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	trace = sum(matrix[i][i] for i in range(2))
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    discrim = trace ** 2 - 4 * det
    return [0.5 * (trace + discrim ** 0.5), 0.5 * (trace - discrim ** 0.5)]