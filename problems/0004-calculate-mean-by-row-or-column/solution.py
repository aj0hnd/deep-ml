def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	assert mode in ['row', 'column'], "mode only possible with 'row' or 'column'."
    if mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    elif mode == 'column':
        return [sum(col) / len(col) for col in zip(*matrix)]