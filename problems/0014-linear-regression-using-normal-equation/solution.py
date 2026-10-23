import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_n, y_n = np.array(X), np.array(y).reshape(-1, 1)
    # theta = (A^TA)^-1b^TA
    return np.round(np.linalg.inv(X_n.T @ X) @ X_n.T @ y_n).squeeze().tolist()
