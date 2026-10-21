def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	num_features, num_samples = len(vectors), len(vectors[0])
    
    means = [sum(vec) / num_samples for vec in vectors]
    for i in range(num_features):
        for j in range(num_samples):
            vectors[i][j] -= means[i]
    
    cov = [[0] * num_features for _ in range(num_features)]
    for i in range(num_features):
        for j in range(num_features):
            if i <= j:
                cov[i][j] = (1 / (num_samples - 1)) * sum(vectors[i][k] * vectors[j][k] for k in range(num_samples))
            else:
                cov[i][j] = cov[j][i]
    return cov