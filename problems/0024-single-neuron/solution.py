import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    mse, n_features = 0.0, len(features)
    probabilities = []
    for (feature, label) in zip(features, labels):
        logits = 0.0
        for i in range(len(weights)):
            logits += feature[i] * weights[i]
        logits += bias

        logits = sigmoid(logits)
        probabilities.append(logits)

        mse += (logits - label) ** 2

    mse /= n_features
    return probabilities, mse