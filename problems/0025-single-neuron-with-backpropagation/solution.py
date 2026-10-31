import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	updated_weights = initial_weights[:]
    updated_bias = initial_bias
    n_features = features.shape[0]
    mse_values = []
    for epoch in range(epochs):
        logits = features @ updated_weights + updated_bias
        logits = 1 / (1 + np.exp(-logits))

        loss = np.sum((logits - labels) ** 2) / n_features
        mse_values.append(round(loss.item(), 4))

        dmse = (2 / n_features) * (logits - labels) 
        dsig = dmse * logits * (1 - logits)
        dbias = np.sum(dsig)
        dweight = features.T @ dsig

        updated_bias -= learning_rate * dbias
        updated_weights -= learning_rate * dweight
    return [round(weight.item(), 4) for weight in updated_weights], round(updated_bias.item(), 4), mse_values