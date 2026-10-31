import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    updated_bias = initial_bias
    updated_weights = initial_weights[:]
    mse_values = []
    for epoch in range(epochs):
        logits = features @ updated_weights + updated_bias
        logits = torch.sigmoid(logits)

        loss = torch.nn.functional.mse_loss(logits, labels)
        mse_values.append(round(loss.item(), 4))

        dmse = (2 / features.shape[0]) * (logits - labels)
        dsigmoid = dmse * logits * (1 - logits)
        dbias = torch.sum(dsigmoid)
        dweight = features.T @ dsigmoid

        updated_bias -= learning_rate * dbias
        updated_weights -= learning_rate * dweight
    return [round(weight.item(), 4) for weight in updated_weights], round(updated_bias.item(), 4), mse_values