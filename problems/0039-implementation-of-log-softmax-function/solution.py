import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    Z = np.log(np.sum(np.exp(scores)))
    return scores - Z