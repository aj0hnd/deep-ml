import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col is None:
        n_col = max(x) + 1
        
    out = np.zeros((x.shape[0], n_col), dtype=np.float32)
    out[range(x.shape[0]), x] = 1.0
    return out