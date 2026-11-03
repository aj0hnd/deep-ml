import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n_exmaple = X.shape[0]
    
    batches = []
    start_idx = 0
    while start_idx < n_exmaple:
        x_batch = X[start_idx:start_idx + batch_size]
        if y is not None:
            y_batch = y[start_idx:start_idx+batch_size]
            batches.append([x_batch.tolist(), y_batch.tolist()])
        else:
            batches.append([x_batch.tolist()])
        start_idx += batch_size
    
    return batches