import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    n_rows = X.shape[0]
    n_elems = n_rows if replacements else n_rows // 2

    results = []
    for _ in range(n_subsets):
        indices = np.random.choice(n_rows, n_elems, replace=replacements)
        X_subsets = X[indices]
        y_subsets = y[indices]
        results.append((X_subsets.tolist(), y_subsets.tolist()))

    return results