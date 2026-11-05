import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    poly_feats = [[] for _ in range(len(X))]
    for i, row in enumerate(X):
        for comb in combinations_with_replacement([1] + row.tolist(), degree):
            accum = 1
            for elem in comb:
                accum *= elem
            poly_feats[i].append(accum)

    poly_feats = np.array(poly_feats, dtype=np.float32)
    poly_feats.sort(axis=1)
    return poly_feats