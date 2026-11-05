import torch
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial feature combinations
    (with constant term), sorted for each sample from smallest to largest.
    """
    # Hint: Use combinations_with_replacement and torch.prod.
    poly_feats = [[torch.prod(torch.tensor(comb)) for comb in combinations_with_replacement([1] + row.tolist(), degree)] for row in X]
    poly_feats = torch.tensor(poly_feats, dtype=torch.float)
    poly_feats.sort(dim=1)
    return poly_feats
