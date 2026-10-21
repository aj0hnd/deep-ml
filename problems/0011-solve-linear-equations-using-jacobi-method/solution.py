import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    
    D = torch.diag(A_t.diag())
    LU = A_t - D
    
    x = torch.zeros_like(b_t)
    for _ in range(n):
        x = torch.inverse(D) @ (b_t - LU @ x)
    return x
