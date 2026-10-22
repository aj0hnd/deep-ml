import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here
    a1, a2 = A[:, 0], A[:, 1]
    alpha, beta, gamma = a1 @ a1, a2 @ a2, a1 @ a2
    theta = 0.5 * torch.arctan2(2 * gamma, alpha - beta)

    V = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ])
    UVt = A @ V

    S = torch.norm(UVt, dim=0)
    singular = torch.diag(S)
    U = UVt @ torch.linalg.inv(singular)
    return (U, S, V.T)