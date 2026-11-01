import torch
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = C⁻¹ B**.

    - *B*, *C* may be 2×2 or 3×3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    B_t, C_t = torch.as_tensor(B, dtype=torch.float), torch.as_tensor(C, dtype=torch.float)

    P = torch.linalg.inv(C_t) @ B_t
   
    return [[round(val.item(), 4) for val in row] for row in P]
