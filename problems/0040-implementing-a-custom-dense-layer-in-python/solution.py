import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# DO NOT CHANGE SEED (used for deterministic init)
np.random.seed(42)

def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

class Layer:
    def set_input_shape(self, shape: Tuple[int, ...]):
        self.input_shape = shape

    def layer_name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> int:
        return 0

    def forward_pass(self, X, training: bool = True):
        raise NotImplementedError

    def backward_pass(self, accum_grad):
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, n_units: int, input_shape: tuple[int, ...] | None = None):
        self.n_units = n_units
        self.layer_input = None
        self.input_shape = input_shape

        self.trainable = None
        self.W = None # torch.Tensor with requires_grad=True
        self.b = None # torch.Tensor with requires_grad=True
        self._opt = None

    def initialize(self, optimizer) -> None:
        """
        Initialize weights with a uniform distribution and biases with zeros.
        
        Hint: use numpy to create deterministic arrays, then convert to torch tensors.
        also store the provided optimizer for updates.
        """
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = torch.tensor(np.random.uniform(-limit, limit, size=(self.input_shape[0], self.n_units)), dtype=torch.float, requires_grad=True)
        self.b = torch.zeros(1, self.n_units, dtype=torch.float, requires_grad=True)
        self._opt = optimizer
        print(self.W.shape, self.b.shape)

    def number_of_parameters(self) -> int:
        """Return total number of trainable parameters in W and b"""
        total = 0
        if self.W.requires_grad:
            total += self.W.numel()
        if self.b.requires_grad:
            total += self.b.numel()
        return total

    def forward_pass(self, X, training: bool = True):
        """Use torch.nn.funcional.linear for the forward pass (X @ W + b)."""
        self.layer_input = _to_tensor(X).requires_grad_(True)
        if self.trainable is None:
            self.trainable = training
        self.out = F.linear(self.layer_input, self.W.T, self.b)
        return self.out

    def backward_pass(self, accum_grad):
        """
        Use torch.autograd.grad to get dL/dW, dL/db given upstream gradient.
        Then update params with the provided optimizer and return grad w.r.t input.
        Hint: grad_input can be computed with a matmul against W.T
        """
        dW, db, dinput = torch.autograd.grad(
            outputs=self.out,
            inputs=(self.W, self.b, self.layer_input),
            grad_outputs=_to_tensor(accum_grad).requires_grad_(True)
        )
        self.W = self._opt.update(self.W, dW)
        self.b = self._opt.update(self.b, db)
        return dinput