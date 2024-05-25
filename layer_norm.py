import torch
from torch import Tensor
from torch.nn import Module, Parameter


class LayerNorm(Module):
    """Layer Normalization模块。"""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2: Parameter = Parameter(torch.ones(features))
        self.b_2: Parameter = Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        mean: Tensor = x.mean(-1, keepdim=True)
        std: Tensor = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
