import math

import torch
from torch import Tensor
from torch.nn import Module, Dropout


class PositionalEncoding(Module):
    """Implement the PE function."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
