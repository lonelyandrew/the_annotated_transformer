from torch import Tensor
from torch.nn import Module, Linear, Dropout


class PositionwiseFeedForward(Module):
    """Implements FFN equation."""

    def __init__(self, d_model: int, d_ff: int, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1: Linear = Linear(d_model, d_ff)
        self.w_2: Linear = Linear(d_ff, d_model)
        self.dropout: Dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))
