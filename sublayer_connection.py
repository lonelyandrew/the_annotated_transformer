from torch import Tensor
from torch.nn import Dropout, Module

from layer_norm import LayerNorm


class SublayerConnection(Module):
    """A residual connection followed by a layer norm.

    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        """初始化残差连接层。

        Args:
            size: 单层输出结果张量的大小。
            dropout: Dropout概率。
        """
        super(SublayerConnection, self).__init__()
        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: Dropout = Dropout(p=dropout)

    def forward(self, x: Tensor, sublayer) -> Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
