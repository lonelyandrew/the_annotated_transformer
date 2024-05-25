from typing import Sequence

from torch import Tensor
from torch.nn import Module, ModuleList

from sublayer_connection import SublayerConnection
from utils import clones


class EncoderLayer(Module):
    """单层Encoder结构。"""

    def __init__(self, size: int, self_attn: Module, feed_forward: Module, dropout: float) -> None:
        """模型初始化。

        Args:
            size:
            self_attn: 自注意力模块。
            feed_forward: Position-wise Feed Forward模块。
            dropout: Dropout概率。
        """
        super(EncoderLayer, self).__init__()
        self.self_attn: Module = self_attn
        self.feed_forward: Module = feed_forward
        self.sublayer: ModuleList = clones(SublayerConnection(size, dropout), 2)
        self.size: int = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Follow Figure 1 (left) for connections."""
        x: Tensor = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
