from typing import Sequence

from torch import Tensor
from torch.nn import Module, ModuleList

from sublayer_connection import SublayerConnection
from utils import clones


class EncoderLayer(Module):
    """单层Encoder结构。"""

    def __init__(self, size: int, self_attention: Module, feed_forward: Module, dropout_prob: float) -> None:
        """模型初始化。

        Args:
            size: 编码器计算维度。
            self_attention: 自注意力模块。
            feed_forward: Position-wise Feed Forward模块。
            dropout_prob: Dropout概率。
        """
        super(EncoderLayer, self).__init__()
        self.self_attn: Module = self_attention
        self.feed_forward: Module = feed_forward
        self.sublayer: ModuleList = clones(SublayerConnection(size, dropout_prob), 2)
        self.size: int = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """编码器前馈计算。

        一个多头注意力模块，接一个FeedForward模块，单个模块使用残差连接包裹。

        Args:
            x: 输入张量。
            mask: 掩码张量。

        Returns:
            返回单层Encoder的计算结果。
        """
        x: Tensor = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)
