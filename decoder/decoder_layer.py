from torch import Tensor
from torch.nn import Module, ModuleList

from sublayer_connection import SublayerConnection
from utils import clones


class DecoderLayer(Module):
    """单层的解码器模块。"""

    def __init__(
        self,
        size: int,
        self_attention: Module,
        source_attention: Module,
        feed_forward: Module,
        dropout_prob: float,
    ):
        """模块初始化。

        Args:
            size: 编码器计算维度。
            self_attention: 目标数据的自注意力模块。
            source_attention: 源数据的注意力模块。
            feed_forward: Position-wise Feed Forward模块。
            dropout_prob: Dropout概率。
        """
        super(DecoderLayer, self).__init__()
        self.size: int = size
        self.self_attention: Module = self_attention
        self.source_attention: Module = source_attention
        self.feed_forward: Module = feed_forward
        self.sublayer: ModuleList = clones(SublayerConnection(size, dropout_prob), 3)

    def forward(self, x: Tensor, memory: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入张量。
            memory: 记忆张量。
            source_mask: 源序列掩码张量。
            target_mask: 目标序列掩码张量。

        Returns:
            返回计算结果张量。
        """
        m: Tensor = memory

        # 目标序列自注意力模块计算
        x = self.sublayer[0](x, lambda y: self.self_attention(y, y, y, target_mask))

        # 源序列注意力模块计算
        x = self.sublayer[1](x, lambda y: self.source_attention(y, m, m, source_mask))

        # Position-wise Feedforward模块计算
        return self.sublayer[2](x, self.feed_forward)
