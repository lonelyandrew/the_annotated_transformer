from torch import Tensor
from torch.nn import ModuleList, Module

from layer_norm import LayerNorm
from utils import clones


class Encoder(Module):
    """Core encoder is a stack of n layers"""

    def __init__(self, layer: Module, n: int) -> None:
        """编码器初始化。

        Args:
            layer: 单层计算模块。
            n: 堆叠层数。
        """
        super(Encoder, self).__init__()
        self.layers: ModuleList = clones(layer, n)
        self.norm: LayerNorm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """前馈计算。

        将数据依次通过多次计算模块。

        Args:
            x: 输入张量。
            mask: 遮盖张量。

        Returns:
            返回编码结果张量。
        """
        for layer in self.layers:
            x: Tensor = layer(x, mask)
        return self.norm(x)
