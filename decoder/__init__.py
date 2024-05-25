from torch import Tensor
from torch.nn import Module, ModuleList

from layer_norm import LayerNorm
from utils import clones


class Decoder(Module):
    """解码器模块。"""

    def __init__(self, layer: Module, n: int) -> None:
        """模块初始化。

        Args:
            layer: 单层解码器模块。
            n: 堆叠层数。
        """
        super(Decoder, self).__init__()
        self.layers: ModuleList = clones(layer, n)
        self.norm: LayerNorm = LayerNorm(layer.size)

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
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
