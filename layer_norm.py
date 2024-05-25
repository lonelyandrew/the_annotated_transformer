import torch
from torch import Tensor
from torch.nn import Module, Parameter


class LayerNorm(Module):
    """Layer Normalization模块。"""

    def __init__(self, feature_size: int, eps: float = 1e-6) -> None:
        """模块初始化。

        Args:
            feature_size: 特征大小。
            eps: Epsilon参数。
        """
        super(LayerNorm, self).__init__()
        # 缩放参数
        self.a_2: Parameter = Parameter(torch.ones(feature_size))

        # 平移参数
        self.b_2: Parameter = Parameter(torch.zeros(feature_size))

        # 容错参数
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入张量。

        Returns:
            返回计算结果张量。
        """
        mean: Tensor = x.mean(-1, keepdim=True)
        std: Tensor = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
