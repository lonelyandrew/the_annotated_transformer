from torch import Tensor
from torch.nn import Dropout, Module

from layer_norm import LayerNorm


class SublayerConnection(Module):
    """残差连接模块。"""

    def __init__(self, size: int, dropout_prob: float) -> None:
        """初始化残差连接。

        Args:
            size: 单层输出结果张量的大小。
            dropout_prob: Dropout概率。
        """
        super(SublayerConnection, self).__init__()

        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: Dropout = Dropout(p=dropout_prob)

    def forward(self, x: Tensor, sublayer: Module) -> Tensor:
        """前馈计算。

        Args:
            x: 输入张量。
            sublayer: 子层模块。

        Returns:
            返回计算结果张量。
        """
        # NOTE: 论文中的计算顺序是先进行sublayer的计算，再计算结果的norm，叫作post-norm
        # 代码实现中的计算顺序是再input上直接进行norm计算，叫作pre-norm
        # Ref: https://stackoverflow.com/questions/77864704/annotated-transformer-why-x-dropoutsublayerlayernormx
        return x + self.dropout(sublayer(self.norm(x)))
