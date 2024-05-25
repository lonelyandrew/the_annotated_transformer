from torch import Tensor
from torch.nn import Module, Linear, Dropout


class PositionwiseFeedForward(Module):
    """Position-wise FeedForward模块。"""

    def __init__(self, d_model: int, d_ff: int, dropout_prob: float = 0.1) -> None:
        """模块初始化。

        Args:
            d_model: 模型维度。
            d_ff: FNN输出维度。
            dropout_prob: Dropout概率。
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1: Linear = Linear(d_model, d_ff)
        self.w_2: Linear = Linear(d_ff, d_model)
        self.dropout: Dropout = Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入张量。

        Returns:
            返回计算结果张量。
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))
