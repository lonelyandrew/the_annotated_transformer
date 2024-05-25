import math

import torch
from torch import Tensor
from torch.nn import Module, Dropout


class PositionalEncoding(Module):
    """位置编码模块。"""

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000) -> None:
        """模块初始化。

        Args:
            d_model: 编码维度。
            dropout_prob: Dropout概率。
            max_len: 序列最大长度。
        """
        super(PositionalEncoding, self).__init__()
        self.dropout: Dropout = Dropout(p=dropout_prob)

        # 用0初始化位置编码的维度
        pe: Tensor = torch.zeros(max_len, d_model)

        # 位置索引，形状为(max_len, 1)
        position: Tensor = torch.arange(0, max_len).unsqueeze(1)

        # 计算分母项，使用对数计算提高数值计算的稳定性，避免数值在极端情况下的溢出
        div_term: Tensor = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)

        # 插入batch的维度，最终的形状为(1, max_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入序列。

        Returns:
            输出加上位置编码的张量结果。
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
