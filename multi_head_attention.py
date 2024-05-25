import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout

from utils import clones


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
    dropout: Dropout = None,
) -> tuple[Tensor, Tensor]:
    """计算Scaled Dot Product Attention。

    Args:
        query: Query张量。
        key: Key张量。
        value: Value张量。
        mask: 掩码张量。
        dropout: Dropout模块。

    Returns:
        返回计算结果张量以及Attention的权重张量。
    """
    d_k: int = query.size(-1)

    # 计算Query和Key的分数，Q x K / sqrt(d_k)
    attention_score: Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 掩码张量的填充
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e9)

    # 分数转换成softmax权重
    attention_prob: Tensor = attention_score.softmax(dim=-1)

    # 应用Dropout
    if dropout is not None:
        attention_prob = dropout(attention_prob)

    # Value加权求和
    return torch.matmul(attention_prob, value), attention_prob


class MultiHeadedAttention(Module):
    def __init__(self, h: int, d_model: int, dropout_prob: float = 0.1) -> None:
        """模型初始化。

        Args:
            h: 头数。
            d_model: 向量维度。
            dropout_prob: Dropout概率。
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # 单头的维度
        self.d_k: int = d_model // h

        # 头数
        self.h: int = h

        # 4个线性转换模块，W_Q，W_K，W_V各一个，还有一个W_O
        self.linears: ModuleList = clones(Linear(d_model, d_model), 4)

        self.attn = None
        self.dropout: Dropout = Dropout(p=dropout_prob)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """前馈计算。

        Args:
            query: Query张量。
            key: Key张量。
            value: Value张量。
            mask: 遮盖张量。

        Returns:
            返回计算结果张量。
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches: int = query.size(0)

        # 1) 对Q，K，V进行线性转换，然后进行多头的拆分
        # 输入的张量形状为(n_batches x seq_len x d_model)，需要将d_model拆分成多头，即 h x d_k
        # 转换后的张量形状为(n_batches x seq_len x h x d_k)，然后我们对张量的seq_len和d_k这两个dim进行转置，主要是为了计算方便
        # 最终的结果为(n_batches x h x seq_len x d_k)
        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 注意力计算
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 将多个头再拼回去
        # 从1)可知，输入向量的形状为(n_batches x h x seq_len x d_k)
        # 然后将seq_len和h转置，变成(n_batches x seq_len x h x d_k)
        # 然后再进行view变换，最后形状为(n_batches, seq_len, d_model)
        # 基本上就是1) 的逆向操作
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        del query
        del key
        del value

        # 4) 最后的输出结果再接一层线性转换，就是论文中的W_O
        return self.linears[-1](x)
