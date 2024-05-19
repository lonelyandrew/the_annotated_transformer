import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout

from utils import clones


def attention(query, key, value, mask=None, dropout=None) -> tuple[Tensor, Tensor]:
    """计算Scaled Dot Product Attention。

    Args:
        query:
        key:
        value:
        mask:
        dropout:

    Returns:
        返回计算结果张量。
    """
    d_k: int = query.size(-1)
    scores: Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        """模型初始化。

        Args:
            h: 头数。
            d_model: 向量维度。
            dropout: Dropout概率。
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k: int = d_model // h
        self.h: int = h
        self.linears: ModuleList = clones(Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = Dropout(p=dropout)

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
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batches: int = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)
