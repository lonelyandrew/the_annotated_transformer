import math

from torch import Tensor, LongTensor
from torch.nn import Module, Embedding


class Embeddings(Module):
    def __init__(self, d_model: int, vocab: int) -> None:
        """初始化词嵌入。

        Args:
            d_model: 嵌入向量维度。
            vocab: 词汇表大小。
        """
        super(Embeddings, self).__init__()
        self.lut: Embedding = Embedding(vocab, d_model)
        self.d_model: int = d_model

    def forward(self, x: LongTensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入序列。

        Returns:
            返回嵌入结果张量。
        """
        return self.lut(x) * math.sqrt(self.d_model)
