from torch import log_softmax, Tensor
from torch.nn import Linear, Module


class Generator(Module):
    """生成器模块。

    生成器将解码器的计算结果映射到目标词汇表的one-hot向量上。
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """生成器初始化。

        Args:
            d_model: 模型表征向量维度。
            vocab_size: 词汇表大小。
        """
        super(Generator, self).__init__()
        self.proj: Linear = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """前馈计算。

        Args:
            x: 输入张量。

        Returns:
            返回计算结果张量。
        """
        return log_softmax(self.proj(x), dim=-1)
