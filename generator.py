from torch import log_softmax, Tensor
from torch.nn import Linear, Module


class Generator(Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model: int, vocab: int) -> None:
        """生成器初始化。

        Args:
            d_model: 模型表征向量维度。
            vocab: 词汇表大小。
        """
        super(Generator, self).__init__()
        self.proj: Linear = Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(self.proj(x), dim=-1)
