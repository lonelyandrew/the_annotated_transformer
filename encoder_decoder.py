from torch import Tensor, LongTensor
from torch.nn import Module

from decoder import Decoder
from encoder import Encoder
from generator import Generator


class EncoderDecoder(Module):
    """编码器-解码器架构模型。"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embedding: Module,
        target_embedding: Module,
        generator: Generator,
    ) -> None:
        """模型初始化。

        Args:
            encoder: 编码器。
            decoder: 解码器。
            source_embedding: 源序列Embedding层。
            target_embedding: 目标序列Embedding层。
            generator: 生成器。
        """
        super(EncoderDecoder, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.source_embedding: Module = source_embedding
        self.target_embedding: Module = target_embedding
        self.generator: Generator = generator

    def forward(self, source: LongTensor, target: LongTensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
        """前馈计算。

        Args:
            source: 输入序列。
            target: 输出序列。
            source_mask: 输入序列掩码张量。
            target_mask: 输出序列掩码张量。

        Returns:
            返回计算结果张量。
        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, src: LongTensor, src_mask: Tensor) -> Tensor:
        """编码器计算。

        Args:
            src: 输入序列。
            src_mask: 输入掩码张量。

        Returns:
            返回编码结果张量。
        """
        return self.encoder(self.source_embedding(src), src_mask)

    def decode(self, memory: Tensor, source_mask: Tensor, target: LongTensor, target_mask: Tensor) -> Tensor:
        """解码器计算。

        Args:
            memory: 记忆张量，即源序列的编码结果。
            source_mask: 源序列掩码张量。
            target: 目标序列。
            target_mask: 目标序列掩码张量。

        Returns:
            返回计算结果张量。
        """
        return self.decoder(self.target_embedding(target), memory, source_mask, target_mask)
