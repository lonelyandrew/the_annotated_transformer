from torch import Tensor, LongTensor
from torch.nn import Module


class EncoderDecoder(Module):
    """编码器-解码器架构模型。"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        """模型初始化。

        Args:
            encoder: 编码器。
            decoder: 解码器。
            src_embed: 源数据表征。
            tgt_embed: 目标数据表征。
            generator: 生成器。
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: LongTensor, tgt: LongTensor, src_mask, tgt_mask) -> Tensor:
        """前馈计算。

        Args:
            src: 输入序列。
            tgt: 输出序列。
            src_mask: 输入遮盖。
            tgt_mask: 输出遮盖。

        Returns:
            返回计算结果张量。
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask) -> Tensor:
        """编码器计算。

        Args:
            src: 输入序列。
            src_mask: 输入遮盖。

        Returns:
            返回编码结果张量。
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask) -> Tensor:
        """解码器计算。

        Args:
            memory:
            src_mask:
            tgt:
            tgt_mask:

        Returns:

        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
