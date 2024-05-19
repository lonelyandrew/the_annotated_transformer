from copy import deepcopy

from torch.nn import Sequential
from torch.nn.init import xavier_uniform_

from decoder import Decoder
from decoder.decoder_layer import DecoderLayer
from embedding import Embeddings
from encoder import Encoder
from encoder.encoder_decoder import EncoderDecoder
from encoder.encoder_layer import EncoderLayer
from generator import Generator
from multi_head_attention import MultiHeadedAttention
from positional_encoding import PositionalEncoding
from positionwise_feedforward import PositionwiseFeedForward


def make_model(
    src_vocab,
    tgt_vocab,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """构建Transformer模型。

    model -> encoder + decoder

    Args:
        src_vocab: 源数据词汇表的大小。
        tgt_vocab: 目标数据词汇表的大小。
        n: 模型层数。
        d_model: 表征向量的维度。
        d_ff: 前馈层的维度。
        h: 多头注意力的头数。
        dropout: Dropout概率。

    Returns:
        返回构建完成的模型。
    """
    multi_head_attention: MultiHeadedAttention = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(multi_head_attention), deepcopy(ff), dropout), n),
        Decoder(
            DecoderLayer(
                d_model, deepcopy(multi_head_attention), deepcopy(multi_head_attention), deepcopy(ff), dropout
            ),
            n,
        ),
        Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
    return model
