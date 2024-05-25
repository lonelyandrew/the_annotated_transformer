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
    src_vocab: int,
    tgt_vocab: int,
    layer_count: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """构建Transformer模型。

    Args:
        src_vocab: 源数据词汇表的大小。
        tgt_vocab: 目标数据词汇表的大小。
        layer_count: 模型层数。
        d_model: 表征向量的维度。
        d_ff: 前馈层的维度。
        h: 多头注意力的头数。
        dropout: Dropout概率。

    Returns:
        返回构建完成的模型。
    """
    # 多头注意力模块
    multi_head_attention: MultiHeadedAttention = MultiHeadedAttention(h, d_model)

    # Position-wise Feedforward模块
    pff: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 位置编码模块
    positional_encoding: PositionalEncoding = PositionalEncoding(d_model, dropout)

    # 1. 编码器部分

    # 1.1 单层编码器
    encoder_layer: EncoderLayer = EncoderLayer(d_model, deepcopy(multi_head_attention), deepcopy(pff), dropout)

    # 1.2 多层堆叠编码器
    encoder: Encoder = Encoder(encoder_layer, layer_count)

    # 2. 解码器部分

    # 2.1 单层解码器
    decoder_layer: DecoderLayer = DecoderLayer(
        d_model,
        deepcopy(multi_head_attention),
        deepcopy(multi_head_attention),
        deepcopy(pff),
        dropout,
    )

    # 2.2 多层堆叠解码器
    decoder: Decoder = Decoder(decoder_layer, layer_count)

    # 3. 输入序列Embedding层
    source_emb: Sequential = Sequential(Embeddings(d_model, src_vocab), deepcopy(positional_encoding))

    # 4. 输出序列Embedding层
    target_emb: Sequential = Sequential(Embeddings(d_model, tgt_vocab), deepcopy(positional_encoding))

    # 5. 生成器
    generator: Generator = Generator(d_model, tgt_vocab)

    # 6. 模型组装
    model: EncoderDecoder = EncoderDecoder(encoder, decoder, source_emb, target_emb, generator)

    # 7. 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)

    return model
