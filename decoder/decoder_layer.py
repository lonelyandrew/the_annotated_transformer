from torch.nn import Module

from sublayer_connection import SublayerConnection
from utils import clones


class DecoderLayer(Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
