from typing import Sequence

from torch import Tensor
from torch.nn import Module

from sublayer_connection import SublayerConnection
from utils import clones


class EncoderLayer(Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(
        self, size: Sequence[int], self_attn, feed_forward, dropout: float
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size: int = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Follow Figure 1 (left) for connections."""
        x: Tensor = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
