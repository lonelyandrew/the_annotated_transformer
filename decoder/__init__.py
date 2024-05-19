import torch
from torch.nn import Module

from layer_norm import LayerNorm
from utils import clones


class Decoder(Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, n: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



