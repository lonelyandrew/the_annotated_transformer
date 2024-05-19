import copy

import altair
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module, ModuleList


def clones(module: Module, n: int) -> ModuleList:
    """Produce N identical layers."""
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """Mask out subsequent positions."""
    attn_shape: tuple[int, int, int] = (1, size, size)
    subsequent_mask_result: Tensor = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask_result == 0
