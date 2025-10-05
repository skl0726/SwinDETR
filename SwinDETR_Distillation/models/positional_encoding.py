""" Positional Encoding for DETR Encoder-Decoder """


import torch
import torch.nn as nn
from torch import Tensor

import math
from typing import Tuple


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        N, _, H, W = x.shape
        
        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), -1).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 1::2].sin(), pos_y[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((pos_y, pos_x), 3).permute(0, 3, 1, 2), mask