""" DenseNet Backbone (to compare with transformer backbone) """


import torch
from torch import nn

from typing import List


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, num_groups):
        super().__init__()
        self.out_channels = int(in_channels / 2)
        self.module = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.out_channels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.module(x)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_groups, growth_rate):
        super().__init__()
        self.out_channels = growth_rate
        self.module = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1),
            nn.GroupNorm(num_groups, 4 * growth_rate),
            nn.ReLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.module(x)
    

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_blocks, num_groups, growth_rate):
        super().__init__()
        self.out_channels = in_channels

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(ConvBlock(self.out_channels, num_groups, growth_rate))
            self.out_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, num_groups, growth_rate, num_blocks: List[int]):
        super().__init__()
        self.out_channels = 64

        self.input = nn.Sequential(
            nn.Conv2d(3, self.out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups, self.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layers = [self.input]

        for blocks in num_blocks:
            block = DenseBlock(self.out_channels, blocks, num_groups, growth_rate)
            self.out_channels = block.out_channels
            trans = TransitionLayer(self.out_channels, num_groups)
            self.out_channels = trans.out_channels
            layers.append(block)
            layers.append(trans)

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)