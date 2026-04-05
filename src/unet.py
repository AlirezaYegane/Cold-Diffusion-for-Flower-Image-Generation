from __future__ import annotations

import torch
import torch.nn as nn

from src.blocks import ResBlock, Downsample, Upsample
from src.embeddings import TimeEmbedding


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.ds1 = Downsample(base_channels)

        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.ds2 = Downsample(base_channels * 2)

        # Bottleneck
        self.mid1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        # Decoder
        self.us1 = Upsample(base_channels * 4)
        self.up1 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)

        self.us2 = Upsample(base_channels * 2)
        self.up2 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        x0 = self.in_conv(x)

        h1 = self.down1(x0, t_emb)       # [B, 64, 64, 64]
        h = self.ds1(h1)                 # [B, 64, 32, 32]

        h2 = self.down2(h, t_emb)        # [B, 128, 32, 32]
        h = self.ds2(h2)                 # [B, 128, 16, 16]

        h = self.mid1(h, t_emb)          # [B, 256, 16, 16]
        h = self.mid2(h, t_emb)          # [B, 256, 16, 16]

        h = self.us1(h)                  # [B, 256, 32, 32]
        h = torch.cat([h, h2], dim=1)    # [B, 384, 32, 32]
        h = self.up1(h, t_emb)           # [B, 128, 32, 32]

        h = self.us2(h)                  # [B, 128, 64, 64]
        h = torch.cat([h, h1], dim=1)    # [B, 192, 64, 64]
        h = self.up2(h, t_emb)           # [B, 64, 64, 64]

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h