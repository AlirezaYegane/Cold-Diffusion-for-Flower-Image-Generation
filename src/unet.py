from __future__ import annotations

import torch
import torch.nn as nn

from src.blocks import AttentionBlock, Downsample, ResBlock, Upsample, make_group_norm
from src.embeddings import TimeEmbedding


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.time_embedding = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.down1 = ResBlock(c1, c1, time_dim, dropout=dropout)
        self.ds1 = Downsample(c1)

        self.down2 = ResBlock(c1, c2, time_dim, dropout=dropout)
        self.attn2 = AttentionBlock(c2)
        self.ds2 = Downsample(c2)

        self.mid1 = ResBlock(c2, c3, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(c3)
        self.mid2 = ResBlock(c3, c3, time_dim, dropout=dropout)

        self.us1 = Upsample(c3)
        self.up1 = ResBlock(c3 + c2, c2, time_dim, dropout=dropout)
        self.up1_attn = AttentionBlock(c2)

        self.us2 = Upsample(c2)
        self.up2 = ResBlock(c2 + c1, c1, time_dim, dropout=dropout)

        self.out_norm = make_group_norm(c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x0 = self.in_conv(x)

        h1 = self.down1(x0, t_emb)
        h = self.ds1(h1)

        h2 = self.down2(h, t_emb)
        h2 = self.attn2(h2)
        h = self.ds2(h2)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        h = self.us1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, t_emb)
        h = self.up1_attn(h)

        h = self.us2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, t_emb)

        return self.out_conv(self.out_act(self.out_norm(h)))
