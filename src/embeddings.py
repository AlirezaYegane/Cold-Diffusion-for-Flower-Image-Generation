from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B]
    returns: [B, dim]
    """
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = sinusoidal_timestep_embedding(t, self.time_dim)
        return self.mlp(x)