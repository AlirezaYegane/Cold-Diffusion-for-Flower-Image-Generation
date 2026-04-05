from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _ensure_odd(kernel_size: int) -> int:
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def linear_schedule(t: torch.Tensor, num_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    if num_steps <= 1:
        return torch.full_like(t, fill_value=sigma_max, dtype=torch.float32)
    t = t.float()
    return sigma_min + (sigma_max - sigma_min) * (t / (num_steps - 1))


def quadratic_schedule(t: torch.Tensor, num_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    if num_steps <= 1:
        return torch.full_like(t, fill_value=sigma_max, dtype=torch.float32)
    t = t.float() / (num_steps - 1)
    return sigma_min + (sigma_max - sigma_min) * (t ** 2)


def gaussian_kernel_2d(kernel_size: int, sigma: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns per-sample 2D Gaussian kernels.
    sigma: shape [B]
    output: shape [B, kernel_size, kernel_size]
    """
    kernel_size = _ensure_odd(kernel_size)
    radius = kernel_size // 2

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid_sq = xx.pow(2) + yy.pow(2)  # [K, K]

    sigma = sigma.clamp(min=1e-4).to(device=device, dtype=dtype)  # [B]
    sigma_sq = sigma.view(-1, 1, 1).pow(2)

    kernel = torch.exp(-0.5 * grid_sq.view(1, kernel_size, kernel_size) / sigma_sq)
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return kernel


def apply_gaussian_blur_batch(x: torch.Tensor, sigma: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    x: [B, C, H, W]
    sigma: [B]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B, C, H, W], got shape={tuple(x.shape)}")

    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype
    kernel_size = _ensure_odd(kernel_size)
    pad = kernel_size // 2

    kernels = gaussian_kernel_2d(kernel_size, sigma, device=device, dtype=dtype)  # [B, K, K]
    weight = kernels[:, None, :, :]                     # [B, 1, K, K]
    weight = weight.repeat_interleave(c, dim=0)        # [B*C, 1, K, K]

    x_grouped = x.reshape(1, b * c, h, w)
    out = F.conv2d(x_grouped, weight, padding=pad, groups=b * c)
    out = out.reshape(b, c, h, w)
    return out


@dataclass
class ColdDiffusionBlurConfig:
    num_steps: int = 100
    sigma_min: float = 0.01
    sigma_max: float = 4.0
    kernel_size: int = 19
    schedule: str = "linear"  # "linear" or "quadratic"


class ColdDiffusionBlur:
    def __init__(self, config: ColdDiffusionBlurConfig) -> None:
        self.config = config
        self.kernel_size = _ensure_odd(config.kernel_size)

    def sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] integer timesteps in [0, num_steps-1]
        returns sigma: [B]
        """
        if t.ndim != 1:
            raise ValueError(f"t must be 1D [B], got shape={tuple(t.shape)}")

        if self.config.schedule == "linear":
            return linear_schedule(t, self.config.num_steps, self.config.sigma_min, self.config.sigma_max)
        if self.config.schedule == "quadratic":
            return quadratic_schedule(t, self.config.num_steps, self.config.sigma_min, self.config.sigma_max)

        raise ValueError(f"Unknown schedule: {self.config.schedule}")

    def degrade(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x0: [B, C, H, W]
        t:  [B]
        """
        sigma = self.sigma_from_t(t).to(device=x0.device, dtype=x0.dtype)
        return apply_gaussian_blur_batch(x0, sigma, kernel_size=self.kernel_size)

    def degrade_single_step(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """
        x0: [C, H, W]
        """
        if x0.ndim != 3:
            raise ValueError(f"x0 must be [C, H, W], got shape={tuple(x0.shape)}")

        x = x0.unsqueeze(0)
        tt = torch.tensor([t], device=x.device, dtype=torch.long)
        out = self.degrade(x, tt)
        return out.squeeze(0)