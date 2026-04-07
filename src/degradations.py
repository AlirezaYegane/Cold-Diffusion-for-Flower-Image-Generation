from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PredictionTarget = str


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
    kernel_size = _ensure_odd(kernel_size)
    radius = kernel_size // 2

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid_sq = xx.pow(2) + yy.pow(2)

    sigma = sigma.clamp(min=1e-4).to(device=device, dtype=dtype)
    sigma_sq = sigma.view(-1, 1, 1).pow(2)

    kernel = torch.exp(-0.5 * grid_sq.view(1, kernel_size, kernel_size) / sigma_sq)
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return kernel


def apply_gaussian_blur_batch(x: torch.Tensor, sigma: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"x must be [B, C, H, W], got shape={tuple(x.shape)}")

    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype
    kernel_size = _ensure_odd(kernel_size)
    pad = kernel_size // 2

    kernels = gaussian_kernel_2d(kernel_size, sigma, device=device, dtype=dtype)
    weight = kernels[:, None, :, :].repeat_interleave(c, dim=0)

    x_grouped = x.reshape(1, b * c, h, w)
    out = F.conv2d(x_grouped, weight, padding=pad, groups=b * c)
    return out.reshape(b, c, h, w)


@dataclass
class ColdDiffusionBlurConfig:
    num_steps: int = 100
    sigma_min: float = 0.01
    sigma_max: float = 4.0
    kernel_size: int = 19
    schedule: str = "linear"


class ColdDiffusionBlur:
    def __init__(self, config: ColdDiffusionBlurConfig) -> None:
        self.config = config
        self.kernel_size = _ensure_odd(config.kernel_size)

    def sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be 1D [B], got shape={tuple(t.shape)}")
        if self.config.schedule == "linear":
            return linear_schedule(t, self.config.num_steps, self.config.sigma_min, self.config.sigma_max)
        if self.config.schedule == "quadratic":
            return quadratic_schedule(t, self.config.num_steps, self.config.sigma_min, self.config.sigma_max)
        raise ValueError(f"Unknown schedule: {self.config.schedule}")

    def degrade(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_from_t(t).to(device=x0.device, dtype=x0.dtype)
        return apply_gaussian_blur_batch(x0, sigma, kernel_size=self.kernel_size)

    def degrade_single_step(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        if x0.ndim != 3:
            raise ValueError(f"x0 must be [C, H, W], got shape={tuple(x0.shape)}")
        x = x0.unsqueeze(0)
        tt = torch.tensor([t], device=x.device, dtype=torch.long)
        return self.degrade(x, tt).squeeze(0)

    def target_from_pair(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        prediction_target: PredictionTarget = "x0",
    ) -> torch.Tensor:
        if prediction_target == "x0":
            return x0
        if prediction_target == "residual":
            return x0 - x_t
        raise ValueError(f"Unknown prediction_target: {prediction_target}")

    def pred_x0_from_model_output(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        prediction_target: PredictionTarget = "x0",
    ) -> torch.Tensor:
        if prediction_target == "x0":
            return model_out
        if prediction_target == "residual":
            return x_t + model_out
        raise ValueError(f"Unknown prediction_target: {prediction_target}")

    def reverse_step(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction_target: PredictionTarget = "x0",
    ) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be 1D [B], got shape={tuple(t.shape)}")

        pred_x0 = self.pred_x0_from_model_output(model_out, x_t, prediction_target=prediction_target)

        if torch.all(t == 0):
            return pred_x0

        t_prev = (t - 1).clamp(min=0)
        return x_t - self.degrade(pred_x0, t) + self.degrade(pred_x0, t_prev)

    @torch.no_grad()
    def sample_trajectory(
        self,
        model: torch.nn.Module,
        x_init: torch.Tensor,
        prediction_target: PredictionTarget = "x0",
        capture_steps: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        if x_init.ndim != 4:
            raise ValueError(f"x_init must be [B, C, H, W], got shape={tuple(x_init.shape)}")

        device = x_init.device
        num_steps = self.config.num_steps
        x = x_init.clone()

        if capture_steps is None:
            capture_steps = sorted(
                {
                    num_steps - 1,
                    int(0.75 * (num_steps - 1)),
                    int(0.5 * (num_steps - 1)),
                    int(0.25 * (num_steps - 1)),
                    0,
                },
                reverse=True,
            )

        frames: dict[int, torch.Tensor] = {num_steps - 1: x.detach().cpu()}

        was_training = model.training
        model.eval()
        for step in reversed(range(num_steps)):
            t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
            model_out = model(x, t)
            x = self.reverse_step(model_out, x, t, prediction_target=prediction_target)
            if step in capture_steps:
                frames[step] = x.detach().cpu()
        if was_training:
            model.train()

        return frames
