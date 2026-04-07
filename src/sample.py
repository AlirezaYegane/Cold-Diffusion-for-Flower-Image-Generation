from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def save_reconstruction_grid(
    model: torch.nn.Module,
    blur,
    fixed_x0: torch.Tensor,
    out_path: str | Path,
    num_steps: int,
    timestep_frac: float = 0.75,
    prediction_target: str = "x0",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    device = fixed_x0.device
    batch_size = fixed_x0.shape[0]
    t_value = int((num_steps - 1) * timestep_frac)
    t_vis = torch.full((batch_size,), t_value, device=device, dtype=torch.long)

    x_t = blur.degrade(fixed_x0, t_vis)
    model_out = model(x_t, t_vis)
    pred_x0 = blur.pred_x0_from_model_output(model_out, x_t, prediction_target=prediction_target)

    grid = torch.cat(
        [
            denorm(fixed_x0).cpu(),
            denorm(x_t).cpu(),
            denorm(pred_x0).cpu(),
        ],
        dim=0,
    )
    save_image(grid, out_path, nrow=batch_size)

    if was_training:
        model.train()


@torch.no_grad()
def save_reverse_trajectory_grid(
    model: torch.nn.Module,
    blur,
    fixed_x0: torch.Tensor,
    out_path: str | Path,
    num_steps: int,
    capture_steps: list[int] | None = None,
    prediction_target: str = "x0",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = fixed_x0.shape[0]
    device = fixed_x0.device

    if capture_steps is None:
        capture_steps = sorted(
            {
                num_steps - 1,
                int(0.75 * (num_steps - 1)),
                int(0.50 * (num_steps - 1)),
                int(0.25 * (num_steps - 1)),
                0,
            },
            reverse=True,
        )

    t_T = torch.full((batch_size,), num_steps - 1, device=device, dtype=torch.long)
    x_init = blur.degrade(fixed_x0, t_T)

    frames = blur.sample_trajectory(
        model=model,
        x_init=x_init,
        prediction_target=prediction_target,
        capture_steps=capture_steps,
    )

    rows = [denorm(fixed_x0).cpu()]
    for step in capture_steps:
        if step in frames:
            rows.append(denorm(frames[step]))

    grid = torch.cat(rows, dim=0)
    save_image(grid, out_path, nrow=batch_size)
