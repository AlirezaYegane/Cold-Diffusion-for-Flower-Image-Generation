from __future__ import annotations

import torch

from src.degradations import ColdDiffusionBlur, ColdDiffusionBlurConfig
from src.unet import SimpleUNet


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur = ColdDiffusionBlur(
        ColdDiffusionBlurConfig(
            num_steps=100,
            sigma_min=0.01,
            sigma_max=4.0,
            kernel_size=19,
            schedule="linear",
        )
    )
    model = SimpleUNet(base_channels=32, time_dim=128).to(device)

    x0 = torch.randn(2, 3, 64, 64, device=device)
    t0 = torch.zeros(2, dtype=torch.long, device=device)
    t_mid = torch.full((2,), 50, dtype=torch.long, device=device)
    t_hi = torch.full((2,), 99, dtype=torch.long, device=device)

    x_t0 = blur.degrade(x0, t0)
    x_mid = blur.degrade(x0, t_mid)
    x_hi = blur.degrade(x0, t_hi)

    assert x_t0.shape == x0.shape == x_mid.shape == x_hi.shape
    assert blur.sigma_from_t(t_hi).mean() > blur.sigma_from_t(t_mid).mean() > blur.sigma_from_t(t0).mean()
    assert hf_energy(x_hi) < hf_energy(x_mid) < hf_energy(x0)

    out = model(x_mid, t_mid)
    assert out.shape == x0.shape

    loss = (out - x0).abs().mean()
    loss.backward()

    residual_target = blur.target_from_pair(x0, x_mid, prediction_target="residual")
    pred_x0 = blur.pred_x0_from_model_output(residual_target, x_mid, prediction_target="residual")
    assert torch.allclose(pred_x0, x0, atol=1e-5)

    x_prev = blur.reverse_step(out.detach(), x_mid.detach(), t_mid, prediction_target="x0")
    assert x_prev.shape == x0.shape

    print("[ok] sigma schedule monotonic")
    print("[ok] blur strength increases with timestep")
    print("[ok] model forward/backward shape check")
    print("[ok] residual target conversion")
    print("[ok] reverse step shape check")
    print(f"device = {device}")


if __name__ == "__main__":
    main()
