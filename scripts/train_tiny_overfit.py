from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data import OxfordFlowersDataset
from src.degradations import ColdDiffusionBlur, ColdDiffusionBlurConfig
from src.unet import SimpleUNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def save_recon_grid(
    model: nn.Module,
    blur: ColdDiffusionBlur,
    fixed_x0: torch.Tensor,
    step: int,
    out_dir: Path,
    num_steps: int,
) -> None:
    model.eval()

    device = fixed_x0.device
    b = fixed_x0.shape[0]

    # برای visualization ثابت، timestep نسبتاً سخت می‌گیریم
    t_vis = torch.full((b,), fill_value=int(num_steps * 0.75), device=device, dtype=torch.long)

    x_t = blur.degrade(fixed_x0, t_vis)
    pred_x0 = model(x_t, t_vis)

    grid = torch.cat([
        denorm(fixed_x0).cpu(),
        denorm(x_t).cpu(),
        denorm(pred_x0).cpu(),
    ], dim=0)

    out_path = out_dir / f"recon_step_{step:04d}.png"
    save_image(grid, out_path, nrow=b)
    print(f"[saved] {out_path}")

    model.train()


def main() -> None:
    set_seed(42)

    project_root = Path(__file__).resolve().parents[1]
    ckpt_dir = project_root / "outputs" / "checkpoints"
    fig_dir = project_root / "outputs" / "figures" / "tiny_overfit"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # چون الان روی CPU هستی، برای debug سبک‌ترش می‌کنیم
    dataset = OxfordFlowersDataset(
        root=project_root,
        split="train",
        image_size=64,
        train_augment=False,   # برای overfit خاموش می‌کنیم
        max_items=8,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # fixed batch برای visualization
    fixed_items = [dataset[i]["image"] for i in range(4)]
    fixed_x0 = torch.stack(fixed_items, dim=0).to(device)

    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,   # debug-friendly
        time_dim=128,
    ).to(device)

    blur = ColdDiffusionBlur(
        ColdDiffusionBlurConfig(
            num_steps=100,
            sigma_min=0.01,
            sigma_max=4.0,
            kernel_size=19,
            schedule="linear",
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    max_steps = 300
    log_every = 10
    vis_every = 50
    best_loss = float("inf")
    global_step = 0

    while global_step < max_steps:
        for batch in loader:
            x0 = batch["image"].to(device)
            bsz = x0.shape[0]

            t = torch.randint(
                low=0,
                high=blur.config.num_steps,
                size=(bsz,),
                device=device,
                dtype=torch.long,
            )

            x_t = blur.degrade(x0, t)
            pred_x0 = model(x_t, t)

            loss = criterion(pred_x0, x0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            if global_step % log_every == 0 or global_step == 1:
                print(f"step={global_step:04d} | loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": global_step,
                        "best_loss": best_loss,
                    },
                    ckpt_dir / "tiny_overfit_best.pt",
                )

            if global_step % vis_every == 0 or global_step == 1:
                save_recon_grid(
                    model=model,
                    blur=blur,
                    fixed_x0=fixed_x0,
                    step=global_step,
                    out_dir=fig_dir,
                    num_steps=blur.config.num_steps,
                )

            if global_step >= max_steps:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": global_step,
            "best_loss": best_loss,
        },
        ckpt_dir / "tiny_overfit_last.pt",
    )

    print("\nTraining finished.")
    print(f"best_loss={best_loss:.6f}")
    print(f"best_ckpt={ckpt_dir / 'tiny_overfit_best.pt'}")
    print(f"last_ckpt={ckpt_dir / 'tiny_overfit_last.pt'}")


if __name__ == "__main__":
    main()