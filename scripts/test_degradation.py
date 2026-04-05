from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from src.degradations import ColdDiffusionBlur, ColdDiffusionBlurConfig


def denorm_for_plot(x: torch.Tensor) -> torch.Tensor:
    # x in [-1, 1] -> [0, 1]
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "data" / "raw" / "jpg"
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No JPG files found in: {image_dir}")

    transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # چند تصویر برای تست batch
    batch_imgs = []
    for p in image_paths[:4]:
        img = Image.open(p).convert("RGB")
        batch_imgs.append(transform(img))
    x = torch.stack(batch_imgs, dim=0)  # [4, 3, 64, 64]

    blur = ColdDiffusionBlur(
        ColdDiffusionBlurConfig(
            num_steps=100,
            sigma_min=0.01,
            sigma_max=4.0,
            kernel_size=19,
            schedule="linear",
        )
    )

    # تست batch
    t_batch = torch.tensor([0, 15, 50, 99], dtype=torch.long)
    x_blur = blur.degrade(x, t_batch)

    print("input batch shape :", tuple(x.shape))
    print("output batch shape:", tuple(x_blur.shape))
    print("timesteps         :", t_batch.tolist())
    print("sigmas            :", blur.sigma_from_t(t_batch).tolist())

    # blur progression برای یک تصویر
    img0 = x[0]
    timesteps = [0, 10, 25, 50, 75, 99]

    fig, axes = plt.subplots(1, len(timesteps) + 1, figsize=(18, 3))
    axes[0].imshow(denorm_for_plot(img0).permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, t in enumerate(timesteps, start=1):
        xt = blur.degrade_single_step(img0, t)
        sigma = blur.sigma_from_t(torch.tensor([t])).item()
        axes[i].imshow(denorm_for_plot(xt).permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"t={t}\nσ={sigma:.2f}")
        axes[i].axis("off")

    plt.tight_layout()
    out_path = out_dir / "blur_progression.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"saved figure to: {out_path}")


if __name__ == "__main__":
    main()