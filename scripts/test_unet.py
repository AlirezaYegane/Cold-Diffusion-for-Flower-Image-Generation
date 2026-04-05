from __future__ import annotations

import torch

from src.unet import SimpleUNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)

    x = torch.randn(4, 3, 64, 64, device=device)
    t = torch.randint(low=0, high=100, size=(4,), device=device)

    y = model(x, t)

    print("device      :", device)
    print("input shape :", tuple(x.shape))
    print("t shape     :", tuple(t.shape))
    print("output shape:", tuple(y.shape))
    print("params      :", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()