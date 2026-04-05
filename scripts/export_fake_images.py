from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data import OxfordFlowersDataset
from src.degradations import ColdDiffusionBlur, ColdDiffusionBlurConfig
from src.sample import denorm
from src.unet import SimpleUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export fake images for FID")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sampling_schedule(num_steps: int, sample_steps: int) -> list[int]:
    sample_steps = max(2, min(sample_steps, num_steps))
    raw = torch.linspace(num_steps - 1, 0, steps=sample_steps)
    schedule = []
    for step in raw.round().long().tolist():
        step = int(step)
        if not schedule or step != schedule[-1]:
            schedule.append(step)

    if schedule[0] != num_steps - 1:
        schedule.insert(0, num_steps - 1)
    if schedule[-1] != 0:
        schedule.append(0)

    return schedule


@torch.no_grad()
def reverse_sample_batch(model, blur, x0, num_steps: int, sample_steps: int):
    device = x0.device
    batch_size = x0.shape[0]

    schedule = build_sampling_schedule(num_steps=num_steps, sample_steps=sample_steps)

    t_start = torch.full((batch_size,), num_steps - 1, device=device, dtype=torch.long)
    x = blur.degrade(x0, t_start)

    for idx, step in enumerate(schedule):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        pred_x0 = model(x, t)

        is_last = idx == len(schedule) - 1
        if is_last or step == 0:
            x = pred_x0
        else:
            prev_step = schedule[idx + 1]
            t_prev = torch.full((batch_size,), prev_step, device=device, dtype=torch.long)
            x = x - blur.degrade(pred_x0, t) + blur.degrade(pred_x0, t_prev)

    return x, schedule


def main():
    args = parse_args()
    console = Console()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]

    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=train_args["base_channels"],
        time_dim=train_args["time_dim"],
    ).to(device)

    state_key = "ema_model_state_dict" if args.use_ema and "ema_model_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    blur = ColdDiffusionBlur(
        ColdDiffusionBlurConfig(
            num_steps=train_args["num_steps"],
            sigma_min=train_args["sigma_min"],
            sigma_max=train_args["sigma_max"],
            kernel_size=train_args["kernel_size"],
            schedule=train_args["schedule"],
        )
    )

    dataset = OxfordFlowersDataset(
        root=project_root,
        split=args.split,
        image_size=train_args["image_size"],
        train_augment=False,
        max_items=args.max_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    console.print(
        Panel.fit(
            (
                f"[bold]Export Fake Images[/bold]\n"
                f"device = {device}\n"
                f"split = {args.split}\n"
                f"items = {len(dataset)}\n"
                f"batch_size = {args.batch_size}\n"
                f"sample_steps = {args.sample_steps}\n"
                f"state_key = {state_key}\n"
                f"checkpoint = {args.checkpoint}\n"
                f"out_dir = {out_dir}"
            ),
            title="Run Setup",
            border_style="cyan",
        )
    )

    saved = 0
    used_schedule = None
    start = time.perf_counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("export fake", total=len(loader))

        for batch in loader:
            x0 = batch["image"].to(device, non_blocking=True)
            image_ids = batch["image_id"]

            x_fake, used_schedule = reverse_sample_batch(
                model=model,
                blur=blur,
                x0=x0,
                num_steps=train_args["num_steps"],
                sample_steps=args.sample_steps,
            )

            x_fake = denorm(x_fake).cpu()

            for img, image_id in zip(x_fake, image_ids):
                image_id = int(image_id)
                out_path = out_dir / f"{image_id:05d}.png"
                save_image(img, out_path)
                saved += 1

            gpu_mem_gb = (
                torch.cuda.memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )

            progress.update(
                task,
                advance=1,
                description=(
                    f"export fake | saved={saved} | "
                    f"steps={args.sample_steps} | mem={gpu_mem_gb:.2f}GB"
                ),
            )

    elapsed = time.perf_counter() - start

    meta = {
        "checkpoint": str(args.checkpoint),
        "state_key": state_key,
        "sample_steps": args.sample_steps,
        "schedule": used_schedule,
        "saved": saved,
        "split": args.split,
        "device": str(device),
        "time_sec": elapsed,
    }
    with open(out_dir / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    table = Table(title="Fake Export Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="bold white")
    table.add_row("saved", str(saved))
    table.add_row("device", str(device))
    table.add_row("sample_steps", str(args.sample_steps))
    table.add_row("state_key", state_key)
    table.add_row("time_sec", f"{elapsed:.2f}")
    table.add_row("out_dir", str(out_dir))
    console.print(table)

    console.print(f"[green]saved[/green] {out_dir / '_meta.json'}")
    console.print(f"[bold]schedule[/bold] = {used_schedule}")


if __name__ == "__main__":
    main()
