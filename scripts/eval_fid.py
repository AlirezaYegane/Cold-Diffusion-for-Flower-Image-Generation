from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from torchmetrics.image.fid import FrechetInceptionDistance


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID between two folders")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_json", type=str, required=True)
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


def load_image_uint8(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def batched(paths: list[Path], batch_size: int):
    for i in range(0, len(paths), batch_size):
        yield paths[i:i + batch_size]


def main():
    args = parse_args()
    console = Console()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    real_paths = sorted([p for p in real_dir.glob("*.png") if p.is_file()])
    fake_paths = sorted([p for p in fake_dir.glob("*.png") if p.is_file()])

    if len(real_paths) == 0:
        raise RuntimeError(f"No real images found in {real_dir}")
    if len(fake_paths) == 0:
        raise RuntimeError(f"No fake images found in {fake_dir}")

    device = resolve_device(args.device)

    console.print(
        Panel.fit(
            (
                f"[bold]FID Evaluation[/bold]\n"
                f"device = {device}\n"
                f"num_real = {len(real_paths)}\n"
                f"num_fake = {len(fake_paths)}\n"
                f"batch_size = {args.batch_size}\n"
                f"real_dir = {real_dir}\n"
                f"fake_dir = {fake_dir}"
            ),
            title="Run Setup",
            border_style="cyan",
        )
    )

    start = time.perf_counter()
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    total_chunks = ((len(real_paths) + args.batch_size - 1) // args.batch_size) + ((len(fake_paths) + args.batch_size - 1) // args.batch_size)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("fid update(real)", total=total_chunks)

        for chunk in batched(real_paths, args.batch_size):
            batch = torch.stack([load_image_uint8(p) for p in chunk], dim=0).to(device, non_blocking=True)
            fid.update(batch, real=True)

            gpu_mem_gb = (
                torch.cuda.memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            progress.update(
                task,
                advance=1,
                description=f"fid update(real) | seen={min(progress.tasks[0].completed * args.batch_size, len(real_paths))} | mem={gpu_mem_gb:.2f}GB",
            )

        for chunk in batched(fake_paths, args.batch_size):
            batch = torch.stack([load_image_uint8(p) for p in chunk], dim=0).to(device, non_blocking=True)
            fid.update(batch, real=False)

            gpu_mem_gb = (
                torch.cuda.memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            progress.update(
                task,
                advance=1,
                description=f"fid update(fake) | mem={gpu_mem_gb:.2f}GB",
            )

    score = float(fid.compute().item())
    elapsed = time.perf_counter() - start

    payload = {
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "num_real": len(real_paths),
        "num_fake": len(fake_paths),
        "fid": score,
        "device": str(device),
        "time_sec": elapsed,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    table = Table(title="FID Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="bold white")
    table.add_row("fid", f"{score:.6f}")
    table.add_row("device", str(device))
    table.add_row("num_real", str(len(real_paths)))
    table.add_row("num_fake", str(len(fake_paths)))
    table.add_row("time_sec", f"{elapsed:.2f}")
    table.add_row("out_json", str(out_json))
    console.print(table)


if __name__ == "__main__":
    main()
