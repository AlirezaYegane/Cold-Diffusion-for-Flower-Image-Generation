from __future__ import annotations

import argparse
import time
from pathlib import Path

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
from src.sample import denorm


def parse_args():
    parser = argparse.ArgumentParser(description="Export real images for FID")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = OxfordFlowersDataset(
        root=project_root,
        split=args.split,
        image_size=args.image_size,
        train_augment=False,
        max_items=args.max_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    console.print(
        Panel.fit(
            (
                f"[bold]Export Real Images[/bold]\n"
                f"split = {args.split}\n"
                f"items = {len(dataset)}\n"
                f"batch_size = {args.batch_size}\n"
                f"out_dir = {out_dir}"
            ),
            title="Run Setup",
            border_style="cyan",
        )
    )

    saved = 0
    start = time.perf_counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("export real", total=len(loader))
        for batch in loader:
            imgs = denorm(batch["image"])
            image_ids = batch["image_id"]

            for img, image_id in zip(imgs, image_ids):
                image_id = int(image_id)
                out_path = out_dir / f"{image_id:05d}.png"
                save_image(img, out_path)
                saved += 1

            progress.update(
                task,
                advance=1,
                description=f"export real | saved={saved}",
            )

    elapsed = time.perf_counter() - start

    table = Table(title="Real Export Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="bold white")
    table.add_row("saved", str(saved))
    table.add_row("time_sec", f"{elapsed:.2f}")
    table.add_row("out_dir", str(out_dir))
    console.print(table)


if __name__ == "__main__":
    main()
