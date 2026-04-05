from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data import OxfordFlowersDataset
from src.degradations import ColdDiffusionBlur, ColdDiffusionBlurConfig
from src.sample import save_reconstruction_grid, save_reverse_trajectory_grid
from src.unet import SimpleUNet
from src.utils import (
    AverageMeter,
    EMA,
    append_metrics_csv,
    count_parameters,
    get_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold Diffusion full training")

    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=4.0)
    parser.add_argument("--kernel_size", type=int, default=19)
    parser.add_argument("--schedule", type=str, default="linear")

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])

    parser.add_argument("--sample_every", type=int, default=5)

    parser.add_argument("--train_max_items", type=int, default=None)
    parser.add_argument("--val_max_items", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    return parser.parse_args()


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def get_amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_train_epoch(
    *,
    console: Console,
    model: nn.Module,
    ema_model: nn.Module,
    ema: EMA,
    blur,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_clip: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    meter = AverageMeter()
    start = time.perf_counter()

    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

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
        task = progress.add_task(f"train {epoch}/{total_epochs}", total=total_batches)

        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            x0 = batch["image"].to(device, non_blocking=True)
            batch_size = x0.shape[0]

            t = torch.randint(
                low=0,
                high=blur.config.num_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            x_t = blur.degrade(x0, t)

            optimizer.zero_grad(set_to_none=True)

            with get_amp_context(device=device, enabled=scaler.is_enabled()):
                pred_x0 = model(x_t, t)
                loss = criterion(pred_x0, x0)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(ema_model, model)

            meter.update(loss.item(), batch_size)

            lr = optimizer.param_groups[0]["lr"]
            gpu_mem_gb = (
                torch.cuda.memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )

            progress.update(
                task,
                advance=1,
                description=(
                    f"train {epoch}/{total_epochs} | "
                    f"loss={meter.avg:.4f} | lr={lr:.2e} | mem={gpu_mem_gb:.2f}GB"
                ),
            )

    elapsed = time.perf_counter() - start
    return {"loss": meter.avg, "time_sec": elapsed}


@torch.no_grad()
def run_val_epoch(
    *,
    console: Console,
    model: nn.Module,
    blur,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    meter = AverageMeter()
    start = time.perf_counter()

    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

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
        task = progress.add_task(f"val {epoch}/{total_epochs}", total=total_batches)

        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            x0 = batch["image"].to(device, non_blocking=True)
            batch_size = x0.shape[0]

            t = torch.randint(
                low=0,
                high=blur.config.num_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            x_t = blur.degrade(x0, t)
            pred_x0 = model(x_t, t)
            loss = criterion(pred_x0, x0)

            meter.update(loss.item(), batch_size)

            progress.update(
                task,
                advance=1,
                description=f"val {epoch}/{total_epochs} | loss={meter.avg:.4f}",
            )

    elapsed = time.perf_counter() - start
    return {"loss": meter.avg, "time_sec": elapsed}


def main() -> None:
    args = parse_args()
    console = Console()
    project_root = Path(args.project_root).resolve()

    seed_everything(args.seed)
    device = get_device()
    pin_memory = device.type == "cuda"

    outputs_dir = project_root / "outputs"
    ckpt_dir = outputs_dir / "checkpoints"
    sample_dir = outputs_dir / "samples"
    metrics_dir = outputs_dir / "metrics"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OxfordFlowersDataset(
        root=project_root,
        split="train",
        image_size=args.image_size,
        train_augment=True,
        max_items=args.train_max_items,
    )
    val_dataset = OxfordFlowersDataset(
        root=project_root,
        split="val",
        image_size=args.image_size,
        train_augment=False,
        max_items=args.val_max_items,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    ).to(device)

    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    ema = EMA(decay=args.ema_decay)

    blur = ColdDiffusionBlur(
        ColdDiffusionBlurConfig(
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            kernel_size=args.kernel_size,
            schedule=args.schedule,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    fixed_batch = next(iter(val_loader))
    fixed_x0 = fixed_batch["image"][:4].to(device)

    history_csv = metrics_dir / "train_history.csv"
    latest_json = metrics_dir / "latest_metrics.json"
    config_json = metrics_dir / "train_config.json"

    save_json(vars(args), config_json)

    console.print(
        Panel.fit(
            (
                f"[bold]Cold Diffusion Full Training[/bold]\n"
                f"device = {device}\n"
                f"train size = {len(train_dataset)}\n"
                f"val size = {len(val_dataset)}\n"
                f"params = {count_parameters(model):,}"
            ),
            title="Run Setup",
            border_style="cyan",
        )
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            console=console,
            model=model,
            ema_model=ema_model,
            ema=ema,
            blur=blur,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )

        val_metrics = run_val_epoch(
            console=console,
            model=ema_model,
            blur=blur,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            max_batches=args.max_val_batches,
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        if epoch % args.sample_every == 0 or epoch == 1:
            recon_path = sample_dir / f"epoch_{epoch:04d}_recon.png"
            reverse_path = sample_dir / f"epoch_{epoch:04d}_reverse.png"

            save_reconstruction_grid(
                model=ema_model,
                blur=blur,
                fixed_x0=fixed_x0,
                out_path=recon_path,
                num_steps=args.num_steps,
            )
            save_reverse_trajectory_grid(
                model=ema_model,
                blur=blur,
                fixed_x0=fixed_x0,
                out_path=reverse_path,
                num_steps=args.num_steps,
            )
            console.print(f"[green]saved[/green] {recon_path}")
            console.print(f"[green]saved[/green] {reverse_path}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        last_ckpt = ckpt_dir / "full_train_last.pt"
        best_ckpt = ckpt_dir / "full_train_best.pt"

        save_checkpoint(checkpoint, last_ckpt)
        if is_best:
            save_checkpoint(checkpoint, best_ckpt)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "best_val_loss": round(best_val_loss, 6),
            "train_time_sec": round(train_metrics["time_sec"], 2),
            "val_time_sec": round(val_metrics["time_sec"], 2),
            "lr": optimizer.param_groups[0]["lr"],
            "device": str(device),
            "best_flag": int(is_best),
        }
        append_metrics_csv(row, history_csv)
        save_json(row, latest_json)

        table = Table(title=f"Epoch {epoch}/{args.epochs} Summary")
        table.add_column("metric", style="cyan")
        table.add_column("value", style="bold white")

        table.add_row("train_loss", f"{train_metrics['loss']:.6f}")
        table.add_row("val_loss", f"{val_metrics['loss']:.6f}")
        table.add_row("best_val_loss", f"{best_val_loss:.6f}")
        table.add_row("train_time_sec", f"{train_metrics['time_sec']:.2f}")
        table.add_row("val_time_sec", f"{val_metrics['time_sec']:.2f}")
        table.add_row("checkpoint", "best + last" if is_best else "last")
        console.print(table)

    console.print(
        Panel.fit(
            (
                f"[bold green]Training finished[/bold green]\n"
                f"best_val_loss = {best_val_loss:.6f}\n"
                f"best_ckpt = {ckpt_dir / 'full_train_best.pt'}\n"
                f"last_ckpt = {ckpt_dir / 'full_train_last.pt'}\n"
                f"history = {history_csv}"
            ),
            title="Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()