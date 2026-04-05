from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("outputs/metrics/train_history.csv")
out_dir = Path("outputs/report_figures")
out_dir.mkdir(parents=True, exist_ok=True)

if csv_path.exists():
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, markersize=5, label="Train")
    plt.plot(df["epoch"], df["val_loss"], marker="o", linewidth=2, markersize=5, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_loss_curves.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["best_val_loss"], marker="o", linewidth=2, markersize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_best_val_curve.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("saved training curves")
else:
    print("train_history.csv not found, skipped training curve generation")
