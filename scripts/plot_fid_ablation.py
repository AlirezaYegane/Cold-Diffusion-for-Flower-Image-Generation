from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("outputs/tables/fid_ablation.csv")
fig_dir = Path("outputs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

plot_df = df[df["run"].isin(["25", "50", "100"])].copy()
plot_df["run"] = plot_df["run"].astype(int)
plot_df = plot_df.sort_values("run")

plt.figure(figsize=(7, 4.5))
plt.plot(plot_df["run"], plot_df["fid"], marker="o")
plt.xlabel("Reverse sampling steps")
plt.ylabel("FID")
plt.title("FID vs Reverse Sampling Steps")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "fid_ablation_steps.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.bar(df["run"].astype(str), df["time_sec"])
plt.xlabel("Run")
plt.ylabel("Evaluation time (sec)")
plt.title("FID Evaluation Runtime")
plt.tight_layout()
plt.savefig(fig_dir / "fid_runtime.png", dpi=200)
plt.close()

print("saved", fig_dir / "fid_ablation_steps.png")
print("saved", fig_dir / "fid_runtime.png")
