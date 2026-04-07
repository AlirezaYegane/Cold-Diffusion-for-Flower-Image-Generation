import json
from pathlib import Path
import pandas as pd

specs = [
    ("best_15", 15, "outputs/metrics/fid_final_15.json"),
    ("step_25", 25, "outputs/metrics/fid_final_25.json"),
    ("step_50", 50, "outputs/metrics/fid_final_50.json"),
    ("step_100", 100, "outputs/metrics/fid_final_100.json"),
]

rows = []
for run_name, steps, path_str in specs:
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    rows.append({
        "run": run_name,
        "steps": steps,
        "fid": data["fid"],
        "num_real": data["num_real"],
        "num_fake": data["num_fake"],
        "device": data["device"],
        "time_sec": data["time_sec"],
    })

df = pd.DataFrame(rows).sort_values("steps")
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("outputs/tables/fid_ablation.csv", index=False)
print(df.to_string(index=False))
