import numpy as np
from pathlib import Path
import csv

depth_dir = Path("depth_maps")
pred_dir = Path("predicted_depth")

results = []

for gt_file in sorted(depth_dir.glob("*_depth.npy")):

    scene = gt_file.stem.replace("_depth", "")

    pred_file = pred_dir / f"{scene}_depth.npy"

    if not pred_file.exists():
        print(f"Skipping {scene} (prediction missing)")
        continue

    gt = np.load(gt_file).astype(np.float32)
    pred = np.load(pred_file).astype(np.float32)

    if gt.shape != pred.shape:
        print(f"Shape mismatch for {scene}")
        continue

    # Scale alignment
    a = np.sum(gt * pred) / np.sum(pred * pred)
    pred = pred * a

    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    absrel = np.mean(np.abs(gt - pred) / (gt + 1e-6))

    results.append([scene, mae, rmse, absrel])

print("\n==============================")
print("Baseline Evaluation")
print("==============================")

for r in results:
    print(f"{r[0]:30s} MAE={r[1]:.4f} RMSE={r[2]:.4f} AbsRel={r[3]:.4f}")

mean_mae = np.mean([r[1] for r in results])
mean_rmse = np.mean([r[2] for r in results])
mean_absrel = np.mean([r[3] for r in results])

print("\n------------------------------")
print(f"Mean MAE    : {mean_mae:.4f}")
print(f"Mean RMSE   : {mean_rmse:.4f}")
print(f"Mean AbsRel : {mean_absrel:.4f}")

with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["Scene", "MAE", "RMSE", "AbsRel"])

    for r in results:
        writer.writerow([
            r[0],
            f"{r[1]:.6f}",
            f"{r[2]:.6f}",
            f"{r[3]:.6f}"
        ])

    writer.writerow([])
    writer.writerow([
        "Mean",
        f"{mean_mae:.6f}",
        f"{mean_rmse:.6f}",
        f"{mean_absrel:.6f}"
    ])

print("\nSaved baseline_results.csv")
