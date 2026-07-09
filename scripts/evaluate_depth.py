import sys
import numpy as np

if len(sys.argv) < 2:
    print("ERROR: scene name required, e.g. python3 evaluate_depth.py world1_baseline")
    sys.exit(1)

scene = sys.argv[1]

# ------------------------
# Load depth maps
# ------------------------

gt = np.load(f"depth_maps/{scene}_depth.npy")
pred = np.load(f"predicted_depth/{scene}_depth.npy")

# ------------------------
# Resize prediction if needed
# ------------------------

if pred.shape != gt.shape:
    raise ValueError(
        f"Shape mismatch: GT={gt.shape}, Prediction={pred.shape}"
    )

# ------------------------
# IMPORTANT:
# Depth Anything predicts relative depth.
# Align prediction scale to ground truth.
# ------------------------

pred = pred.astype(np.float32)
gt = gt.astype(np.float32)

# Linear scale alignment
a = np.sum(gt * pred) / np.sum(pred * pred)
pred_aligned = a * pred

# ------------------------
# Metrics
# ------------------------

mae = np.mean(np.abs(gt - pred_aligned))

rmse = np.sqrt(
    np.mean((gt - pred_aligned) ** 2)
)

absrel = np.mean(
    np.abs(gt - pred_aligned) /
    (gt + 1e-6)
)

print(f"Scene : {scene}")
print(f"MAE   : {mae:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"AbsRel: {absrel:.4f}")