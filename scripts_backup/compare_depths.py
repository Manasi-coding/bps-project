import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) < 2:
    print("ERROR: scene name required, e.g. python3 compare_depths.py world1_baseline")
    sys.exit(1)

scene = sys.argv[1]

# -----------------------------
# Load RGB image
# -----------------------------
rgb = cv2.imread(f"rgb_images/{scene}_rgb.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# -----------------------------
# Load Gazebo ground truth
# -----------------------------
gt_depth = np.load(f"depth_maps/{scene}_depth.npy")

# -----------------------------
# Load predicted depth
# -----------------------------
pred_depth = np.load(f"predicted_depth/{scene}_depth.npy")

# -----------------------------
# Normalise for display only
# -----------------------------
gt_vis = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())

pred_vis = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gt_vis, cmap="plasma")
plt.title("Gazebo Ground Truth")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_vis, cmap="plasma")
plt.title("Depth Anything V2")
plt.axis("off")

plt.tight_layout()

plt.savefig(f"predicted_depth/{scene}_comparison.png", dpi=300)

plt.show()