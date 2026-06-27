
import cv2
import numpy as np
from pathlib import Path

# Create results directory if it doesn't exist
Path("results").mkdir(exist_ok=True)

def boundary_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union > 0 else 0.0

def interior_rmse(d1, d2, boundary_mask):
    interior = ~boundary_mask
    diff = d1[interior] - d2[interior]
    diff = diff[np.isfinite(diff)]
    return np.sqrt((diff ** 2).mean()) if len(diff) > 0 else 0.0

scene = "isolated_usb"

# Load clean depth
depth_clean = np.load(f"depth_maps/{scene}_depth.npy")

# Load GT boundary ring
boundary_mask = (
    cv2.imread(
        f"data/masks/{scene}/usb_boundary.png",
        cv2.IMREAD_GRAYSCALE
    ) > 0
)

print(f"Scene: {scene}")
print(f"Depth map shape: {depth_clean.shape}")
print(f"Boundary mask shape: {boundary_mask.shape}")
print(f"Boundary pixels: {boundary_mask.sum()}")

print("\n✓ Script ready for Day 6")
print("Next step: create corrupted depth maps and compare against this baseline.")

