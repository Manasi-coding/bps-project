import sys
import numpy as np
import cv2
from pathlib import Path

Path('boundary_masks').mkdir(exist_ok=True)

scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"

# Load raw depth
depth_path = f'depth_maps/{scene_name}_depth.npy'
rgb_path   = f'rgb_images/{scene_name}_rgb.png'

depth = np.load(depth_path).astype(np.float32)

# Replace zeros with NaN for cleaner gradients
depth_clean = depth.copy()
depth_clean[depth_clean == 0] = np.nan

# Sobel gradient on metric depth
gx = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
grad = np.sqrt(gx**2 + gy**2)
grad = np.nan_to_num(grad, nan=0.0)

# Save raw gradient
np.save(f'boundary_masks/{scene_name}_grad.npy', grad)

# Threshold — 0.05 metres/pixel gradient = boundary
THRESHOLD = 0.01
mask = (grad > THRESHOLD).astype(np.uint8) * 255

# Save binary mask
cv2.imwrite(f'boundary_masks/{scene_name}_mask.png', mask)

# Save overlay on RGB for visual inspection
if Path(rgb_path).exists():
    rgb = cv2.imread(rgb_path)
    rgb_resized = cv2.resize(rgb, (mask.shape[1], mask.shape[0]))
    overlay = rgb_resized.copy()
    overlay[mask > 0] = [0, 0, 255]  # red overlay on boundaries
    cv2.imwrite(f'boundary_masks/{scene_name}_overlay.png', overlay)
    print(f"Overlay saved: boundary pixels = {(mask>0).sum()}")

print(f"Mask saved: {scene_name}")
print(f"Gradient range: min={grad.min():.4f} max={grad.max():.4f}")
print(f"Boundary pixels: {(mask>0).sum()} / {mask.size} "
      f"({100*(mask>0).mean():.1f}%)")