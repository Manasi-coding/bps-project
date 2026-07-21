import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

model = DepthAnythingV2(**MODEL_CONFIGS["vits"])

model.load_state_dict(
    torch.load(
        "checkpoints/depth_anything_v2_vits.pth",
        map_location=DEVICE,
    )
)

model.to(DEVICE)
model.eval()

if len(sys.argv) < 2:
    print("ERROR: scene name required, e.g. python3 run_depth_anything.py world1_baseline")
    sys.exit(1)

scene = sys.argv[1]

image_path = f"rgb_images/{scene}_rgb.png"

print(f"Loading {image_path}")

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

print(f"Image shape: {image.shape}")

depth = model.infer_image(image)

print(f"Depth shape: {depth.shape}")

np.save(
    f"predicted_depth/{scene}_depth.npy",
    depth
)

depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
depth_vis = (depth_vis * 255).astype(np.uint8)

cv2.imwrite(
    f"predicted_depth/{scene}_depth.png",
    depth_vis
)

print("Depth map saved successfully.")