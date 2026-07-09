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

rgb_dir = Path("rgb_images")
output_dir = Path("predicted_depth")
output_dir.mkdir(exist_ok=True)

rgb_files = sorted(rgb_dir.glob("*_rgb.png"))

print(f"Found {len(rgb_files)} RGB images.\n")

for rgb_path in rgb_files:

    scene = rgb_path.stem.replace("_rgb", "")

    print(f"Processing {scene}...")

    image = cv2.imread(str(rgb_path))

    if image is None:
        print(f"Could not read {rgb_path}")
        continue

    depth = model.infer_image(image)

    np.save(
        output_dir / f"{scene}_depth.npy",
        depth
    )

    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis = (depth_vis * 255).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / f"{scene}_depth.png"),
        depth_vis
    )

    print(f"Saved {scene}")

print("\nFinished processing all scenes.")