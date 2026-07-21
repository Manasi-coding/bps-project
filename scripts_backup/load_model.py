import sys
from pathlib import Path

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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = DepthAnythingV2(**MODEL_CONFIGS["vits"])

checkpoint = Path("checkpoints/depth_anything_v2_vits.pth")

state_dict = torch.load(checkpoint, map_location=device)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("Model loaded successfully!")