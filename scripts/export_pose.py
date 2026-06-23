import json
import sys
from pathlib import Path

Path('ground_truth').mkdir(exist_ok=True)

SCENE_POSES = {
    "isolated_usb": [
        {"name": "usb", "x": 0.0, "y": 0.0, "z": 0.005,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    ],
    "isolated_bottle": [
        {"name": "bottle", "x": 0.0, "y": 0.0, "z": 0.1,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    ],
    "isolated_pen": [
        {"name": "pen", "x": 0.0, "y": 0.0, "z": 0.075,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    ],
    "touching": [
        {"name": "usb",         "x": -0.08, "y":  0.10, "z": 0.005,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "marker",      "x": -0.04, "y":  0.10, "z": 0.07,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "credit_card", "x":  0.08, "y": -0.05, "z": 0.001,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "ruler",       "x":  0.08, "y": -0.09, "z": 0.0025,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "bottle",      "x":  0.15, "y":  0.15, "z": 0.10,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "pen",         "x": -0.15, "y": -0.15, "z": 0.075,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    ],
    "occluded": [
        {"name": "usb",         "x":  0.0,  "y":  0.0,   "z": 0.005,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "bottle",      "x":  0.04, "y":  0.0,   "z": 0.1,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "credit_card", "x":  0.0,  "y": -0.08,  "z": 0.001,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "ruler",       "x":  0.0,  "y": -0.15,  "z": 0.0025,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "pen",         "x": -0.08, "y":  0.08,  "z": 0.075,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "marker",      "x":  0.12, "y": -0.12,  "z": 0.07,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    ],
    "thin_objects": [
        {"name": "credit_card", "x":  0.0,  "y": -0.08, "z": 0.001,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"name": "ruler",       "x":  0.12, "y":  0.0,  "z": 0.0025,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.35},
        {"name": "pen",         "x": -0.10, "y":  0.08, "z": 0.075,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.785}
    ]
}

scene_name = sys.argv[1] if len(sys.argv) > 1 else None

if scene_name:
    if scene_name not in SCENE_POSES:
        print(f"ERROR: unknown scene '{scene_name}'")
        print(f"Valid scenes: {list(SCENE_POSES.keys())}")
        sys.exit(1)
    scenes = {scene_name: SCENE_POSES[scene_name]}
else:
    scenes = SCENE_POSES

for name, objects in scenes.items():
    output = {"scene": name, "objects": objects}
    path = f'ground_truth/{name}_pose.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {path}")