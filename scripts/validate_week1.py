# Run this on Day 7 to confirm everything exists
from pathlib import Path
import numpy as np
import json

scenes = ['isolated_usb', 'isolated_bottle', 'isolated_pen',
          'touching', 'occluded', 'thin_objects']
errors = []

for scene in scenes:
    checks = {
        f'depth_maps/{scene}_depth.npy':       'depth',
        f'rgb_images/{scene}_rgb.png':          'rgb',
        f'ground_truth/{scene}_pose.json':      'pose',
        f'boundary_masks/{scene}_mask.png':     'mask',
        f'boundary_masks/{scene}_overlay.png':  'overlay',
        f'boundary_masks/{scene}_grad.npy':     'gradient',
    }
    for path, label in checks.items():
        if not Path(path).exists():
            errors.append(f"MISSING [{scene}] {label}: {path}")

    # Verify depth values are metric
    depth_path = f'depth_maps/{scene}_depth.npy'
    if Path(depth_path).exists():
        d = np.load(depth_path)
        valid = d[d > 0]
        if len(valid) == 0:
            errors.append(f"EMPTY depth [{scene}]")
        elif valid.min() > 2.0 or valid.max() < 0.1:
            errors.append(f"SUSPICIOUS depth range [{scene}]: "
                         f"{valid.min():.3f}–{valid.max():.3f}m")
        else:
            print(f"OK [{scene}] depth: "
                  f"{valid.min():.3f}–{valid.max():.3f}m")

    # Verify pose JSON is valid
    pose_path = f'ground_truth/{scene}_pose.json'
    if Path(pose_path).exists():
        with open(pose_path) as f:
            p = json.load(f)
        print(f"OK [{scene}] pose: {len(p['objects'])} objects")

if errors:
    print("\nFAILED:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nALL CHECKS PASSED — Week 1 complete")