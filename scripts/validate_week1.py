# Run this after the automated pipeline to confirm everything exists
import sys
from pathlib import Path
import numpy as np
import json

BOUNDARY_DIR = Path("boundary_masks")


def discover_scenes(boundary_dir):
    """Discover processed scenes from boundary_masks/ subdirectories containing a combined boundary mask."""
    if not boundary_dir.exists():
        print(f"ERROR: {boundary_dir} does not exist — nothing to validate")
        return []
    scenes = []
    for entry in sorted(boundary_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / f"{entry.name}_combined_boundary.png").exists():
            scenes.append(entry.name)
        else:
            print(f"  (skipping {entry}: no combined boundary mask found)")
    return scenes


scenes = discover_scenes(BOUNDARY_DIR)

if not scenes:
    print("FAILED: no scenes discovered under boundary_masks/")
    sys.exit(1)

print(f"Discovered {len(scenes)} scene(s): {scenes}\n")

errors = []

for scene in scenes:
    print(f"[{scene}] Validating...")

    checks = {
        f'rgb_images/{scene}_rgb.png':                                     'rgb',
        f'depth_maps/{scene}_depth.npy':                                   'depth',
        f'ground_truth/{scene}_pose.json':                                 'pose',
        f'boundary_masks/{scene}/{scene}_combined_boundary.png':           'combined boundary (png)',
        f'boundary_masks/{scene}/{scene}_combined_boundary.npy':           'combined boundary (npy)',
        f'boundary_masks/{scene}/{scene}_overlay.png':                     'overlay',
    }

    for path, label in checks.items():
        if not Path(path).exists():
            msg = f"MISSING [{scene}] {label}: {path}"
            errors.append(msg)
            print(f"  ✗ {msg}")

    # Verify semantic mask directory has at least one object mask
    mask_dir = Path(f'data/masks/{scene}')
    if not mask_dir.exists() or not any(mask_dir.glob("*_mask.npy")):
        msg = f"MISSING [{scene}] semantic object masks in {mask_dir}"
        errors.append(msg)
        print(f"  ✗ {msg}")

    # Verify depth values are metric
    depth_path = f'depth_maps/{scene}_depth.npy'
    if Path(depth_path).exists():
        try:
            d = np.load(depth_path)
            valid = d[d > 0]
            if len(valid) == 0:
                msg = f"EMPTY depth [{scene}]"
                errors.append(msg)
                print(f"  ✗ {msg}")
            elif valid.min() > 2.0 or valid.max() < 0.1:
                msg = (f"SUSPICIOUS depth range [{scene}]: "
                       f"{valid.min():.3f}–{valid.max():.3f}m")
                errors.append(msg)
                print(f"  ✗ {msg}")
            else:
                print(f"  ✓ depth: {valid.min():.3f}–{valid.max():.3f}m")
        except Exception as e:
            msg = f"ERROR [{scene}]: failed to load depth map: {e}"
            errors.append(msg)
            print(f"  ✗ {msg}")

    # Verify pose JSON is valid
    pose_path = f'ground_truth/{scene}_pose.json'
    if Path(pose_path).exists():
        try:
            with open(pose_path) as f:
                p = json.load(f)
            if 'objects' not in p:
                msg = f"INVALID pose JSON [{scene}]: missing 'objects' key"
                errors.append(msg)
                print(f"  ✗ {msg}")
            elif len(p['objects']) == 0:
                msg = f"EMPTY pose [{scene}]: 0 objects"
                errors.append(msg)
                print(f"  ✗ {msg}")
            else:
                print(f"  ✓ pose: {len(p['objects'])} objects")
        except json.JSONDecodeError as e:
            msg = f"INVALID pose JSON [{scene}]: {e}"
            errors.append(msg)
            print(f"  ✗ {msg}")

    print()

if errors:
    print("FAILED:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
