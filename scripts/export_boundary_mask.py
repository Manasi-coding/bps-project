import sys
import numpy as np
import cv2
from pathlib import Path

SCENE_LABELS = {
    "world1_primitives": {
        1: "usb",
        2: "bottle",
        3: "pen",
        4: "ruler",
    },

    "world2_household": {
        1: "coffee_mug",
        2: "smartphone",
        3: "apple",
        4: "remote_control",
    },

    "world3_kitchen_objects": {
        1: "mug",
        2: "bowl",
        3: "bottle",
        4: "spoon",
    },

    "world4_office_objects": {
        1: "usb_drive",
        2: "mouse",
        3: "calculator",
        4: "stapler",
    },

    "world5_mixed_clutter": {
        1: "mug",
        2: "bottle",
        3: "book",
        4: "calculator",
        5: "usb_drive",
        6: "bowl",
        7: "small_box",
    },

    "world6_occlusion": {
        1: "laptop",
        2: "coffee_cup",
        3: "smartphone",
        4: "notebook",
        5: "tape_dispenser",
        6: "water_bottle",
        7: "sunglasses",
    }
}
RING_WIDTH = 5  # must match export_masks.py for consistent boundary semantics; used only in fallback

scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"

LABELS = SCENE_LABELS[scene_name]

masks_dir = Path('data/masks') / scene_name
rgb_path = Path('rgb_images') / f'{scene_name}_rgb.png'
out_dir = Path('boundary_masks') / scene_name
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[{scene_name}] Looking for semantic masks in {masks_dir}...")

if not masks_dir.exists():
    print(f"ERROR [{scene_name}]: Mask directory {masks_dir} does not exist — run export_masks.py first")
    sys.exit(1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_WIDTH, RING_WIDTH))

combined_boundary = None
found = []

for label_id, name in LABELS.items():
    mask_path = masks_dir / f'{name}_mask.npy'
    boundary_path = masks_dir / f'{name}_boundary.npy'

    if boundary_path.exists():
        ring = np.load(boundary_path).astype(np.uint8)
        if ring.ndim != 2 or np.count_nonzero(ring) == 0:
            print(f"[{scene_name}]   WARNING: {name}_boundary.npy is empty or malformed — skipping")
            continue
        source = "reused"
    elif mask_path.exists():
        mask = np.load(mask_path).astype(np.uint8)
        if mask.ndim != 2 or np.count_nonzero(mask) == 0:
            print(f"[{scene_name}]   WARNING: {name}_mask.npy is empty or malformed — skipping")
            continue
        ring = mask - cv2.erode(mask, kernel)
        source = "regenerated"
    else:
        continue  # object not present in this scene, not an error

    if combined_boundary is None:
        combined_boundary = np.zeros_like(ring)
    combined_boundary = np.maximum(combined_boundary, ring)

    np.save(out_dir / f'{name}_boundary.npy', ring)
    cv2.imwrite(str(out_dir / f'{name}_boundary.png'), ring * 255)

    print(f"[{scene_name}]   ✓ {name} ({source}): boundary={np.count_nonzero(ring)}px")
    found.append(name)

if not found:
    print(f"ERROR [{scene_name}]: No valid semantic masks found in {masks_dir} — cannot produce boundaries")
    sys.exit(1)

np.save(out_dir / f'{scene_name}_combined_boundary.npy', combined_boundary)
cv2.imwrite(str(out_dir / f'{scene_name}_combined_boundary.png'), combined_boundary * 255)
print(f"[{scene_name}] Combined boundary saved: "
      f"{np.count_nonzero(combined_boundary)} / {combined_boundary.size} px "
      f"({100 * (combined_boundary > 0).mean():.1f}%)")

if rgb_path.exists():
    rgb = cv2.imread(str(rgb_path))
    if rgb is not None:
        rgb_resized = cv2.resize(rgb, (combined_boundary.shape[1], combined_boundary.shape[0]))
        overlay = rgb_resized.copy()
        overlay[combined_boundary > 0] = [0, 0, 255]
        cv2.imwrite(str(out_dir / f'{scene_name}_overlay.png'), overlay)
        print(f"[{scene_name}] Overlay saved with {np.count_nonzero(combined_boundary)} boundary pixels")
    else:
        print(f"[{scene_name}] WARNING: RGB file found but could not be read — skipping overlay")
else:
    print(f"[{scene_name}] No RGB image found at {rgb_path} — skipping overlay (not fatal)")

print(f"[{scene_name}] Done. {len(found)} object boundaries saved to {out_dir}")
