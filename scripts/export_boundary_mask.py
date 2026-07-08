import sys
import numpy as np
import cv2
from pathlib import Path

# Kept identical to export_masks.py's SCENE_LABELS so that the {name}_mask.npy
# / {name}_boundary.npy filenames this script looks for always match what
# export_masks.py actually wrote to data/masks/<scene>/.
SCENE_LABELS = {
    "world1_baseline": {
        1: "obj_bottle",
        2: "obj_mug",
        3: "obj_bowl",
        4: "obj_apple",
        5: "obj_spoon",
        6: "obj_plate",
        7: "obj_tissue_box",
        8: "obj_cereal_box",
        9: "obj_notebook",
        10: "obj_remote",
        11: "obj_toy_block",
        12: "obj_banana",
    },

    "world2_dense_clutter": {
        1: "obj_bottle",
        2: "obj_mug",
        3: "obj_tissue_box",
        4: "obj_jar",
        5: "obj_cereal_box",
        6: "obj_banana",
        7: "obj_bowl",
        8: "obj_apple",
        9: "obj_tomato",
        10: "obj_plate",
        11: "obj_spoon",
        12: "obj_fork",
        13: "obj_kettle",
        14: "obj_cup",
        15: "obj_chopping_board",
        16: "obj_knife",
    },

    "world3_thin_objects": {
        1: "obj_plate",
        2: "obj_fork",
        3: "obj_bottle",
        4: "obj_mug",
        5: "obj_toothbrush",
        6: "obj_notebook",
        7: "obj_pen",
        8: "obj_pencil",
        9: "obj_knife",
        10: "obj_spoon",
        11: "obj_ruler",
        12: "obj_cable",
        13: "obj_chopstick_a",
        14: "obj_straw",
        18: "obj_chopstick_b",
    },

    "world4_support_scene": {
        1: "obj_cereal_box",
        2: "obj_tissue_box",
        3: "obj_storage_box",
        4: "obj_wooden_crate",
        5: "obj_bottle",
        6: "obj_mug",
        7: "obj_bowl",
        8: "obj_plate",
        9: "obj_notebook",
        10: "obj_book",
        11: "obj_toy_cube",
        12: "obj_sponge",
        13: "obj_can",
        14: "obj_glass_jar",
        15: "obj_container",
    },

    "world5_occlusion_scene": {
        1: "obj_cereal_box",
        2: "obj_bottle",
        3: "obj_jar",
        4: "obj_tissue_box",
        5: "obj_cup",
        6: "obj_bowl",
        7: "obj_apple",
        8: "obj_can",
        9: "obj_remote",
        10: "obj_notebook",
        11: "obj_book",
        12: "obj_plate",
        13: "obj_mug",
        14: "obj_sponge",
        15: "obj_container",
        16: "obj_toy",
        17: "obj_banana",
    },

    "world6_dense_mixed": {
        1: "obj_bottle",
        2: "obj_mug",
        3: "obj_jar",
        4: "obj_cereal_box",
        5: "obj_tissue_box",
        6: "obj_bowl",
        7: "obj_apple",
        8: "obj_banana",
        9: "obj_notebook",
        10: "obj_remote",
        11: "obj_pen",
        12: "obj_ruler",
        13: "obj_plate",
        14: "obj_spoon",
        15: "obj_fork",
        16: "obj_knife",
    },
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