import sys
import cv2
import numpy as np
from pathlib import Path

DEPTH_DIR = Path("depth_maps")
BOUNDARY_DIR = Path("boundary_masks")
SEMANTIC_MASK_DIR = Path("data/masks")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def boundary_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union > 0 else 0.0


def interior_rmse(d1, d2, boundary_mask):
    interior = ~boundary_mask
    diff = d1[interior] - d2[interior]
    diff = diff[np.isfinite(diff)]
    return np.sqrt((diff ** 2).mean()) if len(diff) > 0 else 0.0


def discover_scenes(boundary_dir):
    """Find scene names from subdirectories of boundary_masks/ containing a combined boundary mask."""
    if not boundary_dir.exists():
        print(f"ERROR: {boundary_dir} does not exist")
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


def discover_object_masks(scene):
    """Find all per-object boundary mask files for a scene (anything matching *_boundary.png, excluding the combined one)."""
    scene_dir = SEMANTIC_MASK_DIR / scene
    if not scene_dir.exists():
        return []
    objects = []
    for f in sorted(scene_dir.glob("*_boundary.png")):
        name = f.stem.replace("_boundary", "")
        objects.append(name)
    return objects


scenes = discover_scenes(BOUNDARY_DIR)

if not scenes:
    print("ERROR: no scenes discovered under boundary_masks/ — aborting")
    sys.exit(1)

print(f"Discovered {len(scenes)} scene(s): {scenes}\n")

report_lines = []
report_lines.append("Boundary Independence Check — Baseline Setup\n")
report_lines.append("=" * 50 + "\n\n")

any_failed = False
total_objects_loaded = 0

for scene in scenes:
    print(f"[{scene}] Loading inputs...")
    report_lines.append(f"Scene: {scene}\n")

    depth_path = DEPTH_DIR / f"{scene}_depth.npy"
    if not depth_path.exists():
        print(f"ERROR [{scene}]: missing depth map {depth_path}")
        report_lines.append(f"  ERROR: missing depth map {depth_path}\n\n")
        any_failed = True
        continue

    try:
        depth_clean = np.load(depth_path)
    except Exception as e:
        print(f"ERROR [{scene}]: failed to load depth map: {e}")
        report_lines.append(f"  ERROR: failed to load depth map: {e}\n\n")
        any_failed = True
        continue

    object_names = discover_object_masks(scene)
    if not object_names:
        print(f"ERROR [{scene}]: no per-object boundary masks found in {SEMANTIC_MASK_DIR / scene}")
        report_lines.append(f"  ERROR: no per-object boundary masks found\n\n")
        any_failed = True
        continue

    print(f"  Depth map shape: {depth_clean.shape}")
    report_lines.append(f"  Depth map shape: {depth_clean.shape}\n")
    report_lines.append(f"  Objects found: {object_names}\n")

    scene_failed = False
    for obj_name in object_names:
        mask_path = SEMANTIC_MASK_DIR / scene / f"{obj_name}_boundary.png"
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            print(f"  ERROR: failed to read {mask_path}")
            report_lines.append(f"    {obj_name}: ERROR — failed to read mask\n")
            scene_failed = True
            continue

        boundary_mask = mask_img > 0

        if boundary_mask.shape != depth_clean.shape:
            print(f"  ERROR: {obj_name} mask shape {boundary_mask.shape} != depth shape {depth_clean.shape}")
            report_lines.append(
                f"    {obj_name}: ERROR — shape mismatch "
                f"(mask={boundary_mask.shape}, depth={depth_clean.shape})\n"
            )
            scene_failed = True
            continue

        boundary_pixels = int(boundary_mask.sum())
        print(f"  ✓ {obj_name}: boundary pixels = {boundary_pixels}")
        report_lines.append(f"    {obj_name}: boundary pixels = {boundary_pixels}\n")
        total_objects_loaded += 1

        # NOTE: boundary_iou / interior_rmse intentionally not called here.
        # These metrics require a second depth/mask source to compare against
        # (e.g. corrupted depth maps), which doesn't exist yet. This script
        # only validates and loads the baseline data structures for now.

    if scene_failed:
        any_failed = True
    report_lines.append("\n")

report_lines.append("=" * 50 + "\n")
report_lines.append(f"Scenes processed: {len(scenes)}\n")
report_lines.append(f"Objects loaded successfully: {total_objects_loaded}\n")
report_lines.append(
    "\nMetric computation (boundary_iou, interior_rmse) was skipped:\n"
    "no second depth/mask source (e.g. corrupted depth maps) exists yet to compare against.\n"
    "This script currently validates baseline structure only.\n"
)

report_path = RESULTS_DIR / "independence_check_baseline.txt"
with open(report_path, "w") as f:
    f.writelines(report_lines)

print(f"\nBaseline report saved to {report_path}")
print(f"✓ Script ready: loaded {total_objects_loaded} object(s) across {len(scenes)} scene(s)")

if any_failed:
    print("FAILED: one or more scenes/objects had missing or invalid inputs")
    sys.exit(1)
