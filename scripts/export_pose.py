import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

Path('ground_truth').mkdir(exist_ok=True)

WORLD_DIR = Path('worlds')
VALID_SCENES = [
    "world1_baseline",
    "world2_dense_clutter",
    "world3_thin_objects",
    "world4_support_scene",
    "world5_occlusion_scene",
    "world6_dense_mixed",
]

# Per-scene allow-list of graspable object model names, keyed by label ID.
# These names have been verified directly against the <model name="..."> attributes
# in the corresponding world's SDF file (all "obj_*" models, in the order they
# appear in the world file). None of these worlds use "_semN" suffixes.
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
        14: "obj_chopstick_b",
        15: "obj_straw",
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


def validate_scene_labels(valid_scenes, scene_labels):
    """Ensure every scene in valid_scenes has a corresponding entry in scene_labels."""
    missing = [s for s in valid_scenes if s not in scene_labels]
    if missing:
        print(f"ERROR: VALID_SCENES contains scene(s) with no SCENE_LABELS entry: {missing}")
        print("Fix SCENE_LABELS before running this script.")
        sys.exit(1)


def parse_pose_string(pose_text):
    """SDF <pose> is 'x y z roll pitch yaw' (space-separated, radians)."""
    parts = pose_text.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(parts)}: {pose_text!r}")
    x, y, z, roll, pitch, yaw = (float(p) for p in parts)
    return x, y, z, roll, pitch, yaw

def extract_dimensions(model):
    """
    Extract object dimensions from the model geometry.

    Priority:
      1. First collision geometry
      2. First visual geometry

    Returns:
        (width, height, depth) in metres
    """

    geometry = None

    # -------------------------
    # Try collision first
    # -------------------------
    collision = model.find("link/collision")
    if collision is not None:
        geometry = collision.find("geometry")

    # -------------------------
    # If no collision geometry,
    # fall back to first visual
    # -------------------------
    if geometry is None:
        visual = model.find("link/visual")
        if visual is not None:
            geometry = visual.find("geometry")

    if geometry is None:
        return None, None, None

    # -------------------------
    # Box
    # -------------------------
    box = geometry.find("box")
    if box is not None:
        size = box.find("size")
        if size is not None:
            w, d, h = map(float, size.text.split())
            return w, h, d

    # -------------------------
    # Cylinder
    # -------------------------
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.find("radius").text)
        length = float(cylinder.find("length").text)
        diameter = radius * 2.0
        return diameter, length, diameter

    # -------------------------
    # Sphere
    # -------------------------
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.find("radius").text)
        diameter = radius * 2.0
        return diameter, diameter, diameter

    # -------------------------
    # Cone
    # -------------------------
    cone = geometry.find("cone")
    if cone is not None:
        radius = float(cone.find("radius").text)
        length = float(cone.find("length").text)
        diameter = radius * 2.0
        return diameter, length, diameter

    return None, None, None


def extract_objects_from_sdf(sdf_path, known_objects):
    """
    Parse a Gazebo SDF world file and extract poses for models whose name
    is in known_objects. Returns a list of
    {"name", "x", "y", "z", "roll", "pitch", "yaw"} dicts.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {sdf_path}")

    objects = []
    skipped_unknown = []

    for model in world.findall("model"):
        name = model.get("name")
        if name is None:
            continue
        if name not in known_objects:
            skipped_unknown.append(name)
            continue

        pose_el = model.find("pose")
        if pose_el is None or not pose_el.text:
            print(f"  WARNING: model '{name}' has no <pose> element — skipping")
            continue

        try:
            x, y, z, roll, pitch, yaw = parse_pose_string(pose_el.text)
        except ValueError as e:
            print(f"  WARNING: model '{name}' has malformed pose ({e}) — skipping")
            continue

        width, height, depth = extract_dimensions(model)

        objects.append({
            "name": name,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,

            "width": width,
            "height": height,
            "depth": depth,
        })

    if skipped_unknown:
        print(f"  (ignored {len(skipped_unknown)} non-object models: {skipped_unknown})")

    return objects


validate_scene_labels(VALID_SCENES, SCENE_LABELS)

scene_name = sys.argv[1] if len(sys.argv) > 1 else None

if scene_name is None:
    scenes_to_process = VALID_SCENES
else:
    if scene_name not in VALID_SCENES:
        print(f"ERROR: unknown scene '{scene_name}'")
        print(f"Valid scenes: {VALID_SCENES}")
        sys.exit(1)
    scenes_to_process = [scene_name]

any_failed = False

for name in scenes_to_process:
    sdf_path = WORLD_DIR / f"{name}.sdf"
    print(f"[{name}] Extracting object poses from {sdf_path}...")

    if not sdf_path.exists():
        print(f"ERROR [{name}]: world file {sdf_path} not found")
        any_failed = True
        continue

    known_objects = set(SCENE_LABELS[name].values())

    try:
        objects = extract_objects_from_sdf(sdf_path, known_objects)
    except ET.ParseError as e:
        print(f"ERROR [{name}]: failed to parse {sdf_path} as XML: {e}")
        any_failed = True
        continue
    except ValueError as e:
        print(f"ERROR [{name}]: {e}")
        any_failed = True
        continue

    if not objects:
        print(f"ERROR [{name}]: no known objects (from SCENE_LABELS['{name}']) found in {sdf_path}")
        any_failed = True
        continue

    output = {"scene": name, "objects": objects}
    out_path = Path('ground_truth') / f'{name}_pose.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[{name}] Saved {out_path} ({len(objects)} objects)")

if scene_name is not None and any_failed:
    sys.exit(1)