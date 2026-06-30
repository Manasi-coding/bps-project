import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

Path('ground_truth').mkdir(exist_ok=True)

WORLD_DIR = Path('worlds')
VALID_SCENES = [
    "world1_primitives",
    "world2_household",
    "world3_kitchen_objects",
    "world4_office_objects",
    "world5_mixed_clutter",
    "world6_occlusion",
]

# Per-scene allow-list of graspable object model names, keyed by label ID.
# IMPORTANT: these names must exactly match the <model name="..."> attribute
# in the corresponding world's SDF file, including any suffixes (e.g. "_sem1",
# "_body") that Gazebo or the semantic-segmentation plugin may append. Verify
# against the actual .sdf files — these values have not been confirmed against
# real world files as of this version.
SCENE_LABELS = {
    "world1_primitives": {
        1: "obj_cube",
        2: "obj_cylinder",
        3: "obj_sphere",
        4: "obj_cone",
    },
    
    "world2_household": {
        1: "coffee_mug_sem1",
        2: "smartphone_sem2",
        3: "apple_sem3",
        4: "remote_control_sem4",
    },

    "world3_kitchen_objects": {
        1: "mug_body_sem1",
        2: "bowl_sem2",
        3: "bottle_body_sem3",
        4: "spoon_sem4",
    },

    "world4_office_objects": {
        1: "usb_drive_sem1",
        2: "mouse_sem2",
        3: "calculator_sem3",
        4: "stapler_sem4",
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

        objects.append({
            "name": name,
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
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