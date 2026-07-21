import sys, time
import numpy as np
import cv2
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# Label IDs below were extracted directly from each world's
# <plugin filename="gz-sim-label-system" ...><label>N</label></plugin>
# entries (NOT guessed from model order), so they match exactly what the
# semantic segmentation camera will actually output for each pixel.
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
RING_WIDTH = 5
TIMEOUT = 20


def write_masks_for_labels(label_ch, labels, out_dir, scene_name):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_WIDTH, RING_WIDTH))
    written = []
    for label_id, name in labels.items():
        mask = (label_ch == label_id).astype(np.uint8)
        eroded = cv2.erode(mask, kernel)
        ring = mask - eroded
        cv2.imwrite(str(out_dir / f'{name}_mask.png'), mask * 255)
        cv2.imwrite(str(out_dir / f'{name}_boundary.png'), ring * 255)
        np.save(out_dir / f'{name}_mask.npy', mask)
        np.save(out_dir / f'{name}_boundary.npy', ring)
        pixel_count = np.count_nonzero(mask)
        if pixel_count == 0:
            print(f"[{scene_name}]   ! {name}: no pixels in semantic frame; wrote empty mask")
        else:
            print(f"[{scene_name}]   ✓ {name}: mask={pixel_count}px  ring={np.count_nonzero(ring)}px")
        written.append(name)
    return written


import re


def infer_viewpoint(s):
    """Scene names follow '{world_id}__v{N}' for N>=2, unprefixed for viewpoint 1."""
    m = re.search(r'__v(\d+)$', s)
    return int(m.group(1)) if m else 1


def base_world_id(s):
    """Strip the '__v{N}' viewpoint suffix to get the underlying world id
    used as the SCENE_LABELS key (object labels are viewpoint-independent)."""
    return re.sub(r'__v\d+$', '', s)


scene = sys.argv[1]
world_id = base_world_id(scene)
viewpoint = infer_viewpoint(scene)

if world_id not in SCENE_LABELS:
    print(f"ERROR: No label mapping defined for {world_id} (derived from scene '{scene}')")
    sys.exit(1)

LABELS = SCENE_LABELS[world_id]

# Optional 2nd CLI arg overrides the topic explicitly. If omitted, the topic
# is derived from the scene name using the naming convention from Part 1:
#   viewpoint 1        -> /semantic_camera/labels_map
#   viewpoint N (N>=2)  -> /semantic_camera_v{N}/labels_map
if len(sys.argv) > 2:
    SEMANTIC_TOPIC = sys.argv[2]
else:
    SEMANTIC_TOPIC = '/semantic_camera/labels_map' if viewpoint == 1 else f'/semantic_camera_v{viewpoint}/labels_map'

# Masks are written under the full (viewpoint-qualified) scene name so that
# different viewpoints of the same world don't clobber each other.
out_dir = Path('data/masks') / scene
out_dir.mkdir(parents=True, exist_ok=True)

node = Node()
received = []

def cb(msg):
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        total_pixels = msg.height * msg.width
        if total_pixels == 0 or len(raw) % total_pixels != 0:
            return  # malformed frame, skip
        channels = len(raw) // total_pixels
        arr = raw.reshape(msg.height, msg.width, channels)
        received.append((msg.header.stamp.sec, arr, channels))
    except Exception as e:
        print(f"[{scene}] WARNING: failed to parse semantic frame: {e}")

node.subscribe(Image, SEMANTIC_TOPIC, cb)

print(f"[{scene}] Waiting for semantic frame on {SEMANTIC_TOPIC} (timeout {TIMEOUT}s)...")
start = time.time()
while not received and time.time() - start < TIMEOUT:
    time.sleep(0.1)

if not received:
    print(f"ERROR [{scene}]: No semantic frames received within {TIMEOUT}s — is Gazebo running and not paused?")
    sys.exit(1)

ts, frame, channels = received[-1]

if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0 or channels < 1:
    print(f"ERROR [{scene}]: Invalid semantic frame dimensions/channels — shape={frame.shape}, channels={channels}")
    sys.exit(1)

print(f"[{scene}] Frame t={ts}s  shape={frame.shape}  channels={channels}")

# semantic = label is in channel 0 directly
label_ch = frame[:, :, 0]
print(f"[{scene}] Unique label values: {np.unique(label_ch)}")

found = write_masks_for_labels(label_ch, LABELS, out_dir, scene)

if not found:
    print(f"ERROR [{scene}]: No labeled objects found — check label plugin in SDF")
    sys.exit(1)

print(f"[{scene}] Saved {len(found)} objects to {out_dir}")