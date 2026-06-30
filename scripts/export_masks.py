import sys, time
import numpy as np
import cv2
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

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
RING_WIDTH = 5
TIMEOUT = 20

scene = sys.argv[1]
if scene not in SCENE_LABELS:
    print(f"ERROR: No label mapping defined for {scene}")
    sys.exit(1)

LABELS = SCENE_LABELS[scene]

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

node.subscribe(Image, '/semantic_camera/labels_map', cb)

print(f"[{scene}] Waiting for semantic frame on /semantic_camera/labels_map (timeout {TIMEOUT}s)...")
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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_WIDTH, RING_WIDTH))
found = []
for label_id, name in LABELS.items():
    mask = (label_ch == label_id).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        continue
    eroded = cv2.erode(mask, kernel)
    ring = mask - eroded
    cv2.imwrite(str(out_dir / f'{name}_mask.png'),     mask * 255)
    cv2.imwrite(str(out_dir / f'{name}_boundary.png'), ring * 255)
    np.save(out_dir / f'{name}_mask.npy',     mask)
    np.save(out_dir / f'{name}_boundary.npy', ring)
    print(f"[{scene}]   ✓ {name}: mask={np.count_nonzero(mask)}px  ring={np.count_nonzero(ring)}px")
    found.append(name)

if not found:
    print(f"ERROR [{scene}]: No labeled objects found — check label plugin in SDF")
    sys.exit(1)

print(f"[{scene}] Saved {len(found)} objects to {out_dir}")
