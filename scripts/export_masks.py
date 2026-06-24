import sys, time, os
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

LABELS = {1:'usb', 2:'bottle', 3:'pen', 4:'ruler', 5:'marker', 6:'credit_card'}
RING_WIDTH = 5

scene = sys.argv[1]
out_dir = f'/home/student/bps-project/data/masks/{scene}'
os.makedirs(out_dir, exist_ok=True)

node = Node()
received = []

def cb(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    total_pixels = msg.height * msg.width
    channels = len(raw) // total_pixels
    arr = raw.reshape(msg.height, msg.width, channels)
    received.append((msg.header.stamp.sec, arr, channels))

node.subscribe(Image, '/panoptic/labels_map', cb)

print("Flushing stale messages (3s)...")
time.sleep(3)
received.clear()

print(f"Collecting fresh frames for {scene} (5s)...")
time.sleep(5)

if not received:
    print("✗ No frames — is Gazebo running and Play pressed?")
    sys.exit(1)

ts, frame, channels = received[-1]
print(f"  Frame t={ts}s  shape={frame.shape}  channels={channels}")

# semantic = label is in channel 0 directly
label_ch = frame[:, :, 0]
print(f"  Unique label values: {np.unique(label_ch)}")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_WIDTH, RING_WIDTH))

found = []
for label_id, name in LABELS.items():
    mask = (label_ch == label_id).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        continue
    eroded = cv2.erode(mask, kernel)
    ring = mask - eroded
    cv2.imwrite(f'{out_dir}/{name}_mask.png',     mask * 255)
    cv2.imwrite(f'{out_dir}/{name}_boundary.png', ring * 255)
    np.save(f'{out_dir}/{name}_mask.npy',     mask)
    np.save(f'{out_dir}/{name}_boundary.npy', ring)
    print(f"  ✓ {name}: mask={np.count_nonzero(mask)}px  ring={np.count_nonzero(ring)}px")
    found.append(name)

if not found:
    print("  ✗ No labeled objects found — check label plugin in SDF")

print(f"\nSaved {len(found)} objects to {out_dir}")
