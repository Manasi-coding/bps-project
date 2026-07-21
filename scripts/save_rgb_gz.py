import re
import sys
import time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from pathlib import Path

Path('rgb_images').mkdir(exist_ok=True)


def infer_viewpoint(scene):
    """Scene names follow '{world_id}__v{N}' for N>=2, unprefixed for viewpoint 1."""
    m = re.search(r'__v(\d+)$', scene)
    return int(m.group(1)) if m else 1


scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"
# Optional 2nd CLI arg overrides the topic explicitly. If omitted, the topic
# is derived from the scene name using the naming convention from Part 1:
#   viewpoint 1        -> /rgb_camera
#   viewpoint N (N>=2)  -> /rgb_camera_v{N}
if len(sys.argv) > 2:
    RGB_TOPIC = sys.argv[2]
else:
    viewpoint = infer_viewpoint(scene_name)
    RGB_TOPIC = '/rgb_camera' if viewpoint == 1 else f'/rgb_camera_v{viewpoint}'

TIMEOUT = 60

saved = False

def callback(msg):
    global saved
    if saved:
        return
    data = np.frombuffer(msg.data, dtype=np.uint8)
    data = data.reshape(msg.height, msg.width, 3)
    print(f"[{scene_name}] RGB shape = {data.shape}")
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'rgb_images/{scene_name}_rgb.png', bgr)
    print(f"[{scene_name}] Saved rgb_images/{scene_name}_rgb.png — shape: {data.shape}")
    saved = True

node = Node()
node.subscribe(Image, RGB_TOPIC, callback)

print(f"[{scene_name}] Waiting for RGB frame on {RGB_TOPIC} (timeout {TIMEOUT}s)...")
start = time.time()
while not saved and time.time() - start < TIMEOUT:
    time.sleep(0.1)

if not saved:
    print(f"ERROR [{scene_name}]: No RGB messages received on {RGB_TOPIC} within {TIMEOUT} seconds")
    sys.exit(1)
