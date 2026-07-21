import re
import sys
import time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from pathlib import Path

Path('depth_maps').mkdir(exist_ok=True)


def infer_viewpoint(scene):
    """Scene names follow '{world_id}__v{N}' for N>=2, unprefixed for viewpoint 1."""
    m = re.search(r'__v(\d+)$', scene)
    return int(m.group(1)) if m else 1


scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"
# Optional 2nd CLI arg overrides the topic explicitly. If omitted, the topic
# is derived from the scene name using the naming convention from Part 1:
#   viewpoint 1        -> /depth_camera
#   viewpoint N (N>=2)  -> /depth_camera_v{N}
if len(sys.argv) > 2:
    DEPTH_TOPIC = sys.argv[2]
else:
    viewpoint = infer_viewpoint(scene_name)
    DEPTH_TOPIC = '/depth_camera' if viewpoint == 1 else f'/depth_camera_v{viewpoint}'

TIMEOUT = 60

saved = False
valid_data = False

def callback(msg):
    global saved, valid_data
    if saved:
        return
    data = np.frombuffer(msg.data, dtype=np.float32)
    data = data.reshape(msg.height, msg.width)
    print(f"[{scene_name}] Depth shape = {data.shape}")
    np.save(f'depth_maps/{scene_name}_depth.npy', data)

    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        print(f"ERROR [{scene_name}]: Depth frame received but contains no finite positive values")
        saved = True  # stop waiting; we got a frame, it's just invalid
        valid_data = False
        return

    print(f"[{scene_name}] Saved depth_maps/{scene_name}_depth.npy: "
          f"min={valid.min():.3f}m "
          f"max={valid.max():.3f}m "
          f"mean={valid.mean():.3f}m")
    saved = True
    valid_data = True

node = Node()
node.subscribe(Image, DEPTH_TOPIC, callback)

print(f"[{scene_name}] Waiting for depth frame on {DEPTH_TOPIC} (timeout {TIMEOUT}s)...")
start = time.time()
while not saved and time.time() - start < TIMEOUT:
    time.sleep(0.1)

if not saved:
    print(f"ERROR [{scene_name}]: No depth messages received on {DEPTH_TOPIC} within {TIMEOUT} seconds")
    print(f"[{scene_name}] Check Gazebo is running and not paused")
    sys.exit(1)

if not valid_data:
    print(f"ERROR [{scene_name}]: Depth map saved but contained no valid (finite, positive) measurements")
    sys.exit(1)
