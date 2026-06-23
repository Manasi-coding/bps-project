import sys
import time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from pathlib import Path

Path('depth_maps').mkdir(exist_ok=True)

scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"
saved = False

def callback(msg):
    global saved
    if saved:
        return
    data = np.frombuffer(msg.data, dtype=np.float32)
    data = data.reshape(msg.height, msg.width)
    np.save(f'depth_maps/{scene_name}_depth.npy', data)
    valid = data[np.isfinite(data) & (data > 0)]
    print(f"Saved {scene_name}: "
          f"min={valid.min():.3f}m "
          f"max={valid.max():.3f}m "
          f"mean={valid.mean():.3f}m")
    saved = True

node = Node()
node.subscribe(Image, '/depth_camera', callback)

start = time.time()
while not saved and time.time() - start < 5:
    time.sleep(0.1)

if not saved:
    print("ERROR: No messages received in 5 seconds")
    print("Check Gazebo is running and not paused")
