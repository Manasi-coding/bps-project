import sys
import time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from pathlib import Path

Path('rgb_images').mkdir(exist_ok=True)

scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene"
saved = False

def callback(msg):
    global saved
    if saved:
        return
    data = np.frombuffer(msg.data, dtype=np.uint8)
    data = data.reshape(msg.height, msg.width, 3)
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'rgb_images/{scene_name}_rgb.png', bgr)
    print(f"Saved {scene_name}_rgb.png — shape: {data.shape}")
    saved = True

node = Node()
node.subscribe(Image, '/rgb_camera', callback)

start = time.time()
while not saved and time.time() - start < 5:
    time.sleep(0.1)

if not saved:
    print("ERROR: No messages received in 5 seconds")
