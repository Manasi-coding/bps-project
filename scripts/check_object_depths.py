#!/usr/bin/env python3
"""
check_object_depths.py
-----------------------
Grabs one live depth frame (ROS 2, /depth_model/depth_image) and one
semantic segmentation frame (Gazebo direct, /semantic_camera/labels_map),
then reports min/max/mean depth under EACH object's label mask.

This tells us whether the calibrated MiDaS depth is correct specifically
over the objects (not just corner/background pixels).

Usage:
    python3 check_object_depths.py world1_baseline
"""
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node as RosNode
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage

SCENE_LABELS = {
    "world1_baseline": {
        1: "obj_bottle", 2: "obj_mug", 3: "obj_bowl", 4: "obj_apple",
        5: "obj_spoon", 6: "obj_plate", 7: "obj_tissue_box",
        8: "obj_cereal_box", 9: "obj_notebook", 10: "obj_remote",
        11: "obj_toy_block", 12: "obj_banana",
    },
    "world2_dense_clutter": {
        1: "obj_bottle", 2: "obj_mug", 3: "obj_tissue_box", 4: "obj_jar",
        5: "obj_cereal_box", 6: "obj_banana", 7: "obj_bowl", 8: "obj_apple",
        9: "obj_tomato", 10: "obj_plate", 11: "obj_spoon", 12: "obj_fork",
    },
}

TIMEOUT = 20

if len(sys.argv) < 2 or sys.argv[1] not in SCENE_LABELS:
    print(f"Usage: python3 {sys.argv[0]} <scene_name>")
    print(f"Known scenes: {list(SCENE_LABELS.keys())}")
    sys.exit(1)

scene = sys.argv[1]
LABELS = SCENE_LABELS[scene]

# ── Grab one semantic segmentation frame (Gazebo direct) ──
gz_node = GzNode()
seg_received = []

def seg_cb(msg):
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        total_pixels = msg.height * msg.width
        if total_pixels == 0 or len(raw) % total_pixels != 0:
            return
        channels = len(raw) // total_pixels
        arr = raw.reshape(msg.height, msg.width, channels)
        seg_received.append(arr)
    except Exception as e:
        print(f"WARNING: failed to parse semantic frame: {e}")

gz_node.subscribe(GzImage, '/semantic_camera/labels_map', seg_cb)
print(f"Waiting for semantic frame (timeout {TIMEOUT}s)...")
start = time.time()
while not seg_received and time.time() - start < TIMEOUT:
    time.sleep(0.1)

if not seg_received:
    print("ERROR: No semantic frames received — is Gazebo running?")
    sys.exit(1)

label_frame = seg_received[-1]
label_ch = label_frame[:, :, 0]
print(f"Got semantic frame: shape={label_frame.shape}")

# ── Grab one live depth frame (ROS 2) ──
rclpy.init()
bridge = CvBridge()
depth_received = []

class DepthGrabber(RosNode):
    def __init__(self):
        super().__init__('depth_grabber_check')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            RosImage, '/depth_model/depth_image', self.cb, qos
        )

    def cb(self, msg):
        depth_received.append(bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1'))

node = DepthGrabber()
start = time.time()
while not depth_received and time.time() - start < TIMEOUT:
    rclpy.spin_once(node, timeout_sec=0.5)

if not depth_received:
    print("ERROR: No depth frames received on /depth_model/depth_image")
    sys.exit(1)

depth_frame = depth_received[-1]
print(f"Got depth frame: shape={depth_frame.shape}, dtype={depth_frame.dtype}")

if depth_frame.shape[:2] != label_ch.shape[:2]:
    print(f"WARNING: depth frame shape {depth_frame.shape[:2]} != "
          f"label frame shape {label_ch.shape[:2]} — resolutions differ, "
          f"results below may be misaligned.")

# ── Report depth stats under each object's mask ──
print(f"\n{'Object':<20} {'PixelCount':<12} {'Min(m)':<10} {'Max(m)':<10} {'Mean(m)':<10}")
print("-" * 65)
for label_id, name in LABELS.items():
    mask = (label_ch == label_id)
    count = np.count_nonzero(mask)
    if count == 0:
        print(f"{name:<20} {'0':<12} {'--':<10} {'--':<10} {'--':<10}")
        continue
    depths = depth_frame[mask]
    valid = depths[np.isfinite(depths)]
    if len(valid) == 0:
        print(f"{name:<20} {count:<12} {'no valid depth':<30}")
        continue
    print(f"{name:<20} {count:<12} {valid.min():<10.3f} {valid.max():<10.3f} {valid.mean():<10.3f}")

node.destroy_node()
rclpy.shutdown()
