#!/usr/bin/env python3
"""
depth_publisher.py
------------------
Reads the robot's RGB camera feed from Gazebo (bridged into ROS 2), runs your
depth model using the exact same inference pipeline as your run.py script,
and republishes the result as a ROS 2 sensor_msgs/msg/Image depth topic.

Plug-and-play with your existing weights — set the `load_from` and `encoder`
parameters (CLI or YAML).

Usage:
    ros2 run <your_package> depth_publisher.py --ros-args \
        -p load_from:=checkpoints/your_finetuned_weights.pth \
        -p encoder:=vitl \
        -p max_depth:=20.0
"""

import os

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── Your model — same import as run.py ───────────────────────────────────
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

# MOD: replaced hand-rolled preprocessing (square resize, no ImageNet
# normalization, NEAREST interpolation) with the model's own infer_image()
# method — same call run.py uses. The hand-rolled version diverged from
# dpt.py's actual preprocessing pipeline (missing normalization, wrong
# resize/interpolation, no PrepareForNet formatting), causing a systematic
# ~6x depth overestimate (observed mean=5.03m vs true ~0.75m camera-to-
# object distance in world1_primitives).
def infer_image_metric(model, raw_image, input_size, device):
    depth = model.infer_image(raw_image, input_size)
    return depth.astype(np.float32)

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter('rgb_topic', '/camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/depth_model/depth_image')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('load_from', '')
        self.declare_parameter('encoder', 'vitl')
        self.declare_parameter('max_depth', 20.0)
        self.declare_parameter('input_size', 518)

        rgb_topic         = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic       = self.get_parameter('depth_topic').get_parameter_value().string_value
        load_from         = self.get_parameter('load_from').get_parameter_value().string_value
        encoder            = self.get_parameter('encoder').get_parameter_value().string_value
        max_depth          = self.get_parameter('max_depth').get_parameter_value().double_value
        input_size         = self.get_parameter('input_size').get_parameter_value().integer_value

        # ── Parameter validation ─────────────────────────────────────
        if not load_from:
            raise FileNotFoundError(
                "Parameter 'load_from' is required (path to .pth weights file)."
            )
        if not os.path.isfile(load_from):
            raise FileNotFoundError(
                f"Weight file not found at load_from='{load_from}'. "
                "Check the path and try again."
            )
        if encoder not in MODEL_CONFIGS:
            raise ValueError(
                f"Invalid encoder='{encoder}'. Must be one of {list(MODEL_CONFIGS.keys())}."
            )
        if max_depth <= 0:
            raise ValueError(f"Invalid max_depth={max_depth}. Must be > 0.")
        if input_size <= 0:
            raise ValueError(f"Invalid input_size={input_size}. Must be > 0.")

        self.input_size = input_size

        self.bridge = CvBridge()

        # ── Detect device — same logic as your run.py ────────────────────
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.get_logger().info(f"Selected device: {self.device}")

        # ── Load model — same logic as your run.py ───────────────────────
        self.get_logger().info(f"Loading checkpoint from: {load_from}")

        state_dict = torch.load(load_from, map_location='cpu')

        # Handle 'module.' prefix from DataParallel training — same as run.py
        if any(k.startswith('module.') for k in state_dict.keys()):
            self.get_logger().info("Stripping 'module.' prefix from state dict...")
            state_dict = {k.replace('module.', ''): v
                          for k, v in state_dict.items()}

        self.model = DepthAnythingV2(
            **{**MODEL_CONFIGS[encoder], 'max_depth': max_depth}
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()
        self.get_logger().info(
            f"Model loaded. Encoder: {encoder}  Max depth: {max_depth:.1f}m  "
            f"Input size: {input_size}"
        )

        self.model = self.model.to(self.device).eval()
        self.get_logger().info(
            f"Model loaded. Encoder: {encoder}  Max depth: {max_depth:.1f}m  "
            f"Input size: {input_size}"
        )

        # MOD: the first CPU inference call through a freshly-loaded model
        # is dramatically slower than subsequent calls (kernel selection,
        # memory allocation, no warm operator cache) — observed ~25-28s for
        # a first frame vs ~3-5s steady-state. Run one dummy inference pass
        # here, before subscribing to RGB or declaring ready, so this cost
        # is absorbed at startup instead of silently eating into the first
        # object's pointcloud-wait timeout during an actual experiment run.
        self.get_logger().info("Warming up model with dummy inference pass...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = infer_image_metric(self.model, dummy_image, input_size, self.device)
        self.get_logger().info("Model warm-up complete.")
        
        # ── ROS 2 QoS suitable for camera topics ──────────────────────
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── ROS 2 publishers / subscribers ────────────────────────────
        self.depth_pub = self.create_publisher(Image, depth_topic, camera_qos)

        self.create_subscription(
            Image, rgb_topic, self._rgb_callback, camera_qos
        )

        self.get_logger().info(f"Subscribed to RGB topic: {rgb_topic}")
        self.get_logger().info(f"Publishing depth on: {depth_topic}")
        self.get_logger().info("depth_publisher ready.")

    def _rgb_callback(self, msg: Image):
        print("CALLBACK FIRED", flush=True)
        """Called on every RGB frame from Gazebo (via ros_gz bridge)."""
        try:
            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().error(
                    f"Unsupported image encoding '{msg.encoding}' — skipping frame"
                )
                return

            # ROS Image → OpenCV BGR  (same format your run.py uses)
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            print("STEP: cv2 conversion done", flush=True)
            print("STEP: converted to cv2", flush=True)

            # Run inference — identical to your run.py
            print("STEP: starting inference", flush=True)
            depth = infer_image_metric(
                self.model, bgr, self.input_size, self.device
            )
            print("STEP: inference done", flush=True)

            # Sanity check (mirrors your run.py print statements)
            if np.isnan(depth).any() or np.isinf(depth).any():
                self.get_logger().warning(
                    "Depth contains NaN/Inf — skipping frame"
                )
                return

            # Publish as ROS 32FC1 depth image (float32, metres)
            depth_msg = self.bridge.cv2_to_imgmsg(
                depth.astype(np.float32), encoding="32FC1"
            )
            depth_msg.header = msg.header   # preserve timestamp + frame_id exactly
            self.depth_pub.publish(depth_msg)
            print("STEP: published!", flush=True)

        except Exception as e:
            import traceback
            print("STEP: EXCEPTION CAUGHT:", repr(e), flush=True)
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = DepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()