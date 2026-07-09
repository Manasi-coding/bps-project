#!/usr/bin/env python3
"""
depth_publisher.py
------------------
Reads the robot's RGB camera feed from Gazebo (bridged into ROS 2), runs your
depth model, and republishes the result as a ROS 2 sensor_msgs/msg/Image
depth topic.

Supports two checkpoint families, selected via the `model_type` parameter:

    model_type=relative (default)
        Stock Depth Anything V2 relative-depth model, e.g.
        checkpoints/depth_anything_v2_vits.pth
        -> imports from models.depth_anything_v2.dpt
        -> model.infer_image(image)   (single arg)

    model_type=metric
        Metric-depth fine-tuned model, e.g.
        checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified1.pth
        -> imports from metric_depth.depth_anything_v2.dpt
        -> model.infer_image(image, input_size)   (needs max_depth in ctor)

Usage:
    python3 scripts/depth_publisher.py --ros-args \
        -p load_from:=checkpoints/depth_anything_v2_vits.pth \
        -p encoder:=vits \
        -p model_type:=relative

    python3 scripts/depth_publisher.py --ros-args \
        -p load_from:=checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified1.pth \
        -p encoder:=vits \
        -p model_type:=metric \
        -p max_depth:=20.0
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Make the project root importable, same as run_depth_anything.py / run_depth_all.py / load_model.py
sys.path.append(str(Path(__file__).resolve().parents[1]))


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter('rgb_topic', '/camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/depth_model/depth_image')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('load_from', '')
        self.declare_parameter('encoder', 'vitl')
        self.declare_parameter('model_type', 'relative')  # 'relative' or 'metric'
        self.declare_parameter('max_depth', 20.0)          # only used for model_type=metric
        self.declare_parameter('input_size', 518)

        rgb_topic     = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic   = self.get_parameter('depth_topic').get_parameter_value().string_value
        load_from     = self.get_parameter('load_from').get_parameter_value().string_value
        encoder       = self.get_parameter('encoder').get_parameter_value().string_value
        model_type    = self.get_parameter('model_type').get_parameter_value().string_value
        max_depth     = self.get_parameter('max_depth').get_parameter_value().double_value
        input_size    = self.get_parameter('input_size').get_parameter_value().integer_value

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
        if model_type not in ('relative', 'metric'):
            raise ValueError(
                f"Invalid model_type='{model_type}'. Must be 'relative' or 'metric'."
            )
        if model_type == 'metric' and max_depth <= 0:
            raise ValueError(f"Invalid max_depth={max_depth}. Must be > 0.")
        if input_size <= 0:
            raise ValueError(f"Invalid input_size={input_size}. Must be > 0.")

        self.model_type = model_type
        self.input_size = input_size

        self.bridge = CvBridge()

        # ── Detect device ──────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.get_logger().info(f"Selected device: {self.device}")
        self.get_logger().info(f"Model type: {model_type}")

        # ── Import the correct DepthAnythingV2 for this model_type ──
        if model_type == 'metric':
            # metric_depth lives outside bps-project, under Depth-Anything-V2/
            metric_repo_root = os.path.expanduser('~/Depth-Anything-V2')
            if metric_repo_root not in sys.path:
                sys.path.append(metric_repo_root)
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        else:
            from models.depth_anything_v2.dpt import DepthAnythingV2

        # ── Load model ────────────────────────────────────────────
        self.get_logger().info(f"Loading checkpoint from: {load_from}")

        state_dict = torch.load(load_from, map_location='cpu')

        # Handle 'module.' prefix from DataParallel training
        if any(k.startswith('module.') for k in state_dict.keys()):
            self.get_logger().info("Stripping 'module.' prefix from state dict...")
            state_dict = {k.replace('module.', ''): v
                          for k, v in state_dict.items()}

        if model_type == 'metric':
            self.model = DepthAnythingV2(
                **{**MODEL_CONFIGS[encoder], 'max_depth': max_depth}
            )
        else:
            self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

        if model_type == 'metric':
            self.get_logger().info(
                f"Model loaded. Encoder: {encoder}  Max depth: {max_depth:.1f}m  "
                f"Input size: {input_size}"
            )
        else:
            self.get_logger().info(f"Model loaded. Encoder: {encoder}")

        # ── Warm-up pass (absorbs slow first-inference cost at startup) ──
        self.get_logger().info("Warming up model with dummy inference pass...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self._infer(dummy_image)
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

    def _infer(self, bgr_image):
        """Run inference with the correct call signature for this model_type."""
        if self.model_type == 'metric':
            depth = self.model.infer_image(bgr_image, self.input_size)
        else:
            depth = self.model.infer_image(bgr_image)
        return depth.astype(np.float32)

    def _rgb_callback(self, msg: Image):
        print("CALLBACK FIRED", flush=True)
        """Called on every RGB frame from Gazebo (via ros_gz bridge)."""
        try:
            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().error(
                    f"Unsupported image encoding '{msg.encoding}' — skipping frame"
                )
                return

            # ROS Image → OpenCV BGR
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            print("STEP: cv2 conversion done", flush=True)
            print("STEP: converted to cv2", flush=True)

            # Run inference
            print("STEP: starting inference", flush=True)
            depth = self._infer(bgr)
            print("STEP: inference done", flush=True)

            # Sanity check
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