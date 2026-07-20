#!/usr/bin/env python3
"""
depth_publisher_gz.py
----------------------
Workaround for a broken ros_gz_bridge on this machine (bridge creates topics
but fails to decode gz.msgs.Image -> sensor_msgs/msg/Image, logging
"Unknown message type [N]" and never actually publishing any data).

This version subscribes DIRECTLY to Gazebo Transport (gz.transport13) for the
RGB camera feed, bypassing ros_gz_bridge entirely for input. It still uses
plain rclpy to PUBLISH the resulting depth image, so downstream ROS 2 nodes
(log_results.py, depth_to_pointcloud.py, etc.) see /depth_model/depth_image
exactly as before -- only the input path changed.

Supports the same three checkpoint families as depth_publisher.py, selected
via the `model_type` parameter: 'relative', 'metric', 'midas'.

Usage (same flags as depth_publisher.py, still via --ros-args since we still
use rclpy for the depth publisher / parameters):

    python3 scripts/depth_publisher_gz.py --ros-args \
        -p load_from:=checkpoints/depth_anything_v2_vits.pth \
        -p encoder:=vits \
        -p model_type:=relative \
        -p gz_rgb_topic:=/rgb_camera \
        -p depth_topic:=/depth_model/depth_image

    python3 scripts/depth_publisher_gz.py --ros-args \
        -p load_from:=checkpoints/midas_v21_small_256.pt \
        -p model_type:=midas \
        -p gz_rgb_topic:=/rgb_camera \
        -p depth_topic:=/depth_model/depth_image

Requires: python3-gz-transport13 (already installed on this machine per
`dpkg -l | grep gz-transport13`).
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

from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage

# Make the project root importable, same as depth_publisher.py
sys.path.append(str(Path(__file__).resolve().parents[1]))


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

# Map gz.msgs.Image PixelFormatType values we expect from the camera sensor.
# RGB_INT8 = 3 in gz.msgs.PixelFormatType (matches "pixel_format_type: RGB_INT8"
# seen in `gz topic -e -t /rgb_camera` output on this machine).
GZ_PIXEL_FORMAT_RGB_INT8 = 3


class DepthPublisherGz(Node):
    def __init__(self):
        super().__init__('depth_publisher_gz')

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter('gz_rgb_topic', '/rgb_camera')
        self.declare_parameter('depth_topic', '/depth_model/depth_image')
        self.declare_parameter('frame_id', 'rgb_camera')
        self.declare_parameter('load_from', '')
        self.declare_parameter('encoder', 'vitl')
        self.declare_parameter('model_type', 'relative')  # 'relative', 'metric', or 'midas'
        self.declare_parameter('max_depth', 20.0)          # only used for model_type=metric
        self.declare_parameter('input_size', 518)
        self.declare_parameter('midas_calib_encoder', 'vits')
        self.declare_parameter('midas_calib_load_from', '')  # metric checkpoint used to calibrate MiDaS scale/shift
        self.declare_parameter('midas_calib_points', 100)     # number of sampled pixels for least-squares fit

        gz_rgb_topic  = self.get_parameter('gz_rgb_topic').get_parameter_value().string_value
        depth_topic   = self.get_parameter('depth_topic').get_parameter_value().string_value
        frame_id      = self.get_parameter('frame_id').get_parameter_value().string_value
        load_from     = self.get_parameter('load_from').get_parameter_value().string_value
        encoder       = self.get_parameter('encoder').get_parameter_value().string_value
        model_type    = self.get_parameter('model_type').get_parameter_value().string_value
        max_depth     = self.get_parameter('max_depth').get_parameter_value().double_value
        input_size    = self.get_parameter('input_size').get_parameter_value().integer_value
        midas_calib_encoder    = self.get_parameter('midas_calib_encoder').get_parameter_value().string_value
        midas_calib_load_from  = self.get_parameter('midas_calib_load_from').get_parameter_value().string_value
        midas_calib_points     = self.get_parameter('midas_calib_points').get_parameter_value().integer_value

        if model_type == 'midas' and not midas_calib_load_from:
            raise ValueError(
                "model_type='midas' requires 'midas_calib_load_from' — path to a "
                "metric Depth Anything checkpoint used to calibrate MiDaS's "
                "relative depth onto a real metric scale."
            )
        self.midas_calib_points = midas_calib_points

        # ── Parameter validation ─────────────────────────────────────
        if not load_from:
            raise FileNotFoundError(
                "Parameter 'load_from' is required (path to weights file)."
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
        if model_type not in ('relative', 'metric', 'midas'):
            raise ValueError(
                f"Invalid model_type='{model_type}'. Must be 'relative', 'metric', or 'midas'."
            )
        if model_type == 'metric' and max_depth <= 0:
            raise ValueError(f"Invalid max_depth={max_depth}. Must be > 0.")
        if input_size <= 0:
            raise ValueError(f"Invalid input_size={input_size}. Must be > 0.")

        self.model_type = model_type
        self.input_size = input_size
        self.frame_id = frame_id
        self.depth_topic = depth_topic

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

        # ── Import the correct model class for this model_type ──
        if model_type == 'metric':
            metric_repo_root = os.path.expanduser('~/Depth-Anything-V2')
            if metric_repo_root not in sys.path:
                sys.path.append(metric_repo_root)
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        elif model_type == 'relative':
            from models.depth_anything_v2.dpt import DepthAnythingV2
        # midas: no import needed here, handled via torch.hub below

        # ── Load model ────────────────────────────────────────────
        self.get_logger().info(f"Loading checkpoint from: {load_from}")

        state_dict = torch.load(load_from, map_location='cpu')

        if any(k.startswith('module.') for k in state_dict.keys()):
            self.get_logger().info("Stripping 'module.' prefix from state dict...")
            state_dict = {k.replace('module.', ''): v
                          for k, v in state_dict.items()}

        if model_type == 'metric':
            self.model = DepthAnythingV2(
                **{**MODEL_CONFIGS[encoder], 'max_depth': max_depth}
            )
            self.model.load_state_dict(state_dict, strict=False)
        elif model_type == 'relative':
            self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
            self.model.load_state_dict(state_dict, strict=False)
        else:  # midas
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.model.load_state_dict(state_dict, strict=False)
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.midas_transform = midas_transforms.small_transform

            # ── Load metric Depth Anything as calibration reference ──
            # MiDaS outputs unitless relative depth. We calibrate it to real
            # metres per-frame by fitting scale/shift against this metric
            # model's output on the same frame (metric_depth ≈ scale *
            # midas_depth + shift, via least squares).
            self.get_logger().info(
                f"Loading MiDaS calibration reference (metric) from: {midas_calib_load_from}"
            )
            metric_repo_root = os.path.expanduser('~/Depth-Anything-V2')
            if metric_repo_root not in sys.path:
                sys.path.append(metric_repo_root)
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2

            calib_state_dict = torch.load(midas_calib_load_from, map_location='cpu')
            if any(k.startswith('module.') for k in calib_state_dict.keys()):
                calib_state_dict = {k.replace('module.', ''): v
                                     for k, v in calib_state_dict.items()}

            self.midas_calib_model = MetricDepthAnythingV2(
                **{**MODEL_CONFIGS[midas_calib_encoder], 'max_depth': max_depth}
            )
            self.midas_calib_model.load_state_dict(calib_state_dict, strict=False)
            self.midas_calib_model = self.midas_calib_model.to(self.device).eval()
            self.get_logger().info("MiDaS calibration reference model loaded.")

        self.model = self.model.to(self.device).eval()

        if model_type == 'metric':
            self.get_logger().info(
                f"Model loaded. Encoder: {encoder}  Max depth: {max_depth:.1f}m  "
                f"Input size: {input_size}"
            )
        else:
            self.get_logger().info(f"Model loaded. Encoder: {encoder}")

        # ── Warm-up pass ──
        self.get_logger().info("Warming up model with dummy inference pass...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self._infer(dummy_image)
        self.get_logger().info("Model warm-up complete.")

        # ── ROS 2 publisher for depth output (unchanged from depth_publisher.py) ──
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.depth_pub = self.create_publisher(Image, depth_topic, camera_qos)

        # ── Gazebo Transport subscriber for RGB input (bypasses ros_gz_bridge) ──
        self.gz_node = GzNode()
        ok = self.gz_node.subscribe(GzImage, gz_rgb_topic, self._gz_rgb_callback)
        if not ok:
            raise RuntimeError(
                f"Failed to subscribe to Gazebo topic '{gz_rgb_topic}' via gz.transport13. "
                "Confirm Gazebo is running and `gz topic -l` lists this topic."
            )

        self.get_logger().info(f"Subscribed to Gazebo RGB topic (direct gz.transport13): {gz_rgb_topic}")
        self.get_logger().info(f"Publishing depth on (ROS 2): {depth_topic}")
        self.get_logger().info("depth_publisher_gz ready.")

    def _infer(self, bgr_image):
        """Run inference with the correct call signature for this model_type."""
        if self.model_type == 'midas':
            img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img).to(self.device)
            with torch.no_grad():
                pred = self.model(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=bgr_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = pred.cpu().numpy()

            # ── Calibrate MiDaS's relative depth onto real metres ──
            # MiDaS is unitless; fit metric_depth ≈ scale*midas + shift
            # against the metric reference model's output on this same
            # frame, using a random sample of pixels (least squares).
            with torch.no_grad():
                metric_ref = self.midas_calib_model.infer_image(bgr_image, self.input_size)
            metric_ref = metric_ref.astype(np.float32)

            # --- DEBUG: check shape/alignment ---
            print(f"[ALIGN CHECK] bgr_image.shape={bgr_image.shape}  "
                  f"depth.shape={depth.shape}  metric_ref.shape={metric_ref.shape}", flush=True)
            import os
            if not os.path.exists("debug/align_check_done.flag"):
                os.makedirs("debug", exist_ok=True)
                def to_heat(a):
                    a = a.copy()
                    a[~np.isfinite(a)] = 0
                    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
                    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
                    return (a * 255).astype(np.uint8)
                cv2.imwrite("debug/rgb_input.png", bgr_image)
                cv2.imwrite("debug/midas_raw_heat.png", cv2.applyColorMap(to_heat(depth), cv2.COLORMAP_JET))
                cv2.imwrite("debug/metric_ref_heat.png", cv2.applyColorMap(to_heat(metric_ref), cv2.COLORMAP_JET))
                with open("debug/align_check_done.flag", "w") as f:
                    f.write("done")
                print("[DEBUG] Saved rgb_input.png, midas_raw_heat.png, metric_ref_heat.png to debug/", flush=True)
            # --- end debug ---

            valid_mask = np.isfinite(depth) & np.isfinite(metric_ref) & (metric_ref > 0)
            ys, xs = np.where(valid_mask)
            n_available = len(ys)

            if n_available >= 10:
                n_sample = min(self.midas_calib_points, n_available)
                idx = np.random.choice(n_available, size=n_sample, replace=False)
                midas_vals = depth[ys[idx], xs[idx]]
                metric_vals = metric_ref[ys[idx], xs[idx]]

                scale, shift = np.polyfit(midas_vals, metric_vals, 1)

                # --- DEBUG: dump fit inputs/outputs once for world1_baseline ---
                import os
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, "calib_world1_baseline.npz")
                if not os.path.exists(debug_path):
                    np.savez(debug_path,
                            midas_vals=midas_vals, metric_vals=metric_vals,
                            scale=scale, shift=shift,
                            sample_ys=ys[idx], sample_xs=xs[idx],
                            metric_ref_full=metric_ref, image_shape=np.array(bgr_image.shape))
                    self.get_logger().info(
                        f"[DEBUG] Saved calib dump: scale={scale:.4f} shift={shift:.4f} "
                        f"metric_ref range=[{metric_ref.min():.2f},{metric_ref.max():.2f}]"
                    )
                # --- end debug ---

                depth = scale * depth + shift
                depth = np.clip(depth, 0.05, 20.0)  # guard against fit outliers
            else:
                self.get_logger().warning(
                    "Not enough valid pixels to calibrate MiDaS this frame "
                    f"({n_available} available) — falling back to metric "
                    "reference depth directly."
                )
                depth = metric_ref
        elif self.model_type == 'metric':
            depth = self.model.infer_image(bgr_image, self.input_size)
        else:
            depth = self.model.infer_image(bgr_image)
        return depth.astype(np.float32)

    def _gz_image_to_bgr(self, msg: GzImage):
        """Convert a gz.msgs.Image protobuf message to an OpenCV BGR array."""
        h, w = msg.height, msg.width
        pf = msg.pixel_format_type

        buf = np.frombuffer(msg.data, dtype=np.uint8)

        if pf == GZ_PIXEL_FORMAT_RGB_INT8:
            expected = h * w * 3
            if buf.size != expected:
                raise ValueError(
                    f"Unexpected buffer size {buf.size}, expected {expected} "
                    f"for {w}x{h} RGB_INT8"
                )
            rgb = buf.reshape((h, w, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        else:
            raise ValueError(
                f"Unsupported gz pixel_format_type={pf} — only RGB_INT8 (3) is "
                "currently handled. Check `gz topic -e -t <topic>` for the "
                "actual pixel_format_type and extend _gz_image_to_bgr if needed."
            )

    def _gz_rgb_callback(self, msg: GzImage):
        """Called directly by gz.transport13 on every RGB frame from Gazebo."""
        print("GZ CALLBACK FIRED", flush=True)
        try:
            bgr = self._gz_image_to_bgr(msg)
            print("STEP: gz image converted to cv2", flush=True)

            print("STEP: starting inference", flush=True)
            depth = self._infer(bgr)
            print("STEP: inference done", flush=True)

            if np.isnan(depth).any() or np.isinf(depth).any():
                self.get_logger().warning("Depth contains NaN/Inf — skipping frame")
                return

            depth_msg = self.bridge.cv2_to_imgmsg(
                depth.astype(np.float32), encoding="32FC1"
            )
            # gz.msgs.Image headers use a different time representation than
            # ROS 2; stamp with the node's current ROS clock instead of trying
            # to translate msg.header, and set frame_id from the parameter.
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = self.frame_id

            self.depth_pub.publish(depth_msg)
            print("STEP: published!", flush=True)

        except Exception as e:
            import traceback
            print("STEP: EXCEPTION CAUGHT:", repr(e), flush=True)
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = DepthPublisherGz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
