#!/usr/bin/env python3
"""
camera_info_relay_gz.py
------------------------
Workaround for the broken ros_gz_bridge on this machine: subscribes to
Gazebo's /camera_info topic DIRECTLY via gz.transport13, converts each
message to a ROS 2 sensor_msgs/msg/CameraInfo, and republishes it on
/camera_info via plain rclpy.

depth_to_pointcloud.py only needs camera_info ONCE (it caches self.K on
first valid message and ignores repeats unless intrinsics change), so this
relay does not need to be tightly synced to depth frames — it just needs
to get a handful of valid CameraInfo messages onto the ROS side before
depth_to_pointcloud's CAMERA_INFO_TIMEOUT_SEC (10s) watchdog fires.

Run this alongside depth_publisher_gz.py and depth_to_pointcloud.py —
no ros_gz_bridge needed at all.

Usage:
    python3 scripts/camera_info_relay_gz.py --ros-args \
        -p gz_camera_info_topic:=/camera_info \
        -p ros_camera_info_topic:=/camera_info \
        -p frame_id:=rgb_camera
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CameraInfo

from gz.transport13 import Node as GzNode
from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))


class CameraInfoRelayGz(Node):
    def __init__(self):
        super().__init__('camera_info_relay_gz')

        self.declare_parameter('gz_camera_info_topic', '/camera_info')
        self.declare_parameter('ros_camera_info_topic', '/camera_info')
        self.declare_parameter('frame_id', 'rgb_camera')

        gz_topic  = self.get_parameter('gz_camera_info_topic').get_parameter_value().string_value
        ros_topic = self.get_parameter('ros_camera_info_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.info_pub = self.create_publisher(CameraInfo, ros_topic, camera_qos)

        self.gz_node = GzNode()
        ok = self.gz_node.subscribe(GzCameraInfo, gz_topic, self._gz_camera_info_callback)
        if not ok:
            raise RuntimeError(
                f"Failed to subscribe to Gazebo topic '{gz_topic}' via gz.transport13. "
                "Confirm Gazebo is running and `gz topic -l` lists this topic."
            )

        self._relayed_count = 0
        self.get_logger().info(f"Subscribed to Gazebo camera_info (direct gz.transport13): {gz_topic}")
        self.get_logger().info(f"Publishing camera_info on (ROS 2): {ros_topic}")
        self.get_logger().info("camera_info_relay_gz ready.")

    def _gz_camera_info_callback(self, msg: GzCameraInfo):
        print("GZ CAMERA_INFO CALLBACK FIRED", flush=True)
        try:
            info = CameraInfo()
            info.header.stamp = self.get_clock().now().to_msg()
            info.header.frame_id = self.frame_id
            info.height = msg.height
            info.width = msg.width
            info.distortion_model = "plumb_bob"

            # gz.msgs.CameraInfo carries intrinsics/projection/distortion
            # under msg.intrinsics / msg.projection / msg.distortion.
            # Guard each with getattr-style checks so a field-layout
            # difference doesn't crash the relay outright.
            if len(msg.intrinsics.k) == 9:
                info.k = list(msg.intrinsics.k)
            else:
                self.get_logger().warning(
                    f"Unexpected intrinsics.k length {len(msg.intrinsics.k)} "
                    "(expected 9) — skipping this message."
                )
                return

            if len(msg.projection.p) == 12:
                info.p = list(msg.projection.p)
            else:
                # Not fatal — depth_to_pointcloud.py only reads msg.k.
                self.get_logger().warning(
                    f"Unexpected projection.p length {len(msg.projection.p)} "
                    "(expected 12) — leaving P unset."
                )

            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

            if msg.distortion.k:
                info.d = list(msg.distortion.k)

            self.info_pub.publish(info)
            self._relayed_count += 1
            print(f"STEP: camera_info published! (count={self._relayed_count})", flush=True)

        except Exception as e:
            import traceback
            print("STEP: EXCEPTION CAUGHT:", repr(e), flush=True)
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoRelayGz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
