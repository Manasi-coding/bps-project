#!/usr/bin/env python3
"""
grasp_demo.py (ROS 2 Humble — audited)
----------------------------------------
Demonstration script for the paper. Uses the proposed depth model pipeline
to scan the scene, identify the first graspable object, and execute a
top-down grasp with the OpenMANIPULATOR-X arm.

Assumes:
  - Standard Gazebo table height: 0.75m
  - Robot spawned ~0.6m from table edge (TurtleBot3 Waffle)
  - depth_publisher.py and depth_to_pointcloud.py are already running
  - No MoveIt — uses open-loop joint position control

Usage:
    ros2 run <your_package> grasp_demo
    # or directly:
    python3 grasp_demo.py

The robot will:
  1. Scan the pointcloud for any graspable object (width <= 77mm, height <= 120mm)
  2. Drive to standoff distance from the first graspable object found
  3. Execute a top-down grasp: open -> lower -> approach -> close -> lift -> retreat
  4. Return to home pose

RUNTIME VERIFICATION ITEMS (cannot be guaranteed from source alone — confirm
against your installed packages before trusting this in the loop):
  1. QoS of the /depth_model/pointcloud publisher (depth_to_pointcloud.py).
     This script assumes RELIABLE/VOLATILE (rclpy's create_publisher default)
     because that node is custom code, not a standard camera driver. If you
     ever changed that publisher's QoS, update CLOUD_QOS below to match, or
     the subscription will silently receive nothing.
  2. open_manipulator_msgs/srv/SetJointPosition field names/types. Run:
         ros2 interface show open_manipulator_msgs/srv/SetJointPosition
         ros2 interface show open_manipulator_msgs/msg/JointPosition
     and confirm `joint_position.joint_name`, `joint_position.position`, and
     `path_time` exist with these exact names in your installed version.
  3. Exact PointCloud2 field layout (must contain float32 'x','y','z').
"""

import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.task import Future

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs_py.point_cloud2 as pc2

from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.msg import JointPosition


# -- Scene / robot constants -------------------------------------------------
TABLE_HEIGHT       = 0.75    # metres - standard Gazebo cafe table
CAMERA_HEIGHT      = 0.18    # metres - TurtleBot3 Waffle camera above base

# -- Gripper physical limits --------------------------------------------------
GRIPPER_MAX_WIDTH  = 0.077   # 77 mm
GRIPPER_MAX_HEIGHT = 0.120   # 120 mm

# -- Region of interest --------------------------------------------------------
# Tuned for objects sitting on a 0.75m table ~0.6m in front of the robot.
# Z is forward (camera frame), Y is up, X is lateral.
ROI_X_RANGE = (-0.3,  0.3)   # lateral: 30cm either side of centre
ROI_Y_RANGE = (-0.60, -0.20) # vertical: above table surface, below camera
ROI_Z_RANGE = ( 0.20,  1.00) # forward: 20cm to 100cm from camera

MIN_POINTS_FOR_OBJECT = 40
REQUIRED_CLOUD_FIELDS = ("x", "y", "z")

# -- Motion parameters ----------------------------------------------------------
STANDOFF_DISTANCE  = 0.22    # metres - stop this far from object face
APPROACH_DISTANCE  = 0.05    # metres - final creep
DRIVE_SPEED        = 0.07    # m/s
DRIVE_RATE_HZ      = 20
ARM_MOVE_TIME      = 2.5     # seconds per joint move

# -- OpenMANIPULATOR-X joint poses (radians) -------------------------------------
# Calibrated for TurtleBot3 Waffle + table at 0.75m.
# joint2 negative = arm forward and down; joint4 controls wrist angle.

ARM_HOME = {
    # Safe travel pose - arm folded up, away from camera FOV
    "joint1":  0.000,
    "joint2": -1.050,
    "joint3":  0.350,
    "joint4":  0.700,
}

ARM_PREGRASP = {
    # Arm extended forward, gripper high above table (~30cm above surface)
    "joint1":  0.000,
    "joint2": -0.500,
    "joint3":  0.600,
    "joint4":  0.900,
}

ARM_GRASP_TABLE = {
    # Gripper lowered to ~5cm above table surface for top-down grasp
    # Tuned for 0.75m table with robot camera at 0.18m + arm reach
    "joint1":  0.000,
    "joint2":  0.200,
    "joint3": -0.350,
    "joint4":  0.150,
}

ARM_LIFT = {
    # Lift pose - same as pregrasp but used after closing gripper
    "joint1":  0.000,
    "joint2": -0.500,
    "joint3":  0.600,
    "joint4":  0.900,
}

GRIPPER_OPEN  =  0.019   # metres per finger
GRIPPER_CLOSE = -0.010   # metres per finger (gripping)

# -- Topics / services ------------------------------------------------------------
CLOUD_TOPIC     = "/depth_model/pointcloud"
CMD_VEL_TOPIC   = "/cmd_vel"

# RUNTIME VERIFICATION ITEM #1 (see module docstring): this assumes the
# publisher uses rclpy's default QoS (RELIABLE, VOLATILE, KEEP_LAST/10),
# which is what create_publisher() gives you unless a node explicitly
# overrides it. Custom pipeline nodes (unlike standard camera drivers,
# which typically publish BEST_EFFORT sensor data) usually keep the
# default unless someone deliberately changed it — so RELIABLE is the
# correct default assumption here, not BEST_EFFORT. If depth_to_pointcloud.py
# was written with qos_profile_sensor_data, change this to BEST_EFFORT
# or the subscription will never match the publisher and no data will
# ever arrive (a silent failure — no error, just an empty topic).
CLOUD_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

SERVICE_WAIT_TIMEOUT_SEC = 15.0
SERVICE_CALL_TIMEOUT_SEC = 10.0
CLOUD_WAIT_TIMEOUT_SEC   = 15.0


# =============================================================================
class GraspDemo(Node):
    def __init__(self):
        super().__init__("grasp_demo")

        self.latest_cloud = None
        self._cloud_lock = threading.Lock()

        # -- Publishers --------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 1)

        # -- Arm service clients -------------------------------------------------
        self.get_logger().info("[Demo] Connecting to arm services...")
        self._set_joint_client = self.create_client(
            SetJointPosition, "/goal_joint_space_path")
        self._set_gripper_client = self.create_client(
            SetJointPosition, "/goal_tool_control")

        if not self._set_joint_client.wait_for_service(
                timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            raise RuntimeError(
                "Timed out waiting for service '/goal_joint_space_path'. "
                "Is the OpenMANIPULATOR-X controller node running?")
        if not self._set_gripper_client.wait_for_service(
                timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            raise RuntimeError(
                "Timed out waiting for service '/goal_tool_control'. "
                "Is the OpenMANIPULATOR-X controller node running?")
        self.get_logger().info("[Demo] Arm services connected.")

        # -- Pointcloud subscriber --------------------------------------------
        self.create_subscription(
            PointCloud2, CLOUD_TOPIC, self._cloud_cb, CLOUD_QOS)

    # =========================================================================
    # Top-level demo entry point
    # =========================================================================

    def run(self):
        """
        Main demo loop:
          1. Wait for the first pointcloud (deferred from __init__ so the
             constructor never blocks — see class docstring / change log)
          2. Move arm to home so it doesn't block the camera
          3. Scan for graspable objects
          4. Grasp the first one found
          5. Return home
        """
        self.get_logger().info("[Demo] Waiting for first pointcloud...")
        self._wait_for_cloud(timeout=CLOUD_WAIT_TIMEOUT_SEC)

        self.get_logger().info("[Demo] -- Starting grasp demonstration --")

        # Step 1: home pose so arm is out of camera FOV
        self.get_logger().info("[Demo] Moving arm to home pose...")
        self._move_arm(ARM_HOME)
        self._move_gripper(GRIPPER_OPEN)

        # Step 2: scan scene
        self.get_logger().info("[Demo] Scanning scene for graspable objects...")
        target = self._find_first_graspable()

        if target is None:
            self.get_logger().warn(
                "[Demo] No graspable object found in scene. Exiting.")
            return

        self.get_logger().info(
            "[Demo] Graspable object found! "
            f"Estimated W={target['width']:.3f}m  H={target['height']:.3f}m  "
            f"Distance={target['distance']:.3f}m"
        )

        # Step 3: execute grasp
        success = self._execute_grasp(target)

        if success:
            self.get_logger().info("[Demo] Grasp demonstration complete.")
        else:
            self.get_logger().error("[Demo] Grasp failed - check arm calibration.")

    # =========================================================================
    # Scene scanning
    # =========================================================================

    def _find_first_graspable(self, n_scans: int = 5) -> dict:
        """
        Take n_scans pointcloud snapshots and return the first cluster
        whose estimated dimensions fit within the gripper limits.
        Returns a dict with width, height, distance, centroid - or None.
        """
        for attempt in range(n_scans):
            time.sleep(0.3)   # let a fresh cloud arrive; background executor
                              # thread keeps servicing the subscription while
                              # this thread sleeps (see class docstring)

            with self._cloud_lock:
                cloud = self.latest_cloud

            if cloud is None:
                continue

            try:
                points = self._unpack_cloud(cloud)
            except ValueError as exc:
                self.get_logger().error(
                    f"[Demo] Malformed PointCloud2 on scan "
                    f"{attempt + 1}/{n_scans}: {exc}")
                continue

            roi = self._filter_roi(points)

            if len(roi) < MIN_POINTS_FOR_OBJECT:
                self.get_logger().info(
                    f"[Demo] Scan {attempt + 1}/{n_scans}: "
                    "not enough points in ROI")
                continue

            # Try every cluster in the ROI, not just the largest one,
            # so we find the first graspable even if a large object
            # is closer to the camera.
            clusters = self._extract_clusters(roi)
            self.get_logger().info(
                f"[Demo] Scan {attempt + 1}/{n_scans}: "
                f"found {len(clusters)} clusters")

            for i, cluster in enumerate(clusters):
                w, h, d = self._estimate_dimensions(cluster)
                centroid = cluster.mean(axis=0)
                self.get_logger().info(
                    f"  Cluster {i}: W={w:.3f} H={h:.3f} D={d:.3f} "
                    f"dist={float(centroid[2]):.3f}")
                if w <= GRIPPER_MAX_WIDTH and h <= GRIPPER_MAX_HEIGHT:
                    return {
                        "width":    w,
                        "height":   h,
                        "depth":    d,
                        "distance": float(centroid[2]),  # Z = forward
                        "centroid": centroid,
                    }

        return None   # nothing graspable found

    # =========================================================================
    # Grasp execution sequence
    # =========================================================================

    def _execute_grasp(self, target: dict) -> bool:
        """
        9-step top-down grasp sequence for a tabletop object.

          1.  Pre-grasp pose  - arm forward and high
          2.  Open gripper
          3.  Drive to standoff distance
          4.  Lower arm to grasp height above table
          5.  Creep forward (approach)
          6.  Close gripper
          7.  Lift arm (raise object off table)
          8.  Reverse to start position
          9.  Return arm to home
        """
        total_driven = 0.0
        try:
            # 1. Pre-grasp
            self.get_logger().info("[Grasp] 1/9 Pre-grasp pose")
            self._move_arm(ARM_PREGRASP)

            # 2. Open gripper
            self.get_logger().info("[Grasp] 2/9 Opening gripper")
            self._move_gripper(GRIPPER_OPEN)

            # 3. Drive forward to standoff
            drive_dist = max(0.0, target["distance"] - STANDOFF_DISTANCE)
            self.get_logger().info(
                f"[Grasp] 3/9 Driving {drive_dist:.3f}m to standoff")
            self._drive(drive_dist)
            total_driven += drive_dist

            # 4. Lower arm - interpolate joint angles based on object height
            self.get_logger().info(
                f"[Grasp] 4/9 Lowering arm (obj height {target['height']:.3f}m)")
            grasp_joints = self._joints_for_object_height(target["height"])
            self._move_arm(grasp_joints)

            # 5. Creep forward into object
            self.get_logger().info(
                f"[Grasp] 5/9 Approach creep ({APPROACH_DISTANCE:.3f}m)")
            self._drive(APPROACH_DISTANCE)
            total_driven += APPROACH_DISTANCE

            # 6. Close gripper
            self.get_logger().info("[Grasp] 6/9 Closing gripper")
            self._move_gripper(GRIPPER_CLOSE)
            time.sleep(0.5)   # brief pause to let gripper settle

            # 7. Lift
            self.get_logger().info("[Grasp] 7/9 Lifting object")
            self._move_arm(ARM_LIFT)

            # 8. Reverse
            self.get_logger().info(f"[Grasp] 8/9 Reversing {total_driven:.3f}m")
            self._drive(-total_driven)

            # 9. Home
            self.get_logger().info("[Grasp] 9/9 Returning to home pose")
            self._move_arm(ARM_HOME)
            self._move_gripper(GRIPPER_OPEN)   # release object

            return True

        except (RuntimeError, TimeoutError) as exc:
            self.get_logger().error(f"[Grasp] Execution error: {exc}")
            try:
                self._stop()
                self._move_arm(ARM_HOME)
            except (RuntimeError, TimeoutError) as recovery_exc:
                self.get_logger().error(
                    f"[Grasp] Recovery-to-home also failed: {recovery_exc}")
            return False

    # =========================================================================
    # Joint angle interpolation for object height
    # =========================================================================

    def _joints_for_object_height(self, object_height: float) -> dict:
        """
        Linearly interpolate joint angles between ARM_PREGRASP (high)
        and ARM_GRASP_TABLE (low) based on object height.

        Taller objects (approaching GRIPPER_MAX_HEIGHT) keep the arm
        higher; shorter objects (near 0) bring the arm fully down to
        the table surface pose.
        """
        t = float(np.clip(object_height / GRIPPER_MAX_HEIGHT, 0.0, 1.0))
        joints = {}
        for name in ARM_PREGRASP:
            high = ARM_PREGRASP[name]
            low  = ARM_GRASP_TABLE[name]
            joints[name] = high * t + low * (1.0 - t)
        return joints

    # =========================================================================
    # Motion primitives
    # =========================================================================

    def _drive(self, distance: float):
        """Drive straight forward (+) or backward (-) by distance metres.

        Runs on the calling (main) thread. The subscription callback keeps
        firing normally throughout because the node is spun continuously
        by the background executor thread (see class docstring) — this
        loop does not need to, and must not, spin the node itself.
        """
        if abs(distance) < 1e-3:
            return

        twist = Twist()
        twist.linear.x = float(np.sign(distance) * DRIVE_SPEED)
        duration = abs(distance) / DRIVE_SPEED
        period   = 1.0 / DRIVE_RATE_HZ
        t0       = time.monotonic()

        while rclpy.ok():
            elapsed = time.monotonic() - t0
            if elapsed >= duration:
                break
            self.cmd_vel_pub.publish(twist)
            time.sleep(period)

        self._stop()

    def _stop(self):
        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.2)

    def _call_service_sync(self, client, request, timeout_sec: float):
        """
        Call a service asynchronously and block the calling thread until
        it completes or times out, WITHOUT spinning the node from this
        thread. The node is already being spun by the background executor
        (see class docstring), so calling rclpy.spin_until_future_complete()
        here would spin the same node from two threads at once and is
        unsafe. A threading.Event driven by the future's done-callback is
        the correct pattern for a node spun on a dedicated executor thread.
        """
        done_event = threading.Event()

        def _on_done(_future: Future):
            done_event.set()

        future = client.call_async(request)
        future.add_done_callback(_on_done)

        if not done_event.wait(timeout=timeout_sec):
            # Do not leave a dangling callback on a future we're abandoning.
            future.cancel()
            raise TimeoutError(
                f"Service call to '{client.srv_name}' timed out after "
                f"{timeout_sec}s")

        exc = future.exception()
        if exc is not None:
            raise RuntimeError(
                f"Service call to '{client.srv_name}' raised an exception: "
                f"{exc}")

        return future.result()

    def _move_arm(self, joint_positions: dict):
        """Send a joint position goal and block until motion completes."""
        req = SetJointPosition.Request()
        req.joint_position = JointPosition()
        req.joint_position.joint_name = list(joint_positions.keys())
        req.joint_position.position   = [float(v) for v in
                                          joint_positions.values()]
        req.path_time = ARM_MOVE_TIME

        resp = self._call_service_sync(
            self._set_joint_client, req, SERVICE_CALL_TIMEOUT_SEC)
        if resp is None or not resp.is_planned:
            raise RuntimeError(
                "Arm motion planning failed: " + str(joint_positions)
            )
        time.sleep(ARM_MOVE_TIME + 0.4)

    def _move_gripper(self, position: float):
        """Open or close the gripper. Blocks until motion completes."""
        req = SetJointPosition.Request()
        req.joint_position = JointPosition()
        req.joint_position.joint_name = ["gripper"]
        req.joint_position.position   = [float(position)]
        req.path_time = 1.0

        resp = self._call_service_sync(
            self._set_gripper_client, req, SERVICE_CALL_TIMEOUT_SEC)
        if resp is None or not resp.is_planned:
            raise RuntimeError("Gripper motion failed.")
        time.sleep(1.3)

    # =========================================================================
    # Pointcloud helpers
    # =========================================================================

    def _cloud_cb(self, msg: PointCloud2):
        with self._cloud_lock:
            self.latest_cloud = msg

    def _wait_for_cloud(self, timeout: float = 15.0):
        """
        Block the calling thread until the first pointcloud arrives.
        Uses plain polling with time.sleep rather than spin_once: the
        node is already being spun continuously by the background
        executor thread, so this thread only needs to check the
        flag/value the callback sets (see class docstring).
        """
        t0 = time.monotonic()
        while True:
            with self._cloud_lock:
                if self.latest_cloud is not None:
                    break
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    "[Demo] Timed out waiting for pointcloud on "
                    f"'{CLOUD_TOPIC}'. Is depth_publisher.py / "
                    "depth_to_pointcloud.py running, and does its QoS "
                    "match CLOUD_QOS in this script?")
            time.sleep(0.1)
        self.get_logger().info("[Demo] Pointcloud received.")

    def _unpack_cloud(self, msg: PointCloud2) -> np.ndarray:
        """
        Convert a PointCloud2 message to an (N, 3) float32 array of
        x, y, z. Validates that the required fields exist before
        attempting to read, so a malformed/incompatible cloud raises a
        clear ValueError instead of an obscure exception from pc2.
        """
        present_fields = {f.name for f in msg.fields}
        missing = [f for f in REQUIRED_CLOUD_FIELDS if f not in present_fields]
        if missing:
            raise ValueError(
                f"PointCloud2 message is missing required field(s) "
                f"{missing}; fields present: {sorted(present_fields)}")

        # sensor_msgs_py.point_cloud2.read_points() on ROS 2 Humble returns
        # a structured numpy.ndarray (named fields 'x','y','z'), NOT a
        # generator of tuples as in ROS 1 / older sensor_msgs_py. Index by
        # field name and stack into a plain (N, 3) float32 array.
        structured = pc2.read_points(
            msg, field_names=REQUIRED_CLOUD_FIELDS, skip_nans=True)

        if structured.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        points = np.column_stack([
            np.asarray(structured["x"], dtype=np.float32),
            np.asarray(structured["y"], dtype=np.float32),
            np.asarray(structured["z"], dtype=np.float32),
        ])
        return points

    def _filter_roi(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        mask = (
            (points[:, 0] >= ROI_X_RANGE[0]) & (points[:, 0] <= ROI_X_RANGE[1]) &
            (points[:, 1] >= ROI_Y_RANGE[0]) & (points[:, 1] <= ROI_Y_RANGE[1]) &
            (points[:, 2] >= ROI_Z_RANGE[0]) & (points[:, 2] <= ROI_Z_RANGE[1])
        )
        return points[mask]

    def _extract_clusters(self, points: np.ndarray,
                          voxel: float = 0.02,
                          radius: float = 0.25,
                          max_clusters: int = 8) -> list:
        """
        Voxel-density clustering. Returns a list of point arrays,
        one per cluster, sorted by distance (closest first) so the
        robot tries the nearest graspable object.
        """
        if len(points) == 0:
            return []

        remaining = points.copy()
        clusters  = []

        for _ in range(max_clusters):
            if len(remaining) < MIN_POINTS_FOR_OBJECT:
                break

            # Find densest voxel in remaining points
            vox_idx  = np.floor(remaining / voxel).astype(int)
            unique, counts = np.unique(vox_idx, axis=0, return_counts=True)
            densest  = unique[np.argmax(counts)]
            centroid = densest * voxel + voxel / 2.0

            # All points within radius of that centroid -> one cluster
            dist = np.linalg.norm(remaining - centroid, axis=1)
            mask = dist < radius
            cluster = remaining[mask]

            if len(cluster) >= MIN_POINTS_FOR_OBJECT:
                clusters.append(cluster)

            # Remove this cluster from remaining and repeat
            remaining = remaining[~mask]

        # Sort clusters by forward distance (Z) - closest first
        clusters.sort(key=lambda c: float(c[:, 2].mean()))
        return clusters

    def _estimate_dimensions(self, points: np.ndarray):
        """5th-95th percentile bounding box -> width, height, depth."""
        p5  = np.percentile(points,  5, axis=0)
        p95 = np.percentile(points, 95, axis=0)
        ext = p95 - p5
        return float(ext[0]), float(ext[1]), float(ext[2])


# -- Entry point --------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = None
    spin_thread = None
    exit_code = 0

    try:
        node = GraspDemo()

        # The node is spun continuously on a dedicated background thread
        # for the node's entire lifetime. This is what keeps the
        # PointCloud2 subscription callback (and service-future
        # completion callbacks) running while run() blocks the main
        # thread in time.sleep()/threading.Event.wait() during drives,
        # arm moves, and service calls. A SingleThreadedExecutor is
        # sufficient: this node has only one subscription callback and
        # future-done callbacks, none of which block each other.
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(
            target=executor.spin, name="ros_spin_thread", daemon=True)
        spin_thread.start()

        node.run()

    except (RuntimeError, TimeoutError) as exc:
        if node is not None:
            node.get_logger().error(f"[Demo] Aborted: {exc}")
        else:
            print(f"[Demo] Aborted during startup: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("[Demo] Interrupted by user.")
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())