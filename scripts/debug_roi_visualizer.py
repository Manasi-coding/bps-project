#!/usr/bin/env python3
"""
debug_roi_visualizer.py
------------------------
Standalone diagnostic tool for the BPS grasp-estimation pipeline.

Purpose
-------
grasp_estimator.py has been reporting roi_points=0 for several objects,
and multiple rounds of numeric debugging (percentile printouts, cluster
counts, etc.) have not been conclusive enough to isolate the root cause
among several candidates:
    - wrong coordinate frame (camera-optical vs world)
    - wrong ROI placement/size
    - wrong depth->pointcloud transform (range vs Z-depth, lens model)
    - incorrect assumed object positions
    - some combination of the above

This script does NOT modify, patch, or interact with the pipeline in any
way. It only subscribes to the pointcloud topic, loads ground truth, and
renders everything so the actual geometric relationship between the cloud
and the expected object positions can be seen directly, instead of
inferred from percentile numbers.

IMPORTANT -- this script makes NO assumptions about camera pose,
orientation, or any camera->world transform. Earlier iterations of this
tool hardcoded a candidate transform derived from one specific world's
camera SDF pose; that was scene-specific, easy to get wrong (and, in
practice, WAS wrong more than once during debugging), and gave false
confidence. This version reports only what can be known without
guessing: the PointCloud2 message's own frame_id, and (if a TF tree is
actually being published) whatever transform tf2_ros can resolve for us.
If neither tells us anything, the script says so explicitly rather than
inventing a transform.

Usage
-----
    python3 scripts/debug_roi_visualizer.py \
        --scene_name world1_primitives \
        --object_name obj_cube \
        --cloud_topic /depth_model/pointcloud \
        --world_frame world

If --object_name is omitted, all objects' GT centres are still plotted
(as red dots + labels), but no ROI box is drawn.

This script is intentionally verbose in both comments and terminal
output -- the goal is diagnostic clarity, not code brevity.
"""

import argparse
import json
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# tf2_ros is optional at runtime: some setups running this pipeline may
# not have a TF tree published at all (nothing in the pipeline as built
# so far publishes one). We import defensively and degrade gracefully
# rather than hard-requiring TF, per requirement 2 ("if TF is available,
# use it; otherwise state that no transform is available").
try:
    import tf2_ros
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

import matplotlib
# Use an interactive backend. TkAgg is the most broadly available on a
# typical Ubuntu desktop session; if it's not installed, matplotlib will
# raise an ImportError with a clear message telling you what to install
# (e.g. `sudo apt install python3-tk`).
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


# ── Defaults, mirrored from grasp_estimator.py ────────────────────────────
# These MUST match grasp_estimator.py's DEFAULT_* constants exactly, so
# that "use defaults if parameters are not supplied" means the same thing
# in both scripts. If grasp_estimator.py's defaults ever change, update
# these too.
DEFAULT_ROI_X_RANGE = (0.2, 1.2)
DEFAULT_ROI_Y_RANGE = (-0.4, 0.4)
DEFAULT_ROI_Z_RANGE = (0.0, 0.5)
DEFAULT_ROI_HALF_WIDTH = 0.15   # matches log_results.py's per-object ROI half-width
DEFAULT_CLOUD_TOPIC = "/depth_model/pointcloud"
DEFAULT_GROUND_TRUTH_DIR = "ground_truth"
DEFAULT_WORLD_FRAME = "world"   # the frame ground-truth poses are assumed to be in

DOWNSAMPLE_TARGET_POINTS = 10000

# Length (metres) of the world coordinate axes drawn at the origin, per
# requirement 4. Chosen to be visible against a small tabletop-scale
# scene without dominating the plot; adjust with --axis_length if a
# given scene's scale makes this too big/small to be useful.
DEFAULT_AXIS_LENGTH = 0.3


# ── Ground truth loading ───────────────────────────────────────────────────
def load_ground_truth(scene_name: str, gt_dir: str = DEFAULT_GROUND_TRUTH_DIR) -> dict:
    """
    Load ground_truth/<scene_name>_pose.json and normalise it into a
    dict: object_name -> {x, y, z, width, height, depth, ...}.

    Mirrors grasp_estimator.py's _load_ground_truth_json() normalisation
    logic (tolerates the {"objects": [...]} shape, a flat dict shape, and
    a bare list shape), since we want this script to work on the exact
    same ground-truth files the real pipeline uses.
    """
    path = os.path.join(gt_dir, f"{scene_name}_pose.json")
    if not os.path.isfile(path):
        print(f"[ERROR] Ground-truth file not found: {path}")
        sys.exit(1)

    with open(path, "r") as fh:
        data = json.load(fh)

    raw = data.get("objects", data) if isinstance(data, dict) else data

    objects = {}
    if isinstance(raw, dict):
        objects = raw
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and "name" in entry:
                objects[entry["name"]] = entry
    else:
        print(f"[ERROR] Unrecognised ground-truth structure in {path}")
        sys.exit(1)

    if not objects:
        print(f"[ERROR] No objects found in {path}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(objects)} ground-truth object(s) from {path}: "
          f"{list(objects.keys())}")
    return objects


# ── ROS 2 node: subscribes once, then hands control back to matplotlib ────
class PointCloudGrabber(Node):
    """
    Minimal ROS 2 node whose only job is to grab ONE PointCloud2 message
    off the target topic using the exact QoS profile grasp_estimator.py
    uses, then stop spinning. We deliberately do not build a long-running
    node here -- this script is a one-shot diagnostic snapshot, not a
    live pipeline component.

    It also optionally attempts to resolve a TF transform between the
    cloud's own frame_id and a requested world_frame, using tf2_ros, if
    tf2_ros is importable and a transform is actually being published.
    We do NOT invent or hardcode any transform ourselves -- if TF has
    nothing for us, we say so and move on (requirement 2 / 7).
    """

    def __init__(self, cloud_topic: str, world_frame: str, tf_wait_sec: float):
        super().__init__("debug_roi_visualizer")

        # MOD/NOTE: this QoS profile is copied verbatim from
        # grasp_estimator.py's subscription to /depth_model/pointcloud.
        # If the QoS here doesn't match the publisher's QoS
        # (depth_to_pointcloud.py, which also uses BEST_EFFORT), the
        # subscription silently never receives anything -- this exact
        # class of bug (QoS mismatch) has already bitten this pipeline
        # once before (see: earlier "New publisher discovered ... offering
        # incompatible QoS" warning when a probe script used default
        # RELIABLE QoS against a BEST_EFFORT publisher). Getting this
        # profile right is not optional.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cloud_topic = cloud_topic
        self.world_frame = world_frame
        self.tf_wait_sec = tf_wait_sec
        self.latest_msg: PointCloud2 = None

        self.create_subscription(
            PointCloud2, cloud_topic, self._callback, qos
        )

        # ── Optional TF listener ────────────────────────────────────
        # We do not assume TF is being published anywhere in this
        # pipeline (as of this pipeline's current state, nothing
        # publishes a TF tree). This is purely opportunistic: if tf2_ros
        # is available AND a transform actually shows up, we report it;
        # otherwise we say plainly that no transform is available. This
        # is the only sanctioned source of a frame transform in this
        # script -- there is no hardcoded fallback.
        self.tf_buffer = None
        self.tf_listener = None
        if TF2_AVAILABLE:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print(f"[INFO] Subscribed to '{cloud_topic}' with QoS "
              f"(BEST_EFFORT, KEEP_LAST, depth=1). Waiting for one message...")

    def _callback(self, msg: PointCloud2):
        # Only keep the first message we see -- we want a single
        # deterministic snapshot to inspect, not a moving target.
        if self.latest_msg is None:
            self.latest_msg = msg
            self.get_logger().info(
                f"Received pointcloud: width={msg.width} height={msg.height} "
                f"point_step={msg.point_step} frame_id='{msg.header.frame_id}'"
            )

    def try_lookup_transform(self, source_frame: str):
        """
        Attempt to resolve a TF transform from source_frame to
        self.world_frame. Returns the geometry_msgs/TransformStamped on
        success, or None if TF is unavailable, no such transform exists,
        or the frames are identical (nothing to resolve).

        This NEVER fabricates a transform -- it either finds a real one
        via tf2_ros, or returns None. Callers must handle the None case
        by reporting "no transform available", not by guessing one.
        """
        if not TF2_AVAILABLE or self.tf_buffer is None:
            return None
        if not source_frame or source_frame == self.world_frame:
            return None

        try:
            # Give TF a short window to receive any static/dynamic
            # transforms that might be published, rather than checking
            # instantaneously (which would almost always miss even a
            # genuinely-available static transform on a fresh node).
            deadline = self.get_clock().now().nanoseconds + int(self.tf_wait_sec * 1e9)
            while self.get_clock().now().nanoseconds < deadline:
                if self.tf_buffer.can_transform(
                    self.world_frame, source_frame, rclpy.time.Time()
                ):
                    return self.tf_buffer.lookup_transform(
                        self.world_frame, source_frame, rclpy.time.Time()
                    )
                rclpy.spin_once(self, timeout_sec=0.1)
        except Exception as e:
            # tf2_ros raises several distinct exception types
            # (LookupException, ConnectivityException,
            # ExtrapolationException, ...) for "no transform found" --
            # we collapse all of them to "no transform available" here,
            # since the diagnostic action is identical in every case:
            # report it plainly, do not guess.
            print(f"[INFO] No TF transform resolved from '{source_frame}' to "
                  f"'{self.world_frame}': {e}")
            return None

        return None


def grab_one_cloud_and_tf(cloud_topic: str, world_frame: str,
                          timeout_sec: float, tf_wait_sec: float):
    """
    Spin a temporary node until exactly one PointCloud2 message arrives,
    or the timeout elapses. Also makes one opportunistic attempt to
    resolve a TF transform from the cloud's frame_id to world_frame.

    Returns (msg, tf_transform_or_None).
    """
    rclpy.init(args=None)
    node = PointCloudGrabber(cloud_topic, world_frame, tf_wait_sec)

    deadline = node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
    while node.latest_msg is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > deadline:
            print(f"[ERROR] Timed out after {timeout_sec:.1f}s waiting for a "
                  f"message on '{cloud_topic}'.")
            print("        Check that depth_to_pointcloud.py (and the rest of "
                  "the pipeline: Gazebo, the bridge, depth_publisher.py) is "
                  "actually running and publishing on this topic.")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

    msg = node.latest_msg

    tf_transform = node.try_lookup_transform(msg.header.frame_id)

    node.destroy_node()
    rclpy.shutdown()
    return msg, tf_transform


# ── PointCloud2 -> NumPy ────────────────────────────────────────────────────
def unpack_cloud(msg: PointCloud2) -> np.ndarray:
    """
    Convert a PointCloud2 message into an (N, 3) float32 NumPy array of
    (x, y, z) points, in whatever frame the publisher put them in
    (this script makes NO assumption about that frame -- reporting the
    frame_id, and resolving TF if available, is exactly what this tool
    does instead of guessing).
    """
    structured = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if structured.size == 0:
        print("[WARN] Unpacked pointcloud is EMPTY (0 points after skip_nans).")
        return np.empty((0, 3), dtype=np.float32)

    arr = np.column_stack(
        (structured["x"], structured["y"], structured["z"])
    ).astype(np.float32, copy=False)
    return arr


def downsample(points: np.ndarray, target: int = DOWNSAMPLE_TARGET_POINTS) -> np.ndarray:
    """
    Randomly downsample to ~target points for responsive interactive
    plotting. Uses a fixed seed so repeated runs on the same cloud show
    the same subsample (useful when comparing before/after a code
    change -- you're not fighting random-sample noise on top of a real
    change).
    """
    n = len(points)
    if n <= target:
        return points
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n, size=target, replace=False)
    return points[idx]


# ── ROI box construction, mirroring grasp_estimator.py / log_results.py ──
def compute_roi(object_name: str, gt_objects: dict,
                 roi_x_range, roi_y_range, roi_z_range,
                 half_width: float):
    """
    Reconstruct the exact ROI box grasp_estimator.py / log_results.py
    would use for this object.

    log_results.py centres the ROI on the object's ground-truth (x, y)
    with a fixed half_width, and uses a separately-configured z range
    (table-surface band). If the object isn't found in ground truth, or
    no object was specified, fall back to the raw roi_x/y/z_range
    defaults (i.e. the un-centred box grasp_estimator.py would use with
    no per-object override applied) so this script still produces a
    sensible box rather than crashing.
    """
    if object_name is None:
        return roi_x_range, roi_y_range, roi_z_range, None

    entry = gt_objects.get(object_name)
    if entry is None:
        print(f"[WARN] Object '{object_name}' not found in ground truth -- "
              f"falling back to raw (uncentred) ROI defaults.")
        return roi_x_range, roi_y_range, roi_z_range, None

    gt_x = float(entry["x"])
    gt_y = float(entry["y"])
    gt_z = float(entry.get("z", 0.0))

    centred_x_range = (gt_x - half_width, gt_x + half_width)
    centred_y_range = (gt_y - half_width, gt_y + half_width)
    # z range is NOT centred on the object in the real pipeline -- it's
    # a fixed table-surface band (roi_z_range), independent of object
    # position, so we leave it as-is rather than centring it here too.
    return centred_x_range, centred_y_range, roi_z_range, (gt_x, gt_y, gt_z)


def roi_mask(points: np.ndarray, roi_x_range, roi_y_range, roi_z_range) -> np.ndarray:
    """Boolean mask of which points fall inside the given ROI box."""
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points[:, 0] >= roi_x_range[0]) & (points[:, 0] <= roi_x_range[1]) &
        (points[:, 1] >= roi_y_range[0]) & (points[:, 1] <= roi_y_range[1]) &
        (points[:, 2] >= roi_z_range[0]) & (points[:, 2] <= roi_z_range[1])
    )


# ── Terminal diagnostics (requirement 5) ───────────────────────────────────
def print_diagnostics(points: np.ndarray, roi_x_range, roi_y_range, roi_z_range,
                       object_centre, object_name, cloud_frame_id: str,
                       world_frame: str, tf_transform):
    print("\n" + "=" * 60)
    print("CLOUD / ROI DIAGNOSTICS")
    print("=" * 60)

    # ── Frame reporting (requirement 2 / 5) ────────────────────────
    # We report exactly what was observed: the PointCloud2 message's own
    # frame_id, the frame ground truth is assumed to be expressed in
    # (world_frame, as given via --world_frame), and whether TF actually
    # resolved a transform between them. No transform is assumed or
    # invented if none of this tells us anything.
    print(f"Cloud frame_id (as published): '{cloud_frame_id}'")
    print(f"Ground-truth ('world') frame:  '{world_frame}'")
    if cloud_frame_id == world_frame:
        print("  -> Cloud frame_id matches the ground-truth frame name "
              "exactly. This does NOT by itself prove the data is "
              "actually expressed in that frame -- frame_id is whatever "
              "the publisher chose to put in the header, correctly or "
              "not -- but there is no frame *name* mismatch to flag.")
    elif tf_transform is not None:
        t = tf_transform.transform.translation
        r = tf_transform.transform.rotation
        print(f"  -> TF transform RESOLVED from '{cloud_frame_id}' to "
              f"'{world_frame}':")
        print(f"       translation: x={t.x:.4f} y={t.y:.4f} z={t.z:.4f}")
        print(f"       rotation (quaternion): x={r.x:.4f} y={r.y:.4f} "
              f"z={r.z:.4f} w={r.w:.4f}")
        print("     (This transform came from tf2_ros / the TF tree -- "
              "it was not computed or assumed by this script.)")
    else:
        print(f"  -> NO transform available from '{cloud_frame_id}' to "
              f"'{world_frame}'.")
        if not TF2_AVAILABLE:
            print("     Reason: tf2_ros is not importable in this "
                  "environment.")
        else:
            print("     Reason: tf2_ros is available, but no TF tree is "
                  "publishing a transform between these two frames "
                  "(nothing in this pipeline currently publishes TF).")
        print("     This script will NOT guess a transform. The point "
              "cloud below is plotted exactly as received, in its own "
              "frame_id, with no coordinate change applied.")

    if len(points) == 0:
        print("\nCloud is EMPTY -- no bounds to report.")
    else:
        print("\nCloud bounds:")
        print(f"  X[{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
        print(f"  Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
        print(f"  Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    if object_centre is not None:
        print(f"\nObject centre ('{object_name}', from ground truth, "
              f"frame='{world_frame}'):")
        print(f"  x={object_centre[0]:.3f}  y={object_centre[1]:.3f}  z={object_centre[2]:.3f}")
    else:
        print(f"\nObject centre: N/A (object_name not specified or not found in GT)")

    print(f"\nROI limits:")
    print(f"  xmin={roi_x_range[0]:.3f}  xmax={roi_x_range[1]:.3f}")
    print(f"  ymin={roi_y_range[0]:.3f}  ymax={roi_y_range[1]:.3f}")
    print(f"  zmin={roi_z_range[0]:.3f}  zmax={roi_z_range[1]:.3f}")

    mask = roi_mask(points, roi_x_range, roi_y_range, roi_z_range)
    n_inside = int(np.count_nonzero(mask))
    print(f"\nNumber of cloud points inside ROI: {n_inside} / {len(points)}")

    if object_centre is not None and len(points) > 0:
        cx, cy, cz = object_centre
        inside_x = roi_x_range[0] <= cx <= roi_x_range[1]
        inside_y = roi_y_range[0] <= cy <= roi_y_range[1]
        inside_z = roi_z_range[0] <= cz <= roi_z_range[1]
        print(f"\nIs the GT object centre itself inside the ROI box?")
        print(f"  x: {'YES' if inside_x else 'NO'} ({cx:.3f} vs "
              f"[{roi_x_range[0]:.3f},{roi_x_range[1]:.3f}])")
        print(f"  y: {'YES' if inside_y else 'NO'} ({cy:.3f} vs "
              f"[{roi_y_range[0]:.3f},{roi_y_range[1]:.3f}])")
        print(f"  z: {'YES' if inside_z else 'NO'} ({cz:.3f} vs "
              f"[{roi_z_range[0]:.3f},{roi_z_range[1]:.3f}])")
        if not (inside_x and inside_y and inside_z):
            print("  -> The ROI box does not even contain the object's own "
                  "GT centre. This alone would explain roi_points=0/low, "
                  "independent of anything about the cloud itself.")

    print("=" * 60 + "\n")


# ── Plotting helpers ────────────────────────────────────────────────────────
def draw_wireframe_box(ax, x_range, y_range, z_range, color="blue", label=None):
    """
    Draw a wireframe (edges-only) box on a 3D matplotlib axis, spanning
    x_range x y_range x z_range. Used to show the ROI boundary.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    # 8 corners of the box
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom face
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top face
    ])

    # 12 edges, as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    first = True
    for i, j in edges:
        xs = [corners[i, 0], corners[j, 0]]
        ys = [corners[i, 1], corners[j, 1]]
        zs = [corners[i, 2], corners[j, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.5,
                label=(label if first else None))
        first = False


def draw_world_axes(ax, length: float = DEFAULT_AXIS_LENGTH):
    """
    Draw X (red), Y (green), Z (blue) axis lines at the origin, per
    requirement 4, purely for visual orientation. This draws the axes of
    whatever frame the plotted points are actually in -- it does NOT
    imply or assert that this origin corresponds to any particular
    real-world location; it's just "the (0,0,0) of the data as given."
    """
    origin = np.zeros(3)
    axes = {
        "X": (np.array([length, 0, 0]), "red"),
        "Y": (np.array([0, length, 0]), "green"),
        "Z": (np.array([0, 0, length]), "blue"),
    }
    for label, (vec, color) in axes.items():
        ax.plot([origin[0], vec[0]], [origin[1], vec[1]], [origin[2], vec[2]],
                color=color, linewidth=2.0)
        ax.text(vec[0], vec[1], vec[2], label, color=color, fontsize=10,
                fontweight="bold")


def plot_scene(ax, points, mask, gt_objects, roi_x_range, roi_y_range,
               roi_z_range, object_name, title, axis_length):
    """
    Plot the point cloud (split into in-ROI / out-of-ROI colouring per
    requirement 3), GT object markers, the ROI wireframe box, and the
    world axes, all on a single 3D axis.
    """
    if len(points) > 0:
        inside = points[mask]
        outside = points[~mask]

        if len(outside) > 0:
            ax.scatter(outside[:, 0], outside[:, 1], outside[:, 2],
                       s=1, c="lightgrey", alpha=0.4, label="outside ROI")
        if len(inside) > 0:
            ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2],
                       s=4, c="green", alpha=0.9, label="inside ROI")

    for name, entry in gt_objects.items():
        gx, gy, gz = float(entry["x"]), float(entry["y"]), float(entry.get("z", 0.0))
        ax.scatter([gx], [gy], [gz], c="red", s=60, marker="o",
                   edgecolors="black", linewidths=0.5)
        ax.text(gx, gy, gz, f"  {name}", color="red", fontsize=8)

    if object_name is not None:
        draw_wireframe_box(ax, roi_x_range, roi_y_range, roi_z_range,
                           color="blue", label="ROI box")

    draw_world_axes(ax, length=axis_length)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Standalone visual debugger for the BPS grasp pipeline's "
                    "pointcloud / ROI / ground-truth alignment. Makes no "
                    "assumptions about camera pose or coordinate transforms "
                    "-- reports frame_id and TF (if available) as-is."
    )
    parser.add_argument("--scene_name", type=str, required=True,
                        help="Scene name, e.g. world1_primitives. Used to "
                             "locate ground_truth/<scene_name>_pose.json")
    parser.add_argument("--object_name", type=str, default=None,
                        help="Object to centre the ROI box on (e.g. "
                             "obj_cube). If omitted, all GT objects are "
                             "still plotted, but no ROI box is drawn.")
    parser.add_argument("--cloud_topic", type=str, default=DEFAULT_CLOUD_TOPIC,
                        help=f"PointCloud2 topic to subscribe to "
                             f"(default: {DEFAULT_CLOUD_TOPIC})")
    parser.add_argument("--ground_truth_dir", type=str,
                        default=DEFAULT_GROUND_TRUTH_DIR,
                        help=f"Directory containing <scene>_pose.json "
                             f"(default: {DEFAULT_GROUND_TRUTH_DIR})")
    parser.add_argument("--world_frame", type=str, default=DEFAULT_WORLD_FRAME,
                        help="Name of the frame ground-truth poses are "
                             f"assumed to be expressed in (default: "
                             f"'{DEFAULT_WORLD_FRAME}'). Used only for "
                             "reporting/TF lookup -- never to compute a "
                             "transform ourselves.")
    parser.add_argument("--tf_wait", type=float, default=2.0,
                        help="Seconds to wait for a TF transform to "
                             "become available before giving up on it "
                             "(default: 2.0)")
    parser.add_argument("--roi_half_width", type=float,
                        default=DEFAULT_ROI_HALF_WIDTH,
                        help="Half-width (metres) of the per-object ROI "
                             "box in x/y, matching log_results.py's "
                             "centring logic (default: "
                             f"{DEFAULT_ROI_HALF_WIDTH})")
    parser.add_argument("--roi_x_min", type=float, default=None)
    parser.add_argument("--roi_x_max", type=float, default=None)
    parser.add_argument("--roi_y_min", type=float, default=None)
    parser.add_argument("--roi_y_max", type=float, default=None)
    parser.add_argument("--roi_z_min", type=float, default=None)
    parser.add_argument("--roi_z_max", type=float, default=None)
    parser.add_argument("--axis_length", type=float, default=DEFAULT_AXIS_LENGTH,
                        help="Length (metres) of the world-axis indicator "
                             f"lines drawn at the origin (default: "
                             f"{DEFAULT_AXIS_LENGTH})")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Seconds to wait for a pointcloud message "
                             "before giving up (default: 20.0)")
    args, _ = parser.parse_known_args()  # tolerate stray --ros-args passthrough

    # ── Resolve ROI ranges: explicit args override grasp_estimator.py's
    # defaults, matching requirement ("...or use the same defaults as
    # grasp_estimator.py if parameters are not supplied").
    roi_x_range = (
        args.roi_x_min if args.roi_x_min is not None else DEFAULT_ROI_X_RANGE[0],
        args.roi_x_max if args.roi_x_max is not None else DEFAULT_ROI_X_RANGE[1],
    )
    roi_y_range = (
        args.roi_y_min if args.roi_y_min is not None else DEFAULT_ROI_Y_RANGE[0],
        args.roi_y_max if args.roi_y_max is not None else DEFAULT_ROI_Y_RANGE[1],
    )
    roi_z_range = (
        args.roi_z_min if args.roi_z_min is not None else DEFAULT_ROI_Z_RANGE[0],
        args.roi_z_max if args.roi_z_max is not None else DEFAULT_ROI_Z_RANGE[1],
    )

    # ── Load ground truth ──────────────────────────────────────────────
    gt_objects = load_ground_truth(args.scene_name, args.ground_truth_dir)

    # ── Grab exactly one pointcloud message (+ opportunistic TF) ───────
    print(f"[INFO] Connecting to ROS 2 and waiting for a message on "
          f"'{args.cloud_topic}' (timeout={args.timeout}s)...")
    if not TF2_AVAILABLE:
        print("[INFO] tf2_ros is not importable in this environment -- "
              "TF-based frame resolution will be skipped. Only the "
              "PointCloud2 message's own frame_id will be reported.")
    msg, tf_transform = grab_one_cloud_and_tf(
        args.cloud_topic, args.world_frame, args.timeout, args.tf_wait
    )

    print(f"[INFO] Message frame_id='{msg.header.frame_id}', "
          f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    # ── Unpack + downsample ────────────────────────────────────────────
    points_full = unpack_cloud(msg)
    print(f"[INFO] Unpacked {len(points_full)} raw points from the message.")
    points = downsample(points_full, DOWNSAMPLE_TARGET_POINTS)
    print(f"[INFO] Downsampled to {len(points)} points for plotting "
          f"(target was {DOWNSAMPLE_TARGET_POINTS}).")

    # ── Compute the ROI box exactly as the real pipeline would ─────────
    roi_x_range, roi_y_range, roi_z_range, object_centre = compute_roi(
        args.object_name, gt_objects, roi_x_range, roi_y_range, roi_z_range,
        args.roi_half_width
    )

    # ── Print terminal diagnostics (requirement 5) ─────────────────────
    print_diagnostics(points_full, roi_x_range, roi_y_range, roi_z_range,
                      object_centre, args.object_name, msg.header.frame_id,
                      args.world_frame, tf_transform)

    # ── Build the figure ────────────────────────────────────────────────
    # A single panel: the cloud exactly as received, no transform
    # applied or assumed. Points inside the ROI are highlighted green,
    # everything else grey (requirement 3); world axes are drawn at the
    # origin for orientation (requirement 4).
    mask = roi_mask(points, roi_x_range, roi_y_range, roi_z_range)

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    plot_scene(ax1, points, mask, gt_objects, roi_x_range, roi_y_range,
               roi_z_range, args.object_name,
               title=f"Cloud as received (topic frame_id="
                     f"'{msg.header.frame_id}')",
               axis_length=args.axis_length)

    print("[INFO] Rendering interactive 3D plot. Close the window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()