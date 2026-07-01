"""
grasp_estimator.py
-------------------
BPS pipeline experiment node (ROS 2 Humble / Gazebo Harmonic).

Reads the pointcloud from depth_to_pointcloud.py, isolates the dominant
object in a region of interest, estimates its bounding dimensions, and
decides whether a gripper of configurable size could grasp it. Ground
truth pose data is read from the pipeline's pre-exported
`ground_truth/<world_name>_pose.json` files (produced by export_pose.py).

export_pose.py emits per-object: name, x, y, z, roll, pitch, yaw,
width, height, depth. All fields are read here. Dimension-based evaluation
(gt_can_grasp, decision_correct, width_error_m, height_error_m) is active
whenever width/height are non-None in the JSON entry.

Usage:
    ros2 run <pkg> grasp_estimator.py --ros-args \
        -p object_name:=coffee_mug_sem1 \
        -p world_name:=world2_household \
        -p ground_truth_source:=json \
        -p ground_truth_dir:=/path/to/ground_truth \
        -p boundary_masks_dir:=/path/to/boundary_masks \
        -p semantic_masks_dir:=/path/to/data/masks
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


# ── Configuration (overridable via ROS 2 parameters) ─────────────────────
DEFAULT_CLOUD_TOPIC          = "/depth_model/pointcloud"
DEFAULT_GRIPPER_MAX_WIDTH    = 0.077   # metres
DEFAULT_GRIPPER_MAX_HEIGHT   = 0.120   # metres
DEFAULT_ROI_X_RANGE          = (0.2, 1.2)
DEFAULT_ROI_Y_RANGE          = (-0.4, 0.4)
DEFAULT_ROI_Z_RANGE          = (0.0, 0.5)
DEFAULT_MIN_POINTS           = 50
DEFAULT_VOXEL_SIZE           = 0.02
DEFAULT_CLUSTER_RADIUS       = 0.30
# MOD: eval timer was ticking (1 Hz) faster than the depth model can
# produce frames on CPU (~3-10s per frame), so every object logged 1-2
# "no pointcloud" warnings before a real frame arrived. Not a bug, just
# noisy — slow the timer down to roughly match real cadence.
DEFAULT_EVAL_RATE_HZ         = 0.2   # once every 5s
DEFAULT_CLOUD_STALENESS_SEC  = 2.0   # warn/skip if latest cloud older than this
DEFAULT_GROUND_TRUTH_DIR     = "ground_truth"
DEFAULT_BOUNDARY_MASKS_DIR   = "boundary_masks"
DEFAULT_SEMANTIC_MASKS_DIR   = "data/masks"


@dataclass
class EstimationResult:
    """Structured result so downstream log_results.py gets a stable schema
    instead of a loosely-typed dict (still exposed as dict via .to_dict())."""
    object_detected: bool
    world_name: str = ""
    object_name: str = ""
    estimated_width: float = 0.0
    estimated_height: float = 0.0
    estimated_depth: float = 0.0
    gt_width: Optional[float] = None
    gt_height: Optional[float] = None
    gt_depth: Optional[float] = None
    can_grasp: Optional[bool] = None
    gt_can_grasp: Optional[bool] = None
    decision_correct: Optional[bool] = None
    width_error_m: Optional[float] = None
    height_error_m: Optional[float] = None
    ground_truth_available: bool = False
    gt_dimensions_available: bool = False
    gt_pose_x: Optional[float] = None
    gt_pose_y: Optional[float] = None
    gt_pose_z: Optional[float] = None
    gt_roll: Optional[float] = None
    gt_pitch: Optional[float] = None
    gt_yaw: Optional[float] = None
    num_roi_points: int = 0
    num_cluster_points: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class GraspEstimator(Node):
    def __init__(self):
        super().__init__("grasp_estimator")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("cloud_topic", DEFAULT_CLOUD_TOPIC)
        self.declare_parameter("object_name", "")
        self.declare_parameter("world_name", "")
        self.declare_parameter("gripper_max_width", DEFAULT_GRIPPER_MAX_WIDTH)
        self.declare_parameter("gripper_max_height", DEFAULT_GRIPPER_MAX_HEIGHT)
        self.declare_parameter("roi_x_min", DEFAULT_ROI_X_RANGE[0])
        self.declare_parameter("roi_x_max", DEFAULT_ROI_X_RANGE[1])
        self.declare_parameter("roi_y_min", DEFAULT_ROI_Y_RANGE[0])
        self.declare_parameter("roi_y_max", DEFAULT_ROI_Y_RANGE[1])
        self.declare_parameter("roi_z_min", DEFAULT_ROI_Z_RANGE[0])
        self.declare_parameter("roi_z_max", DEFAULT_ROI_Z_RANGE[1])
        self.declare_parameter("min_points_for_object", DEFAULT_MIN_POINTS)
        self.declare_parameter("voxel_size", DEFAULT_VOXEL_SIZE)
        self.declare_parameter("cluster_radius", DEFAULT_CLUSTER_RADIUS)
        self.declare_parameter("eval_rate_hz", DEFAULT_EVAL_RATE_HZ)
        self.declare_parameter("cloud_staleness_sec", DEFAULT_CLOUD_STALENESS_SEC)
        self.declare_parameter("ground_truth_source", "json")
        self.declare_parameter("ground_truth_dir", DEFAULT_GROUND_TRUTH_DIR)
        self.declare_parameter("boundary_masks_dir", DEFAULT_BOUNDARY_MASKS_DIR)
        self.declare_parameter("semantic_masks_dir", DEFAULT_SEMANTIC_MASKS_DIR)

        self._load_params()

        # ── State ──────────────────────────────────────────────────────
        self.latest_cloud: Optional[PointCloud2] = None
        self.latest_cloud_stamp_sec: float = 0.0
        self._frames_received = 0
        self._frames_evaluated = 0
        self._frames_skipped_stale = 0
        self._frames_skipped_invalid = 0

        self._ground_truth_data: Optional[dict] = None
        self._ground_truth_load_warned = False
        if self.ground_truth_source == "json":
            self._ground_truth_data = self._load_ground_truth_json()
            if self._ground_truth_data is None:
                self.get_logger().warn(
                    "ground_truth_source='json' but no usable ground-truth "
                    "file could be loaded — ground-truth comparison will "
                    "be disabled for this run."
                )

        self._boundary_masks_path = os.path.join(
            self.boundary_masks_dir, self.world_name) if self.world_name else ""
        self._semantic_masks_path = os.path.join(
            self.semantic_masks_dir, self.world_name) if self.world_name else ""
        self._log_mask_dir_status("boundary masks", self._boundary_masks_path)
        self._log_mask_dir_status("semantic masks", self._semantic_masks_path)

        # ── QoS ────────────────────────────────────────────────────────
        cloud_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PointCloud2, self.cloud_topic, self._cloud_callback, cloud_qos
        )

        # ── Evaluation timer ───────────────────────────────────────────
        period = 1.0 / max(self.eval_rate_hz, 1e-3)
        self.create_timer(period, self._evaluate_tick)

        self.get_logger().info(
            f"grasp_estimator ready | "
            f"world='{self.world_name or '<unspecified>'}' | "
            f"object='{self.object_name or '<any>'}' | "
            f"GT='{self.ground_truth_source}' | "
            f"cloud='{self.cloud_topic}'"
        )

    # ── Parameter loading / validation ───────────────────────────────────
    def _load_params(self):
        gp = self.get_parameter

        def f(name):
            return float(gp(name).value)

        self.cloud_topic = str(gp("cloud_topic").value)
        self.object_name = str(gp("object_name").value)
        self.world_name = str(gp("world_name").value)
        self.gripper_max_width = f("gripper_max_width")
        self.gripper_max_height = f("gripper_max_height")
        self.roi_x_range = (f("roi_x_min"), f("roi_x_max"))
        self.roi_y_range = (f("roi_y_min"), f("roi_y_max"))
        self.roi_z_range = (f("roi_z_min"), f("roi_z_max"))
        self.min_points_for_object = int(gp("min_points_for_object").value)
        self.voxel_size = f("voxel_size")
        self.cluster_radius = f("cluster_radius")
        self.eval_rate_hz = f("eval_rate_hz")
        self.cloud_staleness_sec = f("cloud_staleness_sec")
        self.ground_truth_source = str(gp("ground_truth_source").value).lower()
        self.ground_truth_dir = str(gp("ground_truth_dir").value)
        self.boundary_masks_dir = str(gp("boundary_masks_dir").value)
        self.semantic_masks_dir = str(gp("semantic_masks_dir").value)

        problems = []
        if self.roi_x_range[0] >= self.roi_x_range[1]:
            problems.append("roi_x_min must be < roi_x_max")
        if self.roi_y_range[0] >= self.roi_y_range[1]:
            problems.append("roi_y_min must be < roi_y_max")
        if self.roi_z_range[0] >= self.roi_z_range[1]:
            problems.append("roi_z_min must be < roi_z_max")
        if self.gripper_max_width <= 0 or self.gripper_max_height <= 0:
            problems.append("gripper_max_width/height must be > 0")
        if self.min_points_for_object <= 0:
            problems.append("min_points_for_object must be > 0")
        if self.voxel_size <= 0:
            problems.append("voxel_size must be > 0")
        if self.cluster_radius <= 0:
            problems.append("cluster_radius must be > 0")
        if self.eval_rate_hz <= 0:
            problems.append("eval_rate_hz must be > 0")
        if self.ground_truth_source not in ("json", "none"):
            problems.append("ground_truth_source must be 'json' or 'none'")

        if problems:
            msg = "Invalid parameters: " + "; ".join(problems)
            self.get_logger().fatal(msg)
            raise ValueError(msg)

    def _log_mask_dir_status(self, label: str, path: str):
        if not path:
            self.get_logger().warn(
                f"{label} directory unresolved (world_name not set) — "
                f"cannot verify pipeline layout for {label}."
            )
            return
        if os.path.isdir(path):
            self.get_logger().info(f"{label} directory found: {path}")
        else:
            self.get_logger().warn(
                f"{label} directory not found: {path} — this is non-fatal "
                f"for grasp_estimator, but other BPS pipeline stages "
                f"consuming {label} may fail."
            )

    # ── Ground truth (JSON-based) ────────────────────────────────────────
    def _load_ground_truth_json(self) -> Optional[dict]:
        """
        Load ground_truth/<world_name>_pose.json (produced by export_pose.py).

        export_pose.py emits objects with fields:
            name, x, y, z, roll, pitch, yaw, width, height, depth

        Width/height/depth are extracted from SDF geometry by export_pose.py
        and will be non-None for box/cylinder/sphere/cone primitives. They
        may be None if the geometry type was unrecognised.

        Also tolerant of older formats (flat dict keyed by object name, or
        bare list) for backward compatibility.
        """
        if not self.world_name:
            self.get_logger().warn(
                "ground_truth_source='json' but no world_name parameter "
                "was set — cannot locate the ground-truth file."
            )
            return None

        path = os.path.join(self.ground_truth_dir, f"{self.world_name}_pose.json")
        if not os.path.isfile(path):
            self.get_logger().warn(f"Ground-truth file not found: {path}")
            return None

        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Failed to read/parse {path}: {e}")
            return None

        # Normalise three possible shapes into object_name -> entry dict.
        raw = data.get("objects", data) if isinstance(data, dict) else data

        objects: dict[str, dict] = {}

        if isinstance(raw, dict):
            objects = raw
        elif isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict) and "name" in entry:
                    objects[entry["name"]] = entry
                else:
                    self.get_logger().warn(
                        f"Skipping malformed ground-truth entry in {path}: {entry!r}"
                    )
        else:
            self.get_logger().error(
                f"Ground-truth file {path} has an unexpected top-level "
                f"structure ({type(raw)}) — ignoring."
            )
            return None

        # Report whether dimensions are present in this file.
        n_with_dims = sum(
            1 for e in objects.values()
            if isinstance(e, dict)
            and e.get("width") is not None
            and e.get("height") is not None
        )
        self.get_logger().info(
            f"Loaded ground-truth data for world '{self.world_name}' "
            f"from {path} ({len(objects)} objects, "
            f"{n_with_dims} with dimensions)."
        )
        return objects

    def _get_ground_truth(self) -> Optional[dict]:
        """
        Look up the ground-truth entry for the configured object_name.

        Returns a dict with keys:
            x, y, z, roll, pitch, yaw          — always present if entry found
            width, height, depth                — present if export_pose.py
                                                  extracted them from SDF geometry;
                                                  None otherwise

        Returns None if ground truth is disabled, unavailable, or the
        object is not found.
        """
        if self.ground_truth_source == "none" or self._ground_truth_data is None:
            return None
        if not self.object_name:
            return None

        entry = self._ground_truth_data.get(self.object_name)
        if entry is None:
            self.get_logger().warn(
                f"Object '{self.object_name}' not found in ground-truth "
                f"data for world '{self.world_name}'.",
                throttle_duration_sec=10.0,
            )
            return None

        if not isinstance(entry, dict):
            self.get_logger().warn(
                f"Ground-truth entry for '{self.object_name}' is malformed "
                f"(expected a dict) — skipping.",
                throttle_duration_sec=10.0,
            )
            return None

        try:
            result = {
                "x":     float(entry["x"]),
                "y":     float(entry["y"]),
                "z":     float(entry["z"]),
                "roll":  float(entry["roll"]),
                "pitch": float(entry["pitch"]),
                "yaw":   float(entry["yaw"]),
                # Dimensions from export_pose.py — None if geometry was
                # unrecognised or the field is absent (older JSON format).
                "width":  float(entry["width"])  if entry.get("width")  is not None else None,
                "height": float(entry["height"]) if entry.get("height") is not None else None,
                "depth":  float(entry["depth"])  if entry.get("depth")  is not None else None,
            }
        except (KeyError, TypeError, ValueError) as e:
            self.get_logger().warn(
                f"Ground-truth entry for '{self.object_name}' has missing "
                f"or malformed fields ({e}) — skipping.",
                throttle_duration_sec=10.0,
            )
            return None

        if not self._ground_truth_load_warned:
            has_dims = result["width"] is not None and result["height"] is not None
            self.get_logger().info(
                f"Ground-truth loaded for '{self.object_name}': "
                f"pose (x={result['x']:.3f}, y={result['y']:.3f}, z={result['z']:.3f}) | "
                f"dims {'W=%.4fm H=%.4fm' % (result['width'], result['height']) if has_dims else 'not available'}"
            )
            self._ground_truth_load_warned = True

        return result

    # ── Callbacks ─────────────────────────────────────────────────────
    def _cloud_callback(self, msg: PointCloud2):
        self._frames_received += 1
        self.latest_cloud = msg
        self.latest_cloud_stamp_sec = time.monotonic()

    # ── Main evaluation loop (timer-driven) ──────────────────────────────
    def _evaluate_tick(self):
        result = self.estimate()
        if result is not None:
            self._log_result(result)

    def estimate(self) -> Optional[EstimationResult]:
        """Run one full estimation pass on the latest available cloud."""
        if self.latest_cloud is None:
            self.get_logger().warn(
                "No pointcloud received yet on '%s'." % self.cloud_topic,
                throttle_duration_sec=10.0,
            )
            return None

        age = time.monotonic() - self.latest_cloud_stamp_sec
        if age > self.cloud_staleness_sec:
            self._frames_skipped_stale += 1
            self.get_logger().warn(
                f"Latest pointcloud is {age:.2f}s old (> "
                f"{self.cloud_staleness_sec:.2f}s threshold) — skipping "
                f"this evaluation tick.",
                throttle_duration_sec=5.0,
            )
            return None

        cloud_msg = self.latest_cloud

        # ── Step 1: Extract points from ROS message ────────────────────
        try:
            points = self._unpack_cloud(cloud_msg)
        except Exception as e:
            self._frames_skipped_invalid += 1
            self.get_logger().error(f"Failed to unpack PointCloud2: {e}")
            return None

        if points.size == 0:
            self._frames_skipped_invalid += 1
            self.get_logger().warn(
                "Unpacked pointcloud is empty — skipping.",
                throttle_duration_sec=5.0,
            )
            return EstimationResult(object_detected=False,
                                     world_name=self.world_name,
                                     object_name=self.object_name)

        if points.ndim != 2 or points.shape[1] != 3:
            self._frames_skipped_invalid += 1
            self.get_logger().error(
                f"Unpacked pointcloud has unexpected shape {points.shape} "
                f"— expected (N, 3). Skipping.")
            return None

        if not np.all(np.isfinite(points)):
            finite_mask = np.isfinite(points).all(axis=1)
            n_bad = int(np.count_nonzero(~finite_mask))
            points = points[finite_mask]
            self.get_logger().warn(
                f"Dropped {n_bad} non-finite point(s) from cloud.",
                throttle_duration_sec=5.0,
            )

        # MOD: temporary diagnostic — print actual point cloud bounds to check
        # against roi_x/y/z_range assumptions
        self.get_logger().info(
            f"DEBUG cloud bounds: x[{points[:,0].min():.2f},{points[:,0].max():.2f}] "
            f"y[{points[:,1].min():.2f},{points[:,1].max():.2f}] "
            f"z[{points[:,2].min():.2f},{points[:,2].max():.2f}]",
            throttle_duration_sec=2.0,
        )

        # ── Step 2: Filter to region of interest ────────────────────────
        roi_points = self._filter_roi(points)

        # MOD: temporary diagnostic — see the actual z distribution inside
        # this object's ROI before applying any table-height cutoff, since
        # guessing TABLE_Z blind hasn't matched observed behavior.
        if len(roi_points) > 0:
            z_vals = roi_points[:, 2]
            self.get_logger().info(
                f"DEBUG roi z distribution: min={z_vals.min():.3f} "
                f"p10={np.percentile(z_vals,10):.3f} "
                f"p50={np.percentile(z_vals,50):.3f} "
                f"p90={np.percentile(z_vals,90):.3f} "
                f"max={z_vals.max():.3f}",
                throttle_duration_sec=2.0,
            )

        # MOD: ROI alone isn't enough to isolate an object from the table
        # surface — the camera looks straight down, so most points inside
        # a per-object ROI box are the flat tabletop, not the object
        # itself (observed: 3627 ROI points for a 9cm cube, barely reduced
        # by clustering, producing a garbage undersized width estimate
        # dominated by table-plane noise). Objects sit on a known table
        # height (~0.815-0.835m); keep only points meaningfully above
        # that, which isolates the raised object.
        TABLE_Z = 0.82
        above_table_mask = roi_points[:, 2] > (TABLE_Z + 0.01)
        roi_points = roi_points[above_table_mask]

        # MOD: temporary diagnostic — compare ROI point count (post
        # table-height filter) vs post-clustering point count.
        self.get_logger().info(
            f"DEBUG roi->cluster: roi_points={len(roi_points)}",
            throttle_duration_sec=2.0,
        )
        object_points = self._cluster_largest(roi_points)
        self.get_logger().info(
            f"DEBUG cluster result: cluster_points={len(object_points)}",
            throttle_duration_sec=2.0,
        )

        # ── Step 3: Isolate the main object ──────────────────────────
        if len(object_points) < self.min_points_for_object:
            return EstimationResult(
                object_detected=False,
                world_name=self.world_name,
                object_name=self.object_name,
                num_roi_points=len(roi_points),
                num_cluster_points=len(object_points),
            )

        # ── Step 4: Estimate bounding dimensions ────────────────────────
        try:
            est_w, est_h, est_d = self._estimate_dimensions(object_points)
        except Exception as e:
            self.get_logger().error(f"Dimension estimation failed: {e}")
            return None

        # ── Step 5: Grasp decision ──────────────────────────────────────
        can_grasp = (est_w <= self.gripper_max_width and
                     est_h <= self.gripper_max_height)

        result = EstimationResult(
            object_detected=True,
            world_name=self.world_name,
            object_name=self.object_name,
            estimated_width=est_w,
            estimated_height=est_h,
            estimated_depth=est_d,
            can_grasp=can_grasp,
            num_roi_points=len(roi_points),
            num_cluster_points=len(object_points),
        )

        # ── Step 6: Ground truth — pose + dimensions ───────────────────
        # export_pose.py now exports width/height/depth from SDF geometry,
        # so dimension-based evaluation is active whenever those fields are
        # non-None. Pose fields are always attached when available.
        gt = self._get_ground_truth()
        if gt is not None:
            result.ground_truth_available = True
            result.gt_pose_x = gt["x"]
            result.gt_pose_y = gt["y"]
            result.gt_pose_z = gt["z"]
            result.gt_roll   = gt["roll"]
            result.gt_pitch  = gt["pitch"]
            result.gt_yaw    = gt["yaw"]

            gt_w = gt["width"]
            gt_h = gt["height"]
            gt_d = gt["depth"]

            if gt_w is not None and gt_h is not None:
                result.gt_dimensions_available = True
                result.gt_width  = gt_w
                result.gt_height = gt_h
                result.gt_depth  = gt_d  # may still be None for some geometries

                result.gt_can_grasp = (gt_w <= self.gripper_max_width and
                                       gt_h <= self.gripper_max_height)
                result.decision_correct = (can_grasp == result.gt_can_grasp)
                result.width_error_m  = abs(est_w - gt_w)
                result.height_error_m = abs(est_h - gt_h)
            else:
                result.gt_dimensions_available = False
        else:
            result.ground_truth_available = False
            result.gt_dimensions_available = False

        self._frames_evaluated += 1
        return result

    # ── Helpers ───────────────────────────────────────────────────────
    def _unpack_cloud(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to Nx3 numpy array."""
        structured = pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if structured.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        arr = np.column_stack(
            (structured["x"], structured["y"], structured["z"])
        ).astype(np.float32, copy=False)
        return arr

    def _filter_roi(self, points: np.ndarray) -> np.ndarray:
        mask = (
            (points[:, 0] >= self.roi_x_range[0]) & (points[:, 0] <= self.roi_x_range[1]) &
            (points[:, 1] >= self.roi_y_range[0]) & (points[:, 1] <= self.roi_y_range[1]) &
            (points[:, 2] >= self.roi_z_range[0]) & (points[:, 2] <= self.roi_z_range[1])
        )
        return points[mask]

    def _cluster_largest(self, points: np.ndarray) -> np.ndarray:
        """Voxel-density clustering to isolate the largest object.

        TODO: this voxel-density + radius-threshold heuristic picks a
        single densest voxel as the object centroid and grabs everything
        within cluster_radius of it. It works for relatively isolated
        objects but will tend to merge nearby objects (or pick the wrong
        cluster) in cluttered/occluded scenes such as world5_mixed_clutter
        and world6_occlusion. Consider replacing with DBSCAN.
        """
        if len(points) == 0:
            return points

        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        unique, counts = np.unique(voxel_indices, axis=0, return_counts=True)

        densest = unique[np.argmax(counts)]
        centroid = densest * self.voxel_size + self.voxel_size / 2.0

        dist = np.linalg.norm(points - centroid, axis=1)
        return points[dist < self.cluster_radius]

    def _estimate_dimensions(self, points: np.ndarray) -> Tuple[float, float, float]:
        """5th-95th percentile bounding box extent — unchanged from original."""
        p5  = np.percentile(points,  5, axis=0)
        p95 = np.percentile(points, 95, axis=0)
        extent = p95 - p5
        return float(extent[0]), float(extent[1]), float(extent[2])

    def _log_result(self, result: EstimationResult):
        if not result.object_detected:
            self.get_logger().info(
                f"[{self.world_name}] No object detected "
                f"(roi_points={result.num_roi_points})",
                throttle_duration_sec=2.0,
            )
            return

        if result.gt_dimensions_available:
            correct_str = "OK" if result.decision_correct else "X"
            gt_str = (
                f"GT W={result.gt_width:.3f}m H={result.gt_height:.3f}m "
                f"err_w={result.width_error_m:.3f}m err_h={result.height_error_m:.3f}m "
                f"[{correct_str}]"
            )
        elif result.ground_truth_available:
            gt_str = (
                f"GT pose x={result.gt_pose_x:.3f} y={result.gt_pose_y:.3f} "
                f"z={result.gt_pose_z:.3f} (no dims)"
            )
        else:
            gt_str = "no ground truth"

        self.get_logger().info(
            f"[{self.world_name}] dims W:{result.estimated_width:.3f}m "
            f"H:{result.estimated_height:.3f}m D:{result.estimated_depth:.3f}m "
            f"-> {'GRASP' if result.can_grasp else 'SKIP'} | {gt_str}"
        )

        if self._frames_evaluated % 50 == 0:
            self.get_logger().info(
                f"Health: received={self._frames_received} "
                f"evaluated={self._frames_evaluated} "
                f"skipped_stale={self._frames_skipped_stale} "
                f"skipped_invalid={self._frames_skipped_invalid}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GraspEstimator()
        rclpy.spin(node)
    except (KeyboardInterrupt, ValueError):
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
            print(f"grasp_estimator failed to initialize: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
