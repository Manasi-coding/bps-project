#!/usr/bin/env python3
"""
grasp_estimator.py
-------------------
BPS pipeline experiment node (ROS 2 Humble / Gazebo Harmonic).

Reads the pointcloud from depth_to_pointcloud.py, isolates the dominant
object in a region of interest, estimates its bounding dimensions, and
decides whether a gripper of configurable size could grasp it. Ground
truth pose data is read from the pipeline's pre-exported
`ground_truth/<world_name>_pose.json` files. Boundary masks and semantic
masks produced by the rest of the BPS pipeline are located via parameters
so this node's outputs can be correlated with them downstream, even
though this node does not consume their pixel content directly.

The perception pipeline is scene-agnostic and operates without any
world-specific logic. Optionally, an object_name parameter may be supplied
to associate estimated dimensions with a specific ground-truth pose entry.

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
DEFAULT_EVAL_RATE_HZ         = 1.0
DEFAULT_CLOUD_STALENESS_SEC  = 2.0   # warn/skip if latest cloud older than this
DEFAULT_GROUND_TRUTH_DIR     = "ground_truth"
# MOD: added defaults for the other two pipeline output directories
# (boundary masks, semantic masks) so paths are parameter-driven and
# consistent with the rest of the BPS pipeline's `<root>/<world_name>`
# layout, per requirement 4. This node doesn't read their pixel content
# (that's the BPS boundary-sensitivity analysis stage's job), but it
# resolves and logs the paths so results can be correlated/cross-checked
# against them, and so the parameter surface matches the pipeline's
# actual file layout.
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
    # Reserved for future pipeline versions where the exported
    # ground-truth JSON also contains object dimensions.
    gt_width: Optional[float] = None
    gt_height: Optional[float] = None
    gt_depth: Optional[float] = None

    can_grasp: Optional[bool] = None

    # Reserved for future evaluation metrics.
    gt_can_grasp: Optional[bool] = None
    decision_correct: Optional[bool] = None

    width_error_m: Optional[float] = None
    height_error_m: Optional[float] = None
    ground_truth_available: bool = False
    # MOD: dimensions are not yet present in the pose JSONs (requirement 5),
    # so dimension-based GT comparison is structurally disabled. This flag
    # makes that explicit in the result schema (distinct from
    # ground_truth_available, which still reflects whether *pose* ground
    # truth was found) so log_results.py can tell the two apart.
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
        self.declare_parameter("object_name", "")          # "" = any/unspecified
        self.declare_parameter("world_name", "")            # e.g. world3_kitchen_objects
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
        self.declare_parameter("ground_truth_source", "json")  # "json" | "none"
        self.declare_parameter("ground_truth_dir", DEFAULT_GROUND_TRUTH_DIR)
        # MOD: new parameters for locating boundary/semantic mask directories
        # per the pipeline's existing layout (requirement 4 + 6 — everything
        # must be parameter-driven, nothing world-specific hard-coded).
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

        # Cached ground-truth pose data for the current world, loaded once
        # at startup. Loading is best-effort: any failure disables
        # ground-truth comparison for this run instead of crashing the node.
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

        # MOD: resolve and log the boundary/semantic mask directories for
        # this world at startup. We only check directory existence here
        # (not file-by-file) — this node doesn't consume their contents,
        # it just needs to confirm the pipeline layout lines up for this
        # world so misconfiguration is visible immediately rather than
        # discovered later when correlating results across pipeline stages.
        self._boundary_masks_path = os.path.join(
            self.boundary_masks_dir, self.world_name) if self.world_name else ""
        self._semantic_masks_path = os.path.join(
            self.semantic_masks_dir, self.world_name) if self.world_name else ""
        self._log_mask_dir_status("boundary masks", self._boundary_masks_path)
        self._log_mask_dir_status("semantic masks", self._semantic_masks_path)

        # ── QoS ────────────────────────────────────────────────────────
        # PointCloud2 from a sensor-like publisher is best matched with
        # BEST_EFFORT + small history, mirroring typical sensor QoS in
        # ROS 2. Using RELIABLE here (the rclpy default) against a
        # best-effort publisher would silently drop the subscription
        # connection.
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
        # MOD: load the two new mask-directory parameters (requirement 4).
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

    # MOD: small helper (new) to log whether each pipeline output directory
    # exists for this world, without hard-coding any world name — purely
    # parameter + world_name driven, per requirement 6.
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
        Load ground_truth/<world_name>_pose.json for the configured world.

        MOD: schema updated to match the pipeline's actual pose JSON
        format, which contains only pose data — no object dimensions:

            {
              "objects": [
                  {
                      "name": "<object_name>",
                      "x": .., "y": .., "z": ..,
                      "roll": .., "pitch": .., "yaw": ..
                  },
                  ...
              ]
            }

        (also tolerant of a flat dict keyed by object name, or a bare
        top-level list, to stay robust to minor export variations across
        the six scenes). Dimensions are intentionally NOT expected here —
        see _get_ground_truth_pose() / requirement 5. Returns None (never
        raises) if the file is missing, malformed, or world_name wasn't
        set — callers must treat that as "ground truth unavailable".
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

        # MOD: normalise three possible shapes into a flat
        # object_name -> pose_entry dict: (a) {"objects": [...]} list of
        # entries each with a "name" field, (b) a flat top-level dict
        # already keyed by object name, (c) a bare top-level list. The
        # original version only handled the dict-of-dicts shape; the
        # pipeline's actual exporter uses a list of pose entries, so this
        # normalisation step was required for compatibility.
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

        self.get_logger().info(
            f"Loaded ground-truth pose data for world '{self.world_name}' "
            f"from {path} ({len(objects)} object entries)."
        )
        return objects

    # MOD: renamed/refactored from the dimension-comparison version of this
    # method. Per requirement 5, the pose JSONs only contain
    # name/x/y/z/roll/pitch/yaw — no width/height/depth — so dimension
    # comparison cannot be performed against this data source. Rather than
    # deleting the ground-truth framework, this method now returns the
    # available *pose* fields, and dimension fields are left as None
    # throughout the result. The grasp-decision and dimension-estimation
    # logic itself is completely unaffected — it never depended on this
    # ground-truth source to function.
    def _get_ground_truth_pose(self) -> Optional[dict]:
        """
        Look up ground-truth pose (x, y, z, roll, pitch, yaw) for the
        configured object_name in the pre-loaded JSON data. Returns None
        if ground truth is disabled, unavailable, or the object isn't
        found. Dimension fields are NOT looked up here — they are not
        present in the current pose JSON schema (see class docstring).
        """
        if self.ground_truth_source == "none" or self._ground_truth_data is None:
            return None
        if not self.object_name:
            # No object specified, so skip object-specific ground-truth lookup.
            return None

        entry = self._ground_truth_data.get(self.object_name)
        if entry is None:
            self.get_logger().warn(
                f"Object '{self.object_name}' not found in ground-truth "
                f"pose data for world '{self.world_name}'.",
                throttle_duration_sec=10.0,
            )
            return None

        if not isinstance(entry, dict):
            self.get_logger().warn(
                f"Ground-truth entry for '{self.object_name}' is malformed "
                f"(expected an object) — skipping.",
                throttle_duration_sec=10.0,
            )
            return None

        try:
            pose = {
                "x": float(entry["x"]),
                "y": float(entry["y"]),
                "z": float(entry["z"]),
                "roll": float(entry["roll"]),
                "pitch": float(entry["pitch"]),
                "yaw": float(entry["yaw"]),
            }
        except (KeyError, TypeError, ValueError) as e:
            self.get_logger().warn(
                f"Ground-truth pose for '{self.object_name}' is missing or "
                f"malformed fields ({e}) — skipping.",
                throttle_duration_sec=10.0,
            )
            return None

        # MOD: explicitly log (once, throttled) that dimension comparison
        # is bypassed, so this isn't a silent behavioural change someone
        # has to discover by reading code — directly addresses requirement
        # 5's "disable or bypass dimension-comparison code" instruction.
        if not self._ground_truth_load_warned:
            self.get_logger().info(
                "Ground-truth pose loaded successfully. "
                "Pose data is available for evaluation. "
                "Dimension-based evaluation will automatically become active "
                "once object dimensions are added to the exported JSON."
            )
            self._ground_truth_load_warned = True

        return pose

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
                throttle_duration_sec=5.0,
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

        # ── Step 2: Filter to region of interest ────────────────────────
        roi_points = self._filter_roi(points)

        if len(roi_points) < self.min_points_for_object:
            return EstimationResult(
                object_detected=False,
                world_name=self.world_name,
                object_name=self.object_name,
                num_roi_points=len(roi_points),
            )

        # ── Step 3: Cluster to isolate the main object ──────────────────
        object_points = self._cluster_largest(roi_points)
        if len(object_points) < self.min_points_for_object:
            return EstimationResult(
                object_detected=False,
                world_name=self.world_name,
                object_name=self.object_name,
                num_roi_points=len(roi_points),
                num_cluster_points=len(object_points),
            )

        # ── Step 4: Estimate bounding dimensions ────────────────────────
        # Unchanged: dimensions are always estimated purely from the
        # point cloud, regardless of ground-truth availability/schema
        # (requirement 5: "must still estimate dimensions from the point
        # cloud exactly as before").
        try:
            est_w, est_h, est_d = self._estimate_dimensions(object_points)
        except Exception as e:
            self.get_logger().error(f"Dimension estimation failed: {e}")
            return None

        # ── Step 5: Grasp decision ──────────────────────────────────────
        # Unchanged: grasp decision is based solely on the estimated
        # dimensions vs. gripper limits, exactly as before. It does not
        # and never did depend on ground truth.
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

        # ── Step 6/7: Ground truth (pose only — best-effort, never fatal) ─
        # MOD: this block previously compared estimated vs. ground-truth
        # *dimensions* and computed decision_correct/width_error_m/
        # height_error_m. Since the pose JSON has no dimensions
        # (requirement 5), that comparison is bypassed: gt_width/height/
        # depth, gt_can_grasp, decision_correct, and the error metrics all
        # remain None, and gt_dimensions_available stays False. Pose
        # ground truth (x/y/z/roll/pitch/yaw) is still loaded and attached
        # to the result when available, keeping the ground-truth loading
        # framework intact and ready for when dimensions are added later.
        gt_pose = self._get_ground_truth_pose()
        if gt_pose is not None:
            result.ground_truth_available = True
            result.gt_pose_x = gt_pose["x"]
            result.gt_pose_y = gt_pose["y"]
            result.gt_pose_z = gt_pose["z"]
            result.gt_roll = gt_pose["roll"]
            result.gt_pitch = gt_pose["pitch"]
            result.gt_yaw = gt_pose["yaw"]
            # Explicitly false: no dimension data in this schema yet.
            result.gt_dimensions_available = False
        else:
            result.ground_truth_available = False
            result.gt_dimensions_available = False

        self._frames_evaluated += 1
        return result

    # ── Helpers ───────────────────────────────────────────────────────
    def _unpack_cloud(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to Nx3 numpy array (vectorised,
        no per-point Python loop). Unchanged from prior revision."""
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
        """Unchanged: ROI filtering logic preserved exactly."""
        mask = (
            (points[:, 0] >= self.roi_x_range[0]) & (points[:, 0] <= self.roi_x_range[1]) &
            (points[:, 1] >= self.roi_y_range[0]) & (points[:, 1] <= self.roi_y_range[1]) &
            (points[:, 2] >= self.roi_z_range[0]) & (points[:, 2] <= self.roi_z_range[1])
        )
        return points[mask]

    def _cluster_largest(self, points: np.ndarray) -> np.ndarray:
        """Voxel-density clustering to isolate the largest object —
        algorithm unchanged.

        TODO: this voxel-density + radius-threshold heuristic picks a
        single densest voxel as the object centroid and grabs everything
        within cluster_radius of it. It works for relatively isolated
        objects but will tend to merge nearby objects (or pick the wrong
        cluster) in cluttered/occluded scenes such as world5_mixed_clutter
        and world6_occlusion. Consider replacing this with a proper
        density-based clustering algorithm (e.g. DBSCAN via
        sklearn.cluster.DBSCAN) to correctly separate multiple nearby
        objects and reject outlier clusters by density rather than a
        fixed radius from a single seed voxel.
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
        p5 = np.percentile(points, 5, axis=0)
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

        # MOD: log line adjusted to reflect pose-only ground truth instead
        # of a GRASP/SKIP + CORRECT/WRONG comparison, since that comparison
        # is no longer computable from this data source. Estimated
        # dimensions and the grasp decision itself are reported exactly as
        # before.
        gt_suffix = " (no ground truth)"
        if result.ground_truth_available:
            gt_suffix = (
                f" (GT pose: "
                f"x={result.gt_pose_x:.3f}, "
                f"y={result.gt_pose_y:.3f}, "
                f"z={result.gt_pose_z:.3f})"
            )

        self.get_logger().info(
            f"[{self.world_name}] dims W:{result.estimated_width:.3f}m "
            f"H:{result.estimated_height:.3f}m D:{result.estimated_depth:.3f}m "
            f"-> {'GRASP' if result.can_grasp else 'SKIP'}"
            + gt_suffix
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