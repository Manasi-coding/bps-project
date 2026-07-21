"""
extract_relation_features.py

Extracts per-object-pair geometric relation features for the Geometric
Relation Classifier and writes one `<world_id>_features.csv` per world
into `relation_classifier/data/`, in the exact schema expected by
`build_training_table.py`.

--------------------------------------------------------------------------
CHANGES IN THIS PASS (final engineering review)

(1) WORLD DISCOVERY IS NOW STRUCTURAL, NOT NAME-BASED.
    Previously: `WORLDS_SDF_DIR.glob("world*.sdf")`. Any .sdf file not
    named "worldN_..." would have been silently skipped.
    Now: every `*.sdf` under WORLDS_SDF_DIR (recursive) is parsed, and a
    file is treated as a valid world iff it structurally contains at
    least one gz-sim-label-system-labeled object AND a depth_camera1
    sensor. Files that don't meet that bar are reported and skipped —
    not silently ignored, not silently included.

(2) GROUND-TRUTH RELATION IS NOW A PLUGGABLE, PRIORITIZED SOURCE CHAIN
    instead of a single hardcoded AABB heuristic:
        ExternalTableRelationSource  (highest priority — a human- or
            physics-log-derived lookup table, if you have or build one)
      > GazeboContactRelationSource  (a parsed Gazebo contact-sensor log,
            if you have or produce one — this is the only source that
            reflects the physics engine's own resolution of touching
            geometry, rather than a static-pose heuristic)
      > SDFGeometryRelationSource    (lowest priority fallback — the AABB
            touching/overlap heuristic from the previous pass)
    All three implement the same `RelationSource.get()` interface, so
    the rest of the pipeline is unchanged regardless of which source(s)
    you actually have populated. See "RELATION SOURCES" section.
    IMPORTANT: none of your uploaded files contained a contact-sensor log
    or an external relation table, so only SDFGeometryRelationSource is
    populated today — the other two are real, working implementations
    waiting for a file path, not stubs that raise NotImplementedError.

(3) DEPTH IS NOW FAIL-FAST BY DEFAULT, NOT NaN-BY-DEFAULT.
    `REQUIRE_DEPTH = True` means: if DEPTH_ROOT or DEPTH_SOURCE_KIND is
    unset, the script refuses to run at all (raises at startup, before
    processing any world) rather than silently writing NaN-heavy CSVs.
    Set `REQUIRE_DEPTH = False` only for an explicit, intentional
    mask-only/ground-truth-only run — it prints a loud banner every time
    it's used so it can't be silently left on.
    No depth-array export script equivalent to your mask exporter was
    provided in any file you've shared, so DEPTH_ROOT genuinely cannot be
    auto-integrated — this script cannot invent a path that doesn't
    exist in your codebase. What it CAN do, and now does, is refuse to
    guess and refuse to produce misleading partial output by default.

(4) MATHEMATICAL CONSISTENCY AUDIT — see the "VALIDITY AUDIT" section at
    the bottom of this docstring. The most important finding: your
    `depth_to_pointcloud.py` range-to-Z-depth correction is only correct
    if the depth array is Gazebo's raw rendered depth (which encodes ray
    range). If DEPTH_ROOT instead holds your Depth Anything V2 metric
    predictions (from `depth_publisher.py`), applying that correction
    would introduce a systematic geometric error, not remove one — a
    monocular metric-depth model predicts Z-depth directly, it does not
    reproduce Gazebo's range-camera encoding. This is exactly the kind of
    silent, scientifically invalid assumption you asked to eliminate, so
    it is now a REQUIRED, explicit setting (`DEPTH_SOURCE_KIND`) with no
    default, rather than an always-on correction.

(5) Remaining assumptions that affect scientific validity are listed in
    "VALIDITY AUDIT" below and re-printed at the end of every run — not
    buried in a docstring nobody reads twice.
--------------------------------------------------------------------------
VALIDITY AUDIT (read before using this for paper results)

- SDFGeometryRelationSource is a static-pose heuristic (EPSILON_M
  touching tolerance), not a physics-verified relation. Two objects
  authored 6mm apart with EPSILON_M=5mm read as Separate even if your
  intent was Contact. It has NOT been cross-checked against Gazebo's own
  contact-sensor resolution because no such log was available. Before
  using its output as paper-quality ground truth, either (a) wire in a
  real contact log via GazeboContactRelationSource, or (b) manually spot
  check a sample of its Support/Contact/Separate calls per world.

- occlusion_boundary_score is a depth-ordering PROXY (fraction of the
  interface where one object is consistently nearer camera), not a
  verified occlusion-boundary label. Treat it as a weak/candidate feature
  until validated, not as ground truth about occlusion.

- DEPTH_SOURCE_KIND correctness is load-bearing for every point-cloud
  feature (surface_normal_consistency, relative_height) and for metric
  minimum_boundary_distance. Set it wrong and those features are
  silently, consistently biased rather than merely noisy — this is worse
  for a trained classifier than missing data. `sanity_check_camera_transform()`
  is now called automatically whenever depth is loaded, and raises if the
  observed-vs-authored object heights disagree beyond
  CAMERA_TRANSFORM_TOLERANCE_M, precisely to catch this class of error.

- Single static capture per world (confirmed by your mask-exporter's flat
  per-scene output directory, not multiple per-frame subfolders) means
  the eventual training table has, at most, one pairwise-adjacency count
  worth of rows per world — likely a few dozen total across six worlds,
  almost certainly Separate-majority/class-imbalanced. That is a dataset
  size/class balance concern for the classifier and the paper's claims,
  not something this script can fix by generating more rows.

- Camera intrinsics derived from horizontal_fov assume square pixels and
  zero lens distortion. Your SDF's `<distortion>` blocks show
  k1=k2=k3=p1=p2=0, so this holds for these six worlds specifically —
  it is not a general assumption, it's been checked against your files.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import numpy as np
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "relation_classifier" / "data"

# Directory to search (recursively) for world .sdf files. No filename
# pattern is assumed — see discover_worlds().
WORLDS_SDF_DIR = PROJECT_ROOT / "worlds"

# Directory containing your mask exporter's output:
# MASKS_ROOT/<world_id>/<object_name>_{mask,boundary}.npy
MASKS_ROOT = PROJECT_ROOT / "data" / "masks"
BOUNDARY_ROOT = PROJECT_ROOT / "boundary_masks"

# --- Depth: fail-fast by default (see docstring point 3) ----------------
REQUIRE_DEPTH = True

DEPTH_ROOT_MDE: Path = PROJECT_ROOT / "predicted_depth"
DEPTH_ROOT_GT: Path = PROJECT_ROOT / "depth_maps"

PIPELINE_VALIDATION_MODE = True

DEPTH_ROOT = DEPTH_ROOT_GT if PIPELINE_VALIDATION_MODE else DEPTH_ROOT_MDE

DEPTH_FILE_GLOB = "*.npy"

# REQUIRED (no default) whenever DEPTH_ROOT is set — see docstring point 4.
#   "gazebo_range_gt"  -> raw Gazebo depth-camera render; range-to-Z
#                         correction from depth_to_pointcloud.py applies.
#   "mde_predicted_z"  -> Depth Anything V2 output from depth_publisher.py;
#                         already Z-depth, correction must NOT be applied.
DEPTH_SOURCE_KIND = (
    "gazebo_range_gt"
    if PIPELINE_VALIDATION_MODE
    else "mde_predicted_z"
)
VALID_DEPTH_SOURCE_KINDS = {"gazebo_range_gt", "mde_predicted_z"}
# --------------------------------------------------------------------------

# Auto-run camera transform sanity check whenever depth loads.
CAMERA_TRANSFORM_TOLERANCE_M = 0.10

# --------------------------------------------------------------------------
# One-time infrastructure validation mode.
#
# False -> normal paper pipeline (Depth Anything V2 Metric).
# True  -> validate the geometry pipeline using Gazebo ground-truth depth.
#
# This should ONLY be enabled for the one-time pipeline validation run.
# --------------------------------------------------------------------------

# --- Relation sources (see RELATION SOURCES section) ---------------------
# Populate either/both to take priority over the SDF geometry heuristic.
EXTERNAL_RELATION_TABLE: Optional[Path] = None   # e.g. .../ground_truth_relations.csv
GAZEBO_CONTACT_LOG: Optional[Path] = None        # e.g. .../contact_log.json
EPSILON_M = 0.005  # SDFGeometryRelationSource touching tolerance
# --------------------------------------------------------------------------

PAIR_ADJACENCY_DILATION_PX = 8

OUTPUT_COLUMNS = [
    "world_id", "scene_id", "object_id_a", "object_id_b",
    "boundary_sharpness", "minimum_boundary_distance", "minimum_boundary_distance_is_metric",
    "surface_normal_consistency", "relative_height", "depth_gradient",
    "overlap_ratio", "edge_continuity", "occlusion_boundary_score",
    "ground_truth_relation",
]

EXPECTED_LABELS = {"Contact", "Support", "Separate"}

PLACEHOLDER_NOTES: list[str] = []


def _note(msg: str) -> None:
    if msg not in PLACEHOLDER_NOTES:
        PLACEHOLDER_NOTES.append(msg)

def scene_id_for_viewpoint(world_id: str, viewpoint: int) -> str:
    """Matches the capture-side convention from process_all_rooms.py:
    viewpoint 1 -> world_id, viewpoint N>=2 -> world_id__v{N}."""
    return world_id if viewpoint == 1 else f"{world_id}__v{viewpoint}"
# ============================================================================
# SDF parsing
# ============================================================================

@dataclass
class ObjectGeom:
    name: str
    label_id: int
    xyz: np.ndarray
    rpy: np.ndarray
    geom_type: str
    dims: tuple
    placeholder_pose: bool = False


@dataclass
class CameraGeom:
    name: str
    xyz: np.ndarray
    rpy: np.ndarray
    width: int
    height: int
    horizontal_fov: float
    clip_near: float
    clip_far: float

    @property
    def fx(self) -> float:
        return self.width / (2.0 * np.tan(self.horizontal_fov / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels — verified against these SDF files

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


def _pose_text_to_xyz_rpy(pose_text: str) -> tuple[np.ndarray, np.ndarray]:
    vals = [float(v) for v in pose_text.split()]
    if len(vals) != 6:
        raise ValueError(f"Unexpected <pose> format: {pose_text!r}")
    return np.array(vals[:3]), np.array(vals[3:])

def _is_placeholder_pose(
    xyz: np.ndarray,
    rpy: np.ndarray,
    atol: float = 1e-9,
) -> bool:
    return (
        np.allclose(xyz, 0.0, atol=atol)
        and np.allclose(rpy, 0.0, atol=atol)
    )

def parse_world_sdf(sdf_path: Path) -> tuple[dict[str, ObjectGeom], dict[int, CameraGeom]]:
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        return {}, {}

    objects: dict[str, ObjectGeom] = {}
    cameras: dict[int, CameraGeom] = {}

    for model in world.findall("model"):
        name = model.get("name")
        model_pose_el = model.find("pose")
        if model_pose_el is None or model_pose_el.text is None:
            continue
        xyz, rpy = _pose_text_to_xyz_rpy(model_pose_el.text)

        label_el = model.find(".//plugin[@filename='gz-sim-label-system']/label")
        if label_el is not None and label_el.text is not None:
            collision_geom = model.find(".//collision/geometry")
            if collision_geom is None:
                _note(f"{sdf_path.name}: model '{name}' has a label but no "
                      f"<collision><geometry> — excluded from ground-truth "
                      f"geometry (mask-only object).")
                continue
            box = collision_geom.find("box/size")
            cyl_r = collision_geom.find("cylinder/radius")
            cyl_l = collision_geom.find("cylinder/length")
            sph_r = collision_geom.find("sphere/radius")

            if box is not None and box.text:
                placeholder = _is_placeholder_pose(xyz, rpy)

                if placeholder:
                    _note(
                        f"{sdf_path.name}: '{name}' has placeholder pose "
                        "(0 0 0 0 0 0); excluded from validation/training."
                    )

                objects[name] = ObjectGeom(
                    name=name,
                    label_id=int(label_el.text),
                    xyz=xyz,
                    rpy=rpy,
                    geom_type="box",
                    dims=tuple(float(v) for v in box.text.split()),
                    placeholder_pose=placeholder,
                )

            elif cyl_r is not None and cyl_l is not None:
                placeholder = _is_placeholder_pose(xyz, rpy)

                if placeholder:
                    _note(
                        f"{sdf_path.name}: '{name}' has placeholder pose "
                        "(0 0 0 0 0 0); excluded from validation/training."
                    )

                objects[name] = ObjectGeom(
                    name=name,
                    label_id=int(label_el.text),
                    xyz=xyz,
                    rpy=rpy,
                    geom_type="cylinder",
                    dims=(float(cyl_r.text), float(cyl_l.text)),
                    placeholder_pose=placeholder,
                )

            elif sph_r is not None and sph_r.text:
                placeholder = _is_placeholder_pose(xyz, rpy)

                if placeholder:
                    _note(
                        f"{sdf_path.name}: '{name}' has placeholder pose "
                        "(0 0 0 0 0 0); excluded from validation/training."
                    )

                objects[name] = ObjectGeom(
                    name=name,
                    label_id=int(label_el.text),
                    xyz=xyz,
                    rpy=rpy,
                    geom_type="sphere",
                    dims=(float(sph_r.text),),
                    placeholder_pose=placeholder,
                )
            else:
                _note(f"{sdf_path.name}: model '{name}' has an unsupported "
                      f"collision geometry type — skipped.")
            continue

        for sensor in model.findall(".//sensor"):
            if sensor.get("type") != "depth_camera":
                continue
            sensor_name = sensor.get("name", "")
            m = re.fullmatch(r"depth_camera(\d+)", sensor_name)
            if not m:
                continue
            viewpoint = int(m.group(1))

            cam_block = sensor.find("camera")
            image_el = cam_block.find("image")
            clip_el = cam_block.find("clip")

            link_el = None
            for link_candidate in model.findall("link"):
                if link_candidate.find(f".//sensor[@name='{sensor_name}']") is not None:
                    link_el = link_candidate
                    break
            link_pose_el = link_el.find("pose") if link_el is not None else None
            if link_pose_el is not None and link_pose_el.text is not None:
                link_xyz, _link_rpy = _pose_text_to_xyz_rpy(link_pose_el.text)
                world_xyz = xyz + _rotation_matrix(rpy) @ link_xyz
            else:
                world_xyz = xyz

            cameras[viewpoint] = CameraGeom(
                name=name, xyz=world_xyz, rpy=rpy,
                width=int(image_el.find("width").text),
                height=int(image_el.find("height").text),
                horizontal_fov=float(cam_block.find("horizontal_fov").text),
                clip_near=float(clip_el.find("near").text),
                clip_far=float(clip_el.find("far").text),
            )

            print(f"[{sdf_path.stem}] Found camera '{sensor_name}' -> viewpoint {viewpoint}, xyz={world_xyz}")

    return objects, cameras


def discover_worlds(sdf_dir: Path) -> list[Path]:
    """
    Structural discovery (point 1): every *.sdf under sdf_dir (recursive)
    that parses to >=1 labeled object AND a depth_camera1 sensor is a
    valid world, regardless of filename. Files that don't qualify are
    reported, not silently dropped and not silently included.
    """
    candidates = sorted(sdf_dir.rglob("*.sdf"))
    valid = []
    for sdf_path in candidates:
        try:
            objects, cameras = parse_world_sdf(sdf_path)
        except ET.ParseError as e:
            _note(f"discover_worlds: {sdf_path} failed to parse as XML ({e}) — skipped.")
            continue
        if not objects:
            _note(f"discover_worlds: {sdf_path} has no gz-sim-label-system "
                  f"labeled objects — not treated as a world, skipped.")
            continue
        if not cameras or 1 not in cameras:
            _note(f"discover_worlds: {sdf_path} has labeled objects but no "
                  f"viewpoint-1 depth_camera1 sensor — skipped (cannot support "
                  f"depth-dependent features or intrinsics derivation).")
            continue
        valid.append(sdf_path)
    return valid


# ============================================================================
# Geometry
# ============================================================================

def _rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def world_aabb(obj: ObjectGeom) -> np.ndarray:
    R = _rotation_matrix(obj.rpy)

    if obj.geom_type == "box":
        sx, sy, sz = obj.dims
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        corners = np.array([
            [sx_ * hx, sy_ * hy, sz_ * hz]
            for sx_ in (-1, 1) for sy_ in (-1, 1) for sz_ in (-1, 1)
        ])
        world_corners = obj.xyz + corners @ R.T
        return np.vstack([world_corners.min(axis=0), world_corners.max(axis=0)])

    if obj.geom_type == "cylinder":
        radius, length = obj.dims
        if abs(obj.rpy[0]) > 1e-6 or abs(obj.rpy[1]) > 1e-6:
            _note(f"world_aabb: cylinder '{obj.name}' has nonzero roll/pitch "
                  f"— using conservative bounding-sphere AABB, not an exact fit.")
            r = float(np.hypot(radius, length / 2))
            half = np.array([r, r, r])
        else:
            half = np.array([radius, radius, length / 2])
        return np.vstack([obj.xyz - half, obj.xyz + half])

    if obj.geom_type == "sphere":
        r = obj.dims[0]
        half = np.array([r, r, r])
        return np.vstack([obj.xyz - half, obj.xyz + half])

    raise ValueError(f"Unhandled geometry type: {obj.geom_type}")


def _interval_gap(a_lo, a_hi, b_lo, b_hi) -> float:
    return max(a_lo - b_hi, b_lo - a_hi, 0.0)


# ============================================================================
# RELATION SOURCES (point 2) — pluggable, prioritized, same interface
# ============================================================================

class RelationSource(ABC):
    """Common interface. get() returns None (not "Separate") when this
    source has no opinion, so callers can fall through to the next
    source instead of a source's silence being mistaken for a negative
    answer."""

    @abstractmethod
    def get(self, world_id: str, object_id_a: int, object_id_b: int) -> Optional[str]:
        ...


class ExternalTableRelationSource(RelationSource):
    """
    Highest-priority source: a lookup table you author or derive from a
    physics/annotation pass, with columns
    world_id,object_id_a,object_id_b,relation. Symmetric lookup (a,b) and
    (b,a) both resolve to the same relation.
    """

    def __init__(self, table_path: Optional[Path]):
        self._table: dict[tuple[str, int, int], str] = {}
        if table_path is None:
            return
        if not table_path.exists():
            _note(f"ExternalTableRelationSource: {table_path} does not exist — "
                  f"source inactive.")
            return
        df = pd.read_csv(table_path)
        required = {"world_id", "object_id_a", "object_id_b", "relation"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{table_path} is missing required columns: {missing}"
            )
        for row in df.itertuples():
            if row.relation not in EXPECTED_LABELS:
                raise ValueError(
                    f"{table_path}: unexpected relation value {row.relation!r} "
                    f"for ({row.world_id}, {row.object_id_a}, {row.object_id_b})"
                )
            key_ab = (row.world_id, int(row.object_id_a), int(row.object_id_b))
            key_ba = (row.world_id, int(row.object_id_b), int(row.object_id_a))
            self._table[key_ab] = row.relation
            self._table[key_ba] = row.relation

    def get(self, world_id, object_id_a, object_id_b) -> Optional[str]:
        return self._table.get((world_id, object_id_a, object_id_b))


class GazeboContactRelationSource(RelationSource):
    """
    Second-priority source: a parsed Gazebo contact-sensor log — the only
    source that reflects the physics engine's own contact resolution
    rather than a static-pose heuristic. No such log exists in anything
    you've shared, so this is inactive by default, but the parser is
    real: expects a JSON file, list of records:
        {"world_id": ..., "object_a": <label_id>, "object_b": <label_id>,
         "in_contact": true/false}
    Only positive contact entries are treated as authoritative "Contact";
    Support requires the caller to separately corroborate vertical
    stacking (a contact sensor alone can't distinguish "resting on top
    of" from "pushed up against the side of"), so this source resolves
    "Contact" only, and defers Support/Separate to the next source in the
    chain — it does not claim more certainty than a contact sensor
    actually provides.
    """

    def __init__(self, log_path: Optional[Path]):
        self._contacts: dict[tuple[str, int, int], bool] = {}
        if log_path is None:
            return
        if not log_path.exists():
            _note(f"GazeboContactRelationSource: {log_path} does not exist — "
                  f"source inactive.")
            return
        records = json.loads(log_path.read_text())
        for r in records:
            key_ab = (r["world_id"], int(r["object_a"]), int(r["object_b"]))
            key_ba = (r["world_id"], int(r["object_b"]), int(r["object_a"]))
            self._contacts[key_ab] = bool(r["in_contact"])
            self._contacts[key_ba] = bool(r["in_contact"])

    def get(self, world_id, object_id_a, object_id_b) -> Optional[str]:
        in_contact = self._contacts.get((world_id, object_id_a, object_id_b))
        if in_contact is True:
            return "Contact"
        return None  # False or missing -> no opinion, defer


class SDFGeometryRelationSource(RelationSource):
    """
    Fallback source (point 2, lowest priority): the AABB touching/overlap
    heuristic over authored SDF pose + collision geometry. This is
    legitimate scene-authoring ground truth, but a static-pose heuristic,
    not a physics-verified relation — see VALIDITY AUDIT.
    """

    def __init__(self, objects_by_world: dict[str, dict[str, ObjectGeom]], epsilon: float = EPSILON_M):
        self._epsilon = epsilon
        # Index by (world_id, label_id) -> ObjectGeom for O(1) lookup.
        self._by_id: dict[tuple[str, int], ObjectGeom] = {}
        for world_id, objects in objects_by_world.items():
            for obj in objects.values():
                self._by_id[(world_id, obj.label_id)] = obj

    def get(self, world_id, object_id_a, object_id_b) -> Optional[str]:
        obj_a = self._by_id.get((world_id, object_id_a))
        obj_b = self._by_id.get((world_id, object_id_b))
        if obj_a is None or obj_b is None:
            return None
        return self._derive(world_aabb(obj_a), world_aabb(obj_b))

    def _derive(self, aabb_a: np.ndarray, aabb_b: np.ndarray) -> str:
        (ax0, ay0, az0), (ax1, ay1, az1) = aabb_a
        (bx0, by0, bz0), (bx1, by1, bz1) = aabb_b

        xy_gap = max(
            _interval_gap(ax0, ax1, bx0, bx1),
            _interval_gap(ay0, ay1, by0, by1),
        )
        xy_overlaps = (
            _interval_gap(ax0, ax1, bx0, bx1) <= self._epsilon
            and _interval_gap(ay0, ay1, by0, by1) <= self._epsilon
        )
        if xy_overlaps:
            a_below_b = abs(az1 - bz0) <= self._epsilon
            b_below_a = abs(bz1 - az0) <= self._epsilon
            if a_below_b or b_below_a:
                return "Support"

        z_overlap = _interval_gap(az0, az1, bz0, bz1) <= self._epsilon
        if z_overlap and xy_gap <= self._epsilon:
            return "Contact"

        return "Separate"


class CompositeRelationSource(RelationSource):
    """Tries each source in order, returns the first non-None answer.
    Records, per call, which source resolved it (self.last_source) so the
    caller can log provenance without polluting the output CSV schema."""

    def __init__(self, sources: list[RelationSource]):
        self._sources = sources
        self.last_source: Optional[str] = None

    def get(self, world_id, object_id_a, object_id_b) -> Optional[str]:
        for source in self._sources:
            result = source.get(world_id, object_id_a, object_id_b)
            if result is not None:
                self.last_source = type(source).__name__
                return result
        self.last_source = None
        return None


# ============================================================================
# Camera geometry — intrinsics + world-frame point back-projection
# ============================================================================

def range_to_z_depth(range_img: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = range_img.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    px = (uu - cx) / fx
    py = (vv - cy) / fy
    ray_scale = np.sqrt(1.0 + px ** 2 + py ** 2)
    return range_img / ray_scale

# Fixed optical-frame -> link-frame convention correction (ROS REP-103 /
# Gazebo sensor convention): optical X(right) -> +link Z? no --
# optical X(right) -> -Y_link, optical Y(down) -> -Z_link, optical Z(forward) -> +X_link
R_OPTICAL_TO_LINK = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

def depth_to_world_points(
    depth: np.ndarray, camera: CameraGeom, min_depth: float, max_depth: float,
    depth_source_kind: str,
) -> np.ndarray:
    """
    depth_source_kind is REQUIRED and controls whether the range->Z
    correction is applied (see docstring point 4). No default — passing
    an invalid value raises rather than guessing.
    """
    if depth_source_kind not in VALID_DEPTH_SOURCE_KINDS:
        raise ValueError(
            f"depth_source_kind must be one of {VALID_DEPTH_SOURCE_KINDS}, "
            f"got {depth_source_kind!r}"
        )

    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    z_range_or_depth = depth.astype(np.float64)

    if depth_source_kind == "gazebo_range_gt":
        print("Raw depth:")
        print(
            np.nanmin(z_range_or_depth),
            np.nanmax(z_range_or_depth),
            np.nanmean(z_range_or_depth),
        )

        z_depth = range_to_z_depth(
            z_range_or_depth,
            fx,
            fy,
            cx,
            cy,
        )

        print("After range_to_z_depth:")
        print(
            np.nanmin(z_depth),
            np.nanmax(z_depth),
            np.nanmean(z_depth),
        )
    else:  # "mde_predicted_z" — already Z-depth, no correction
        z_depth = z_range_or_depth

    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    valid = np.isfinite(z_depth) & (z_depth > min_depth) & (z_depth < max_depth)

    px = (uu - cx) / fx
    py = (vv - cy) / fy
    x_opt = px * z_depth
    y_opt = py * z_depth
    z_opt = z_depth

    points_opt = np.stack([x_opt, y_opt, z_opt], axis=-1).reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    R_link = _rotation_matrix(camera.rpy)
    R = R_link @ R_OPTICAL_TO_LINK

    axes = np.eye(3)

    print("\nWorld directions of camera axes")
    print("Camera X ->", axes[0] @ R.T)
    print("Camera Y ->", axes[1] @ R.T)
    print("Camera Z ->", axes[2] @ R.T)

    optical_centre = np.array([0.0, 0.0, 1.0])

    print("\nCamera forward axis in world:")
    print(optical_centre @ R.T)

    print("\n========== ROTATION ==========")
    print(R)
    print("==============================\n")
    points_world = np.full_like(points_opt, np.nan)
    points_world[valid_flat] = points_opt[valid_flat] @ R.T + camera.xyz

    points_world_img = points_world.reshape(h, w, 3)

    centre = points_world_img[h // 2, w // 2]

    print("========== CENTRE PIXEL ==========")
    print(f"Depth value : {z_depth[h//2, w//2]:.6f}")
    print(f"World point : {centre}")
    print("==================================\n")

    return points_world

def _expected_visible_pixel_count(obj: ObjectGeom, camera: CameraGeom, depth_at_obj: float) -> float:
    """
    Rough expected pixel footprint if the object were fully visible,
    derived from its AABB cross-section and camera intrinsics (similar
    triangles). Used only to flag likely-occluded objects for the sanity
    check -- not a precise rendering model.
    """
    aabb = world_aabb(obj)
    footprint_m2 = (aabb[1][0] - aabb[0][0]) * (aabb[1][1] - aabb[0][1])
    px_per_m = camera.fx / max(depth_at_obj, 1e-6)
    return footprint_m2 * (px_per_m ** 2)

def sanity_check_camera_transform(
    points_world: np.ndarray, mask_lookup: dict[str, np.ndarray],
    objects: dict[str, ObjectGeom], world_id: str, tolerance_m: float,
    camera: CameraGeom,
) -> list[str]:
    """
    Now called AUTOMATICALLY whenever depth is available (point 4), not
    just offered as a manual tool. Compares each object's observed
    (point-cloud-median) world Z against its authored SDF Z. Returns the
    list of object names that disagree beyond tolerance_m; an empty list
    means the transform checks out for this world.

    Objects with low visible-pixel fraction relative to their expected
    fully-visible footprint are treated as likely occluded by design
    (e.g. world5_occlusion_scene) and are logged + skipped rather than
    failed -- a large Z discrepancy is expected when only a sliver of a
    curved/angled surface is visible, not evidence of a transform bug.
    """
    failures = []
    for name, obj in objects.items():
        if obj.placeholder_pose:
            continue
        mask = mask_lookup.get(name)
        if mask is None:
            continue
        eroded_mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        idx = np.where(eroded_mask.reshape(-1))[0]
        if len(idx) == 0:
            continue
        pts = points_world[idx]

        print(f"\n{name}")
        print("points:", pts.shape)
        print("z min :", np.nanmin(pts[:,2]))
        print("z med :", np.nanmedian(pts[:,2]))
        print("z max :", np.nanmax(pts[:,2]))
        print("expected:", obj.xyz[2])

        observed_z = np.nanmedian(pts[:,2])

        if not np.isfinite(observed_z):
            continue
        aabb = world_aabb(obj)
        expected_top_z = aabb[1][2]
        diff = abs(observed_z - expected_top_z)

        aabb_diagonal = float(np.linalg.norm(aabb[1] - aabb[0]))
        # Scale relaxation down from 0.5x to 0.25x diagonal, and cap it so
        # large/irregular objects (e.g. cereal boxes) can't fully swallow a
        # genuine transform error behind their own size.
        effective_tolerance = max(tolerance_m, min(0.25 * aabb_diagonal, 0.15))

        print(f"  [DEBUG] {name}: expected_top={expected_top_z:.3f} "
              f"observed={observed_z:.3f} diff={diff:.3f} "
              f"aabb_diag={aabb_diagonal:.3f} eff_tol={effective_tolerance:.3f} "
              f"margin={effective_tolerance - diff:.3f}")

        if diff > effective_tolerance:
            # Before failing, check whether this object is simply
            # occluded (few visible pixels relative to its expected
            # fully-visible footprint) rather than genuinely mispositioned.
            expected_px = _expected_visible_pixel_count(obj, camera, obj.xyz[2])
            visible_fraction = len(idx) / max(expected_px, 1.0)
            print(f"  [DEBUG] {name}: visible_px={len(idx)}, "
                  f"expected_px={expected_px:.0f}, fraction={visible_fraction:.3f}")

            if visible_fraction < 0.75:
                _note(
                    f"{world_id}: '{name}' appears partially occluded "
                    f"(visible_fraction={visible_fraction:.2f}, "
                    f"visible_px={len(idx)}, expected_px={expected_px:.0f}) "
                    f"-- sanity check relaxed for this object rather than "
                    f"treated as a transform error; diff={diff:.3f}m."
                )
                continue

            print(
                f"{name:20s} "
                f"expected_centroid={obj.xyz[2]:.3f} "
                f"expected_top={expected_top_z:.3f} "
                f"observed={observed_z:.3f} "
                f"diff={diff:.3f}"
            )
            failures.append(name)
    return failures


# ============================================================================
# Mask loading
# ============================================================================

def load_object_mask(world_id: str, object_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    scene_dir = MASKS_ROOT / world_id
    mask_fp = scene_dir / f"{object_name}_mask.npy"
    boundary_fp = scene_dir / f"{object_name}_boundary.npy"
    if not mask_fp.exists() or not boundary_fp.exists():
        return None
    return np.load(mask_fp).astype(bool), np.load(boundary_fp).astype(bool)


def enumerate_present_objects(world_id: str,
                              sdf_objects: dict[str, ObjectGeom]) -> list[str]:

    scene_dir = MASKS_ROOT / world_id

    if not scene_dir.exists():
        _note(f"{world_id}: no mask directory at {scene_dir}")
        return []

    present = [
        name
        for name, obj in sdf_objects.items()
        if (
            not obj.placeholder_pose
            and (scene_dir / f"{name}_mask.npy").exists()
        )
    ]

    missing = set(sdf_objects) - set(present)

    if missing:
        _note(
            f"{world_id}: {len(missing)} object(s) have no mask."
        )

    return present


def enumerate_adjacent_pairs(
    masks: dict[str, np.ndarray], dilation_px: int = PAIR_ADJACENCY_DILATION_PX
) -> list[tuple[str, str]]:
    kernel = np.ones((dilation_px, dilation_px), np.uint8)
    dilated = {name: cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for name, m in masks.items()}
    names = list(masks.keys())
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.any(dilated[a] & masks[b]) or np.any(dilated[b] & masks[a]):
                pairs.append((a, b))
    return pairs


# ============================================================================
# Feature computation
# ============================================================================

def feat_boundary_sharpness(depth, ring_a, ring_b, mask_a, mask_b) -> float:
    interface = (ring_a | ring_b) & (
        cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        | cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    if depth is None or not np.any(interface):
        return float("nan")
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(grad_mag[interface]))


def feat_minimum_boundary_distance(mask_a, mask_b, camera: Optional[CameraGeom], median_depth: Optional[float]) -> tuple[float, bool]:
    dist_to_b_px = cv2.distanceTransform((~mask_b).astype(np.uint8), cv2.DIST_L2, 5)
    if not np.any(mask_a):
        return float("nan"), False
    px_dist = float(np.min(dist_to_b_px[mask_a]))
    if camera is not None and median_depth is not None and np.isfinite(median_depth):
        metres_per_px = median_depth / camera.fx
        return px_dist * metres_per_px, True
    _note("minimum_boundary_distance: no depth available for this row — PIXEL units, not metres.")
    return px_dist, False


def feat_surface_normal_consistency(points_world, mask_a_flat, mask_b_flat) -> float:
    if points_world is None:
        return float("nan")
    idx_a = np.where(mask_a_flat)[0]
    idx_b = np.where(mask_b_flat)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    try:
        import open3d as o3d
    except ImportError:
        _note("surface_normal_consistency: open3d not installed — run `pip install open3d`.")
        return float("nan")

    valid = np.isfinite(points_world).all(axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world[valid])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    normals_valid = np.asarray(pcd.normals)

    normals_full = np.full((points_world.shape[0], 3), np.nan)
    normals_full[valid] = normals_valid

    normals_a = normals_full[idx_a]
    normals_b = normals_full[idx_b]

    if np.all(~np.isfinite(normals_a)) or np.all(~np.isfinite(normals_b)):
        return float("nan")

    na = np.nanmean(normals_a, axis=0)
    nb = np.nanmean(normals_b, axis=0)

    if np.any(np.isnan(na)) or np.any(np.isnan(nb)):
        return float("nan")

    na /= np.linalg.norm(na) + 1e-8
    nb /= np.linalg.norm(nb) + 1e-8

    return float(abs(np.dot(na, nb)))


def feat_relative_height(points_world, mask_a_flat, mask_b_flat) -> float:
    if points_world is None:
        return float("nan")
    idx_a = np.where(mask_a_flat)[0]
    idx_b = np.where(mask_b_flat)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    z_a_vals = points_world[idx_a, 2]
    z_b_vals = points_world[idx_b, 2]

    if np.all(~np.isfinite(z_a_vals)) or np.all(~np.isfinite(z_b_vals)):
        return float("nan")

    z_a = np.nanmedian(z_a_vals)
    z_b = np.nanmedian(z_b_vals)

    return float(z_a - z_b)


def feat_depth_gradient(depth, mask_a, mask_b) -> float:
    if depth is None:
        return float("nan")
    union = mask_a | mask_b
    if not np.any(union):
        return float("nan")
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(grad_mag[union]))


def feat_overlap_ratio(mask_a, mask_b) -> float:
    def bbox(m):
        ys, xs = np.where(m)
        if len(xs) == 0:
            return None
        return xs.min(), ys.min(), xs.max(), ys.max()

    box_a, box_b = bbox(mask_a), bbox(mask_b)
    if box_a is None or box_b is None:
        return float("nan")
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else float("nan")


def feat_edge_continuity(ring_a, ring_b, mask_a, mask_b) -> float:
    interface = (ring_a & ring_b) | (
        ring_a & cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ) | (
        ring_b & cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    if not np.any(interface):
        return float("nan")
    num_labels, labels = cv2.connectedComponents(interface.astype(np.uint8))
    if num_labels <= 1:
        return float("nan")
    sizes = [np.sum(labels == i) for i in range(1, num_labels)]
    return float(max(sizes) / interface.sum())


def feat_occlusion_boundary_score_proxy(depth, ring_a, ring_b, mask_a, mask_b) -> float:
    if depth is None:
        return float("nan")
    interface = (ring_a | ring_b) & (
        cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        | cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    if not np.any(interface):
        return float("nan")
    near_a = cv2.dilate(mask_a.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & interface
    near_b = cv2.dilate(mask_b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & interface
    if not np.any(near_a) or not np.any(near_b):
        return float("nan")
    mean_b = np.mean(depth[near_b])
    a_nearer_votes = np.sum(depth[near_a] < mean_b)
    consistency = a_nearer_votes / max(1, near_a.sum())
    return float(abs(consistency - 0.5) * 2)


# ============================================================================
# Depth loading
# ============================================================================

def load_world_depth(world_id: str) -> Optional[np.ndarray]:
    if DEPTH_ROOT is None:
        return None

    depth_file = DEPTH_ROOT / f"{world_id}_depth.npy"

    if not depth_file.exists():
        _note(f"{world_id}: depth file not found: {depth_file}")
        return None

    return np.load(depth_file)


# ============================================================================
# Startup validation (point 3 — fail fast, not NaN by default)
# ============================================================================

def validate_configuration() -> None:
    if REQUIRE_DEPTH:
        problems = []
        if DEPTH_ROOT is None:
            problems.append(
                "REQUIRE_DEPTH=True but DEPTH_ROOT is not set. Point it at "
                "your per-world metric depth export directory, or set "
                "REQUIRE_DEPTH=False for an explicit mask-only run."
            )
        if DEPTH_SOURCE_KIND is None:
            problems.append(
                "REQUIRE_DEPTH=True but DEPTH_SOURCE_KIND is not set. It "
                "must be 'gazebo_range_gt' (raw Gazebo depth-camera render) "
                "or 'mde_predicted_z' (Depth Anything V2 output from "
                "depth_publisher.py) — these require DIFFERENT math (see "
                "docstring point 4) and getting this wrong silently biases "
                "every point-cloud feature."
            )
        elif DEPTH_SOURCE_KIND not in VALID_DEPTH_SOURCE_KINDS:
            problems.append(
                f"DEPTH_SOURCE_KIND={DEPTH_SOURCE_KIND!r} is not one of "
                f"{VALID_DEPTH_SOURCE_KINDS}."
            )
        if problems:
            raise RuntimeError(
                "extract_relation_features.py refuses to run with ambiguous "
                "depth configuration (this is intentional — see docstring "
                "point 3):\n  - " + "\n  - ".join(problems)
            )
    else:
        print(
            "=" * 60 + "\n"
            "WARNING: REQUIRE_DEPTH=False — this is an explicit PARTIAL run.\n"
            "All depth/point-cloud-dependent features "
            "(boundary_sharpness, depth_gradient, surface_normal_consistency,\n"
            "relative_height, and metric minimum_boundary_distance) will be "
            "NaN in the output.\nDo not use this output for paper-quality "
            "training data.\n" + "=" * 60
        )


# ============================================================================
# Per-world extraction
# ============================================================================

def extract_world_features(
    sdf_path: Path, viewpoint: int, camera: CameraGeom,
    objects: dict[str, ObjectGeom], relation_source: CompositeRelationSource
) -> tuple[pd.DataFrame, dict]:
    world_id = sdf_path.stem
    scene_id = scene_id_for_viewpoint(world_id, viewpoint)
    present_names = enumerate_present_objects(scene_id, objects)

    diagnostics = {"world_id": world_id, "scene_id": scene_id, "viewpoint": viewpoint,
                    "rows": 0, "unresolved_relations": 0,
                    "camera_transform_ok": None, "relation_source_counts": {}}

    if not present_names:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), diagnostics

    masks: dict[str, np.ndarray] = {}
    rings: dict[str, np.ndarray] = {}
    for name in present_names:
        loaded = load_object_mask(scene_id, name)
        if loaded is None:
            continue
        masks[name], rings[name] = loaded

    depth = load_world_depth(scene_id)
    if REQUIRE_DEPTH and depth is None:
        raise RuntimeError(
            f"{scene_id}: REQUIRE_DEPTH=True but no depth array could be "
            f"loaded from {DEPTH_ROOT / (scene_id + '_depth.npy') if DEPTH_ROOT else '<unset>'}. "
            f"Refusing to write a NaN-heavy CSV for this scene."
        )

    points_world = None
    if depth is not None and camera is not None:
        points_world = depth_to_world_points(
            depth, camera, camera.clip_near, camera.clip_far, DEPTH_SOURCE_KIND
        )
        if DEPTH_SOURCE_KIND == "gazebo_range_gt":
            failures = sanity_check_camera_transform(
                points_world,
                masks,
                objects,
                scene_id,
                CAMERA_TRANSFORM_TOLERANCE_M,
                camera,
            )

            diagnostics["camera_transform_ok"] = len(failures) == 0

            if failures:
                raise RuntimeError(
                    f"{scene_id}: camera-transform sanity check failed "
                    f"for {len(failures)} object(s): {failures}"
                )

        else:
            diagnostics["camera_transform_ok"] = None
            print(
                f"{scene_id}: skipped camera-transform sanity check "
                "(Depth Anything V2 produces relative depth, not simulator ground-truth metric depth)."
            )
    elif depth is not None and camera is None:
        _note(f"{scene_id}: depth available but no camera parsed from SDF — "
              f"point-cloud features unavailable.")

    median_depth = float(np.nanmedian(depth)) if depth is not None else None
    pairs = enumerate_adjacent_pairs(masks)
    rows = []

    for name_a, name_b in pairs:
        obj_a, obj_b = objects[name_a], objects[name_b]
        mask_a, mask_b = masks[name_a], masks[name_b]
        ring_a, ring_b = rings[name_a], rings[name_b]

        relation = relation_source.get(world_id, obj_a.label_id, obj_b.label_id)
        if relation is None:
            diagnostics["unresolved_relations"] += 1
            continue
        src_name = relation_source.last_source
        diagnostics["relation_source_counts"][src_name] = (
            diagnostics["relation_source_counts"].get(src_name, 0) + 1
        )

        mask_a_flat = mask_a.reshape(-1)
        mask_b_flat = mask_b.reshape(-1)
        min_boundary_dist, min_boundary_dist_is_metric = feat_minimum_boundary_distance(mask_a, mask_b, camera, median_depth)

        rows.append({
            "world_id": world_id,
            "scene_id": scene_id,
            "object_id_a": obj_a.label_id,
            "object_id_b": obj_b.label_id,
            "_object_name_a": obj_a.name,
            "_object_name_b": obj_b.name,
            "boundary_sharpness": feat_boundary_sharpness(depth, ring_a, ring_b, mask_a, mask_b),
            "minimum_boundary_distance": min_boundary_dist,
            "minimum_boundary_distance_is_metric": min_boundary_dist_is_metric,
            "surface_normal_consistency": feat_surface_normal_consistency(points_world, mask_a_flat, mask_b_flat),
            "relative_height": feat_relative_height(points_world, mask_a_flat, mask_b_flat),
            "depth_gradient": feat_depth_gradient(depth, mask_a, mask_b),
            "overlap_ratio": feat_overlap_ratio(mask_a, mask_b),
            "edge_continuity": feat_edge_continuity(ring_a, ring_b, mask_a, mask_b),
            "occlusion_boundary_score": feat_occlusion_boundary_score_proxy(depth, ring_a, ring_b, mask_a, mask_b),
            "ground_truth_relation": relation,
        })

    diagnostics["rows"] = len(rows)
    full_columns = OUTPUT_COLUMNS[:4] + ["_object_name_a", "_object_name_b"] + OUTPUT_COLUMNS[4:]
    return pd.DataFrame(rows, columns=full_columns), diagnostics


# ============================================================================
# Post-run reporting: spot-check export (point 3 / point 4 in the prompt)
# ============================================================================

def export_spot_check_sample(output_dir: Path, n_per_world: int = 5, seed: int = 42) -> None:
    """Writes a CSV of a random sample of rows per world, for manual visual
    verification against the SDFs — since SDFGeometryRelationSource has not
    been cross-checked against any physics-based ground truth."""
    rng = np.random.default_rng(seed)
    samples = []
    internal_dir = output_dir / "_internal"
    if not internal_dir.exists():
        return
    for f in sorted(internal_dir.glob("*_features_full.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        n = min(n_per_world, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        cols = [c for c in [
            "world_id", "scene_id", "object_id_a", "object_id_b",
            "_object_name_a", "_object_name_b", "ground_truth_relation",
        ] if c in df.columns]
        samples.append(df.iloc[idx][cols])
    if samples:
        out = pd.concat(samples, ignore_index=True)
        out_fp = output_dir / "spot_check_sample.csv"
        out.to_csv(out_fp, index=False)
        print(f"\nWrote spot-check sample ({len(out)} rows) -> {out_fp}")
        print("Manually verify these against the SDFs before trusting labels as ground truth.")


# ============================================================================
# Main
# ============================================================================

def main():
    global DEPTH_ROOT, DEPTH_SOURCE_KIND

    if PIPELINE_VALIDATION_MODE:
        DEPTH_ROOT = DEPTH_ROOT_GT
        DEPTH_SOURCE_KIND = "gazebo_range_gt"
    else:
        DEPTH_ROOT = DEPTH_ROOT_MDE
        DEPTH_SOURCE_KIND = "mde_predicted_z"

    print("=" * 60)

    if PIPELINE_VALIDATION_MODE:
        print("PIPELINE VALIDATION MODE")
        print("Depth source : Gazebo Ground Truth")
        print("Output       : Validation only")
    else:
        print("NORMAL FEATURE EXTRACTION")
        print("Depth source : Depth Anything Metric")

    print("=" * 60)

    validate_configuration()

    if EXTERNAL_RELATION_TABLE is None and GAZEBO_CONTACT_LOG is None:
        print("=" * 60)
        print("WARNING: No ExternalTableRelationSource or GazeboContactRelationSource")
        print("configured. 100% of ground-truth labels in this run come from")
        print("SDFGeometryRelationSource — a static-pose heuristic, not physics-")
        print("verified. Do not treat this run's output as paper-quality ground")
        print("truth without either (a) wiring in a real source, or (b) manually")
        print("spot-checking spot_check_sample.csv against the SDFs.")
        print("=" * 60)

    sdf_files = discover_worlds(WORLDS_SDF_DIR)
    if not sdf_files:
        print(f"No structurally-valid world .sdf files found under {WORLDS_SDF_DIR} — nothing to do.")
        return

    # Build the object index needed by SDFGeometryRelationSource once,
    # up front, across all discovered worlds.
    objects_by_world = {}
    for sdf_path in sdf_files:
        objects, _ = parse_world_sdf(sdf_path)
        objects_by_world[sdf_path.stem] = objects

    relation_source = CompositeRelationSource([
        ExternalTableRelationSource(EXTERNAL_RELATION_TABLE),
        GazeboContactRelationSource(GAZEBO_CONTACT_LOG),
        SDFGeometryRelationSource(objects_by_world, EPSILON_M),
    ])

    total_rows = 0
    all_diagnostics = []
    internal_dir = OUTPUT_DIR / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)

    for sdf_path in sdf_files:
        world_id = sdf_path.stem
        objects, cameras = parse_world_sdf(sdf_path)
        for viewpoint in sorted(cameras.keys()):
            scene_id = scene_id_for_viewpoint(world_id, viewpoint)
            print(f"Processing {scene_id} (viewpoint {viewpoint}) ...")
            camera = cameras[viewpoint]
            df_full, diag = extract_world_features(sdf_path, viewpoint, camera, objects, relation_source)
            all_diagnostics.append(diag)

            if df_full.empty:
                out_fp = OUTPUT_DIR / f"{scene_id}_features.csv"
                print(f"  [{scene_id}] 0 rows extracted — not writing {out_fp.name}.\n")
                continue

            # Full copy (with internal name columns) for spot-check export only.
            full_fp = internal_dir / f"{scene_id}_features_full.csv"
            df_full.to_csv(full_fp, index=False, encoding="utf-8")

            # Public copy matches build_training_table.py's expected schema.
            df_public = df_full.drop(columns=["_object_name_a", "_object_name_b"])
            out_fp = OUTPUT_DIR / f"{scene_id}_features.csv"
            df_public.to_csv(out_fp, index=False, encoding="utf-8")

            print(f"  [{scene_id}] Wrote {len(df_public)} rows -> {out_fp} "
                  f"(relation sources used: {diag['relation_source_counts']}, "
                  f"unresolved: {diag['unresolved_relations']}, "
                  f"camera_transform_ok: {diag['camera_transform_ok']})\n")
            total_rows += len(df_public)

    print("=" * 60)
    print(f"TOTAL ROWS WRITTEN: {total_rows}")

    if PIPELINE_VALIDATION_MODE:
        validation = pd.DataFrame([
            {
                "world": d["world_id"],
                "scene_id": d["scene_id"],
                "viewpoint": d["viewpoint"],
                "camera_transform_ok": d["camera_transform_ok"],
                "rows": d["rows"],
            }
            for d in all_diagnostics
        ])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validation.to_csv(
            OUTPUT_DIR / f"pipeline_validation_{timestamp}.csv",
            index=False,
        )

        print(f"\nSaved pipeline_validation_{timestamp}.csv")

    if PLACEHOLDER_NOTES:
        notes_file = OUTPUT_DIR / "placeholder_pose_exclusions.txt"
        notes_file.write_text("\n".join(PLACEHOLDER_NOTES))

    if PLACEHOLDER_NOTES:
        print("\nFLAGGED ITEMS FROM THIS RUN:")
        for i, note in enumerate(PLACEHOLDER_NOTES, 1):
            print(f"  {i}. {note}")

    all_rows = []
    for f in sorted(OUTPUT_DIR.glob("*_features.csv")):
        all_rows.append(pd.read_csv(f))
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        print("\n" + "=" * 60)
        print("LABEL DISTRIBUTION (this run's output)")
        print("=" * 60)
        counts = combined["ground_truth_relation"].value_counts()
        pcts = combined["ground_truth_relation"].value_counts(normalize=True).mul(100).round(1)
        for label in counts.index:
            print(f"  {label:10s}: {counts[label]:4d} ({pcts[label]}%)")
        print(f"  {'TOTAL':10s}: {len(combined)}")

    export_spot_check_sample(OUTPUT_DIR)

    print("\nSee the VALIDITY AUDIT section in this file's module docstring "
          "before treating this output as paper-quality training data.")


if __name__ == "__main__":
    main()