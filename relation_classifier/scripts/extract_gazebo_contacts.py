"""
extract_gazebo_contacts.py

INDEPENDENT PHYSICS-BASED GROUND-TRUTH CROSS-CHECK for the Geometric
Relation Classifier. This does NOT replace SDFGeometryRelationSource (the
static-pose AABB heuristic that currently produces all training labels) --
it is a second, independent signal to check that heuristic against.

--------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS / THE STATIC-TO-DYNAMIC FLIP
--------------------------------------------------------------------------
Contact sensors were added to every obj_* model in all 6 worlds in a prior
pass. A live test on world1_baseline.sdf with obj_* left <static>true</static>
confirmed (via `gz topic -l | grep contact` showing nothing) that this
Gazebo Harmonic/DART build prunes collision resolution entirely for
static-static pairs -- no contact topic ever gets advertised, regardless
of whether objects are touching. That is a settled finding; this script
does not re-investigate it.

The fix applied upstream (see flip_static_for_objects.py) was to flip all
obj_* models to <static>false</static> so DART actually resolves their
collisions and contact sensors have something to report. This is not
free: once objects are dynamic, they fall under gravity and *may* settle
to a pose that differs from the SDF-authored one the AABB heuristic's
labels are based on. If an object moves enough, the "ground truth" this
script produces would actually be describing a different scene than the
one SDFGeometryRelationSource labeled -- silently comparing apples to
oranges. That is exactly what the pose-delta measurement below exists to
catch: every object's authored (t=0, pre-physics) pose and settled
(post-settle-period) pose are both recorded, the delta is computed, and
any object drifting past DRIFT_FLAG_THRESHOLD_M is flagged rather than
silently trusted.

--------------------------------------------------------------------------
IMPORTANT CAVEAT -- THIS SCRIPT IS NOT EXECUTION-TESTED
--------------------------------------------------------------------------
This sandbox has no Gazebo install and no network access, so none of the
gz.transport13/gz.msgs10 calls below have been run against a live
gz-sim process. The service/topic names and message field names used
(`/world/<world>/control` + gz.msgs.WorldControl for pause/step,
`/world/<world>/pose/info` + gz.msgs.Pose_V for poses, and the
already-confirmed `.../sensor/<model>_contact_sensor/contact` +
gz.msgs.Contacts for contacts) are the standard Gazebo Harmonic APIs,
but "standard" is not the same as "verified in your build." Run this on
one world first and sanity-check the printed diagnostics before trusting
a full 6-world batch.

--------------------------------------------------------------------------
CONTACT SEMANTICS
--------------------------------------------------------------------------
Each obj_*'s contact sensor publishes gz.msgs.Contacts on its own topic.
A published message with an EMPTY contact list is a legitimate "not
touching right now" reading -- it means the sensor is alive and reporting
zero. That is different from a topic that never publishes anything at
all, which means we have no idea what's happening (sensor not attaching
correctly, plugin not loaded for this model, timing issue, etc.). Per
object:
  - if at least one message was received  -> "observed" (its reports are
    trustworthy for whatever pairs it does/doesn't mention)
  - if zero messages were received within CONTACT_TOPIC_TIMEOUT_S -> the
    object is "silent"; every pair involving it is recorded as contact
    ="unknown", not defaulted to 0.
For a pair (A, B) where both are "observed": contact=1 if either object's
sensor ever reported the other as a contact partner during the settle
window, else contact=0. If either side is "silent" and neither reported
a positive contact, the pair is "unknown".
--------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

WORLDS_DIR = Path("../../worlds_dynamic_with_contact_sensors")
OUTPUT_DIR = Path("../data")

WORLD_FILES = [
    "world1_baseline.sdf",
    "world2_dense_clutter.sdf",
    "world3_thin_objects.sdf",
    "world4_support_scene.sdf",
    "world5_occlusion_scene.sdf",
    "world6_dense_mixed.sdf",
]

LINK_NAME = "link"  # confirmed uniform across all obj_* models in this project

# Confirmed-working topic pattern (do not re-derive).
CONTACT_TOPIC_TEMPLATE = "/world/{world}/model/{model}/link/{link}/sensor/{model}_contact_sensor/contact"
POSE_TOPIC_TEMPLATE = "/world/{world}/pose/info"
CONTROL_SERVICE_TEMPLATE = "/world/{world}/control"

# --- Settle period: TUNABLE, see docstring. Given as sim TIME, converted
# to a step count via each world's own <max_step_size> (all 6 worlds use
# 0.001s in the uploaded files, but this is read per-file rather than
# hardcoded in case that ever changes).
SETTLE_TIME_S = 2.5
DEFAULT_MAX_STEP_SIZE = 0.001  # fallback only if a world's <physics> block is unreadable

# How long to wait for gz-sim to come up and start publishing before we
# give up on a world entirely.
STARTUP_TIMEOUT_S = 15.0

# How long, after the settle step request returns, to keep listening for
# contact messages before deciding a given object's topic is "silent".
CONTACT_TOPIC_TIMEOUT_S = 5.0

# Position delta (metres) beyond which an object's authored vs settled
# pose is flagged as having moved enough to potentially invalidate the
# AABB-heuristic label for any pair involving it. TUNABLE -- 2cm is a
# starting guess, not a validated value; tighten/loosen based on what the
# flagged_objects.csv output actually shows once you've run this for real.
DRIFT_FLAG_THRESHOLD_M = 0.02
DRIFT_FLAG_THRESHOLD_DEG = 15.0

GZ_SIM_BINARY = "gz"


# ============================================================================
# SDF inspection (world name + object list) -- reused pattern from
# extract_relation_features.py's parse_world_sdf, kept minimal here since
# this script only needs names, not full geometry.
# ============================================================================

def get_world_name_and_objects(sdf_path: Path) -> tuple[str, list[str]]:
    text = sdf_path.read_text()
    world_match = re.search(r"<world name='([^']+)'>", text)
    if world_match is None:
        raise RuntimeError(f"{sdf_path.name}: could not find <world name='...'>")
    world_name = world_match.group(1)
    objects = sorted(set(re.findall(r"<model name='(obj_[^']+)'>", text)))
    if not objects:
        raise RuntimeError(f"{sdf_path.name}: no obj_* models found")
    return world_name, objects


def get_max_step_size(sdf_path: Path) -> float:
    text = sdf_path.read_text()
    m = re.search(r"<max_step_size>([\d.eE+-]+)</max_step_size>", text)
    if m is None:
        print(f"  [WARN] {sdf_path.name}: no <max_step_size> found, "
              f"defaulting to {DEFAULT_MAX_STEP_SIZE}s -- verify this matches reality.")
        return DEFAULT_MAX_STEP_SIZE
    return float(m.group(1))


# ============================================================================
# Pose bookkeeping
# ============================================================================

@dataclass
class PoseSample:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


def _extract_pose_from_pose_v(msg, object_names: set[str]) -> dict[str, PoseSample]:
    """msg is a gz.msgs.Pose_V. Returns {object_name: PoseSample} for
    whichever of object_names appear in this message (scene-broadcaster
    pose/info messages should contain every entity, but we only keep the
    ones we asked about)."""
    out: dict[str, PoseSample] = {}
    for p in msg.pose:
        name = p.name
        if name in object_names:
            out[name] = PoseSample(
                x=p.position.x, y=p.position.y, z=p.position.z,
                qx=p.orientation.x, qy=p.orientation.y,
                qz=p.orientation.z, qw=p.orientation.w,
            )
    return out


def _position_delta_m(a: PoseSample, b: PoseSample) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5)


def _orientation_delta_deg(a: PoseSample, b: PoseSample) -> float:
    """Angle between two quaternions, in degrees, via the dot-product
    formula. Clamps for numerical safety before acos."""
    import math
    dot = a.qx * b.qx + a.qy * b.qy + a.qz * b.qz + a.qw * b.qw
    dot = max(-1.0, min(1.0, abs(dot)))  # abs() handles the double-cover (q and -q are the same rotation)
    return math.degrees(2.0 * math.acos(dot))


# ============================================================================
# Contact bookkeeping
# ============================================================================

@dataclass
class ContactTracker:
    object_names: list[str]
    published_any: dict[str, bool] = field(default_factory=dict)
    # frozenset({a,b}) -> (contact_seen: bool, num_points: int)
    pair_hits: dict[frozenset, tuple[bool, int]] = field(default_factory=dict)

    def __post_init__(self):
        for name in self.object_names:
            self.published_any[name] = False

    def record_message(self, owner: str, msg) -> None:
        self.published_any[owner] = True
        for c in msg.contact:
            name1 = _model_from_scoped_name(c.collision1.name)
            name2 = _model_from_scoped_name(c.collision2.name)
            other = name2 if name1 == owner else name1 if name2 == owner else None
            if other is None or other == owner:
                continue  # self-contact (e.g. obj_cable's two segments) -- not a pairwise relation
            if other not in self.object_names:
                continue  # contact against a non-obj_* body (floor/furniture) -- not our concern here
            n_points = len(c.position)
            key = frozenset((owner, other))
            prev_seen, prev_n = self.pair_hits.get(key, (False, 0))
            self.pair_hits[key] = (prev_seen or n_points > 0 or True, max(prev_n, n_points))

    def resolve(self) -> list[dict]:
        rows = []
        for a, b in itertools.combinations(sorted(self.object_names), 2):
            key = frozenset((a, b))
            a_obs = self.published_any[a]
            b_obs = self.published_any[b]
            if key in self.pair_hits:
                _, n_points = self.pair_hits[key]
                contact_val = 1
            elif a_obs and b_obs:
                n_points = 0
                contact_val = 0
            else:
                n_points = 0
                contact_val = "unknown"
            rows.append({
                "object_a": a, "object_b": b,
                "contact": contact_val, "num_contact_points": n_points,
            })
        return rows


def _model_from_scoped_name(scoped_name: str) -> str:
    """gz-sim scoped entity names look like 'obj_bottle::link::collision'
    (or similar '::'-delimited scoping) -- take the first segment. Falls
    back to the raw string if no '::' is present, so an unexpected naming
    scheme doesn't crash the run, just fails the object_names membership
    check downstream (logged as an unrecognized 'other', silently
    skipped) -- verify against real messages on your first run."""
    return scoped_name.split("::")[0]


# ============================================================================
# Per-world processing
# ============================================================================

def process_world(sdf_path: Path):
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.boolean_pb2 import Boolean

    world_id = sdf_path.stem
    world_name, objects = get_world_name_and_objects(sdf_path)
    max_step_size = get_max_step_size(sdf_path)
    settle_steps = max(1, round(SETTLE_TIME_S / max_step_size))

    print(f"\n{'=' * 60}\nProcessing {world_id} (sdf world name: '{world_name}', "
          f"{len(objects)} obj_* models)\n{'=' * 60}")
    print(f"  settle: {SETTLE_TIME_S}s @ max_step_size={max_step_size}s -> {settle_steps} steps")

    # Launch PAUSED so the very first pose we read is the authored,
    # pre-physics pose -- not a `-r` real-time race against our own
    # subscription setup.
    proc = subprocess.Popen([GZ_SIM_BINARY, "sim", "-s", str(sdf_path)])

    try:
        node = Node()
        pose_topic = POSE_TOPIC_TEMPLATE.format(world=world_name)
        control_service = CONTROL_SERVICE_TEMPLATE.format(world=world_name)
        object_set = set(objects)

        latest_poses: dict[str, PoseSample] = {}

        def pose_callback(msg: Pose_V):
            latest_poses.update(_extract_pose_from_pose_v(msg, object_set))

        ok = node.subscribe(Pose_V, pose_topic, pose_callback)
        print(f"  Subscribed to pose topic {pose_topic}: {'ok' if ok else 'FAILED'}")

        tracker = ContactTracker(object_names=objects)

        def make_contact_callback(owner: str):
            def cb(msg: Contacts):
                tracker.record_message(owner, msg)
            return cb

        for name in objects:
            topic = CONTACT_TOPIC_TEMPLATE.format(world=world_name, model=name, link=LINK_NAME)
            ok = node.subscribe(Contacts, topic, make_contact_callback(name))
            if not ok:
                print(f"  [WARN] subscribe FAILED for {name}: {topic}")

        # --- Wait for gz-sim to come up and publish at least one pose batch.
        start = time.time()
        while len(latest_poses) < len(objects) and (time.time() - start) < STARTUP_TIMEOUT_S:
            time.sleep(0.1)
        if len(latest_poses) < len(objects):
            missing = object_set - set(latest_poses.keys())
            print(f"  [WARN] startup timeout: never got a pose for {missing} -- "
                  f"proceeding with what we have, but their pose-delta rows will be missing/NaN.")

        authored_poses = dict(latest_poses)  # snapshot before stepping

        # --- Step physics forward SETTLE_TIME_S worth of steps via the
        # world control service, then re-read poses.
        req = WorldControl()
        req.multi_step = settle_steps
        result, response = node.request(control_service, req, WorldControl, Boolean, int(STARTUP_TIMEOUT_S * 1000))
        if not result or not response.data:
            print(f"  [WARN] world control step request did not confirm success "
                  f"(result={result}, response={getattr(response, 'data', None)}) -- "
                  f"settled poses below may not reflect a full settle period.")

        # Give the pose/contact topics a moment to publish post-settle state.
        time.sleep(CONTACT_TOPIC_TIMEOUT_S)

        settled_poses = dict(latest_poses)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # --- Pose delta rows ---------------------------------------------------
    pose_rows = []
    flagged = []
    for name in objects:
        a = authored_poses.get(name)
        s = settled_poses.get(name)
        if a is None or s is None:
            print(f"  [WARN] {name}: missing authored or settled pose -- "
                  f"recording NaN deltas, treat as unresolved not as zero drift.")
            pose_rows.append({
                "world_id": world_id, "object": name,
                "authored_x": getattr(a, "x", float("nan")), "authored_y": getattr(a, "y", float("nan")),
                "authored_z": getattr(a, "z", float("nan")),
                "settled_x": getattr(s, "x", float("nan")), "settled_y": getattr(s, "y", float("nan")),
                "settled_z": getattr(s, "z", float("nan")),
                "position_delta_m": float("nan"), "orientation_delta_deg": float("nan"),
                "flagged": True,
            })
            flagged.append(name)
            continue

        pos_delta = _position_delta_m(a, s)
        orient_delta = _orientation_delta_deg(a, s)
        is_flagged = pos_delta > DRIFT_FLAG_THRESHOLD_M or orient_delta > DRIFT_FLAG_THRESHOLD_DEG
        if is_flagged:
            flagged.append(name)

        pose_rows.append({
            "world_id": world_id, "object": name,
            "authored_x": a.x, "authored_y": a.y, "authored_z": a.z,
            "settled_x": s.x, "settled_y": s.y, "settled_z": s.z,
            "position_delta_m": pos_delta, "orientation_delta_deg": orient_delta,
            "flagged": is_flagged,
        })

    pose_df = pd.DataFrame(pose_rows)

    # --- Contact rows --------------------------------------------------
    contact_rows = tracker.resolve()
    for row in contact_rows:
        row["world_id"] = world_id
    contact_df = pd.DataFrame(contact_rows, columns=["world_id", "object_a", "object_b", "contact", "num_contact_points"])

    return pose_df, contact_df, flagged


# ============================================================================
# Main
# ============================================================================

def main():
    try:
        import gz.transport13  # noqa: F401
    except ImportError as e:
        print(f"Cannot import gz.transport13: {e}")
        print("This script must run on a machine with Gazebo Harmonic's Python bindings installed.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_flagged: dict[str, list[str]] = {}

    for fname in WORLD_FILES:
        sdf_path = WORLDS_DIR / fname
        if not sdf_path.exists():
            print(f"[SKIP] {fname}: not found at {sdf_path}")
            continue

        world_id = sdf_path.stem
        pose_df, contact_df, flagged = process_world(sdf_path)

        pose_fp = OUTPUT_DIR / f"{world_id}_pose_delta.csv"
        pose_df.to_csv(pose_fp, index=False)
        print(f"  Wrote {len(pose_df)} rows -> {pose_fp}")

        contact_fp = OUTPUT_DIR / f"{world_id}_gazebo_contacts.csv"
        contact_df.to_csv(contact_fp, index=False)
        n_contact = int((contact_df["contact"] == 1).sum())
        n_unknown = int((contact_df["contact"] == "unknown").sum())
        print(f"  Wrote {len(contact_df)} pair rows -> {contact_fp} "
              f"(contact=1: {n_contact}, unknown: {n_unknown})")

        if flagged:
            all_flagged[world_id] = flagged
            print(f"  [FLAGGED] {len(flagged)} object(s) drifted past "
                  f"{DRIFT_FLAG_THRESHOLD_M}m or have missing pose data: {flagged}")

        print(f"  NOTE: this world's pose-delta and contact CSVs apply to ALL of its "
              f"viewpoint scene_ids (viewpoint 1 = '{world_id}', viewpoint N>=2 = "
              f"'{world_id}__vN') -- object poses don't change across viewpoints, only "
              f"the camera does, so downstream consumers should reuse these files rather "
              f"than expect per-viewpoint variants.")

    # --- Cross-world flagged-objects summary --------------------------
    flagged_rows = []
    for world_id, names in all_flagged.items():
        for name in names:
            flagged_rows.append({"world_id": world_id, "object": name})
    if flagged_rows:
        flagged_df = pd.DataFrame(flagged_rows)
        flagged_fp = OUTPUT_DIR / "flagged_objects.csv"
        flagged_df.to_csv(flagged_fp, index=False)
        print(f"\n{'=' * 60}\nFLAGGED OBJECTS SUMMARY -> {flagged_fp}\n{'=' * 60}")
        print(flagged_df.to_string(index=False))
        print(f"\n{len(flagged_rows)} object(s) across {len(all_flagged)} world(s) drifted "
              f"past DRIFT_FLAG_THRESHOLD_M={DRIFT_FLAG_THRESHOLD_M}m during settling (or had "
              f"missing pose data). Any pair involving these objects should be manually "
              f"reviewed -- and probably excluded -- before trusting the AABB-heuristic "
              f"label as still valid for the settled scene.")
    else:
        print(f"\nNo objects exceeded DRIFT_FLAG_THRESHOLD_M={DRIFT_FLAG_THRESHOLD_M}m in any world.")

    print("\nDone. Remember: gazebo_contacts.csv is an INDEPENDENT CROSS-CHECK, not a "
          "replacement for SDFGeometryRelationSource -- reconcile disagreements manually "
          "(see the parent extract_relation_features.py's ExternalTableRelationSource / "
          "GazeboContactRelationSource priority chain for how to wire a reconciled table in).")


if __name__ == "__main__":
    main()
