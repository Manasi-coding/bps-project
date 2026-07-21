"""
compare_heuristic_vs_gazebo.py (v2)

Cross-checks SDFGeometryRelationSource heuristic labels (world*_features.csv,
per scene_id / viewpoint) against Gazebo physics contact ground truth
(world*_gazebo_contacts.csv, per world -- poses and thus contact are
viewpoint-invariant, so one contact result is reused across all of that
world's scene_ids).

v2 changes vs the original:
  - joins on (world_id, scene_id, object pair) instead of just (world_id,
    object pair), so each row is traceable to the exact viewpoint that
    produced it -- v1 silently collapsed all viewpoints together, which
    made it impossible to tell which specific row's minimum_boundary_distance
    was pixel-fallback vs metric.
  - carries minimum_boundary_distance, minimum_boundary_distance_is_metric,
    overlap_ratio, relative_height through into the comparison output, so
    the pixel-fallback bug's effect on heuristic accuracy can be measured
    directly instead of reconstructed after the fact.
  - reports a three-way crosstab: heuristic_label x gazebo_contact x
    minimum_boundary_distance_is_metric.

Run from ~/bps-project/relation_classifier/.
"""
import re
import glob
from pathlib import Path
import pandas as pd

SDF_DIR = Path("../worlds_dynamic_with_contact_sensors")
DATA_DIR = Path("data")


def get_label_map(world_id: str) -> dict[int, str]:
    sdf_path = SDF_DIR / f"{world_id}.sdf"
    text = sdf_path.read_text()
    mapping = {}
    for m in re.finditer(r"<model name='(obj_[^']+)'>(.*?)</model>", text, re.DOTALL):
        name, body = m.group(1), m.group(2)
        lbl = re.search(r"<label>(\d+)</label>", body)
        if lbl:
            mapping[int(lbl.group(1))] = name
    return mapping


def base_world_id(scene_id: str) -> str:
    """'world1_baseline__v3' -> 'world1_baseline'; 'world1_baseline' -> itself."""
    return scene_id.split("__v")[0]


flagged_path = DATA_DIR / "flagged_objects.csv"
flagged = set()
if flagged_path.exists():
    fdf = pd.read_csv(flagged_path)
    flagged = set(zip(fdf["world_id"], fdf["object"]))

all_rows = []
label_map_cache: dict[str, dict[int, str]] = {}

for feat_path in sorted(glob.glob(str(DATA_DIR / "world*_features.csv"))):
    df = pd.read_csv(feat_path)
    if df.empty:
        continue

    for _, row in df.iterrows():
        scene_id = row["scene_id"]
        world_id = base_world_id(scene_id)

        if world_id not in label_map_cache:
            label_map_cache[world_id] = get_label_map(world_id)
        label_map = label_map_cache[world_id]

        name_a = label_map.get(int(row["object_id_a"]))
        name_b = label_map.get(int(row["object_id_b"]))
        if name_a is None or name_b is None:
            print(f"[WARN] {scene_id}: unmapped label_id in row "
                  f"({row['object_id_a']}, {row['object_id_b']}) -- skipping")
            continue
        if (world_id, name_a) in flagged or (world_id, name_b) in flagged:
            continue  # excluded: pose may have drifted past threshold during settle

        contacts_path = DATA_DIR / f"{world_id}_gazebo_contacts.csv"
        if not contacts_path.exists():
            gazebo_contact = "no_contacts_file"
        else:
            cdf = pd.read_csv(contacts_path)
            key = frozenset((name_a, name_b))
            match = cdf[cdf.apply(lambda r: frozenset((r["object_a"], r["object_b"])) == key, axis=1)]
            gazebo_contact = match["contact"].iloc[0] if not match.empty else "no_gazebo_row"

        all_rows.append({
            "world_id": world_id,
            "scene_id": scene_id,
            "object_a": name_a,
            "object_b": name_b,
            "heuristic_label": row["ground_truth_relation"],
            "gazebo_contact": gazebo_contact,
            "minimum_boundary_distance": row.get("minimum_boundary_distance"),
            "minimum_boundary_distance_is_metric": row.get("minimum_boundary_distance_is_metric"),
            "overlap_ratio": row.get("overlap_ratio"),
            "relative_height": row.get("relative_height"),
        })

result_df = pd.DataFrame(all_rows)
out_path = DATA_DIR / "heuristic_vs_gazebo_comparison_v2.csv"
result_df.to_csv(out_path, index=False)
print(f"Wrote {len(result_df)} compared rows (per scene_id, not deduped) -> {out_path}\n")

print("=" * 70)
print("PER-ROW (all viewpoints): heuristic_label x gazebo_contact")
print("=" * 70)
print(pd.crosstab(result_df["heuristic_label"], result_df["gazebo_contact"]))

print()
print("=" * 70)
print("THREE-WAY: heuristic_label x gazebo_contact x is_metric")
print("=" * 70)
print(pd.crosstab(
    [result_df["heuristic_label"], result_df["minimum_boundary_distance_is_metric"]],
    result_df["gazebo_contact"],
))

print()
mismatches = result_df[(result_df["heuristic_label"] == "Contact") & (result_df["gazebo_contact"] == 0)]
print(f"Contact/gazebo=0 mismatches: {len(mismatches)} total")
print(f"  of which pixel-fallback (is_metric=False): "
      f"{(mismatches['minimum_boundary_distance_is_metric'] == False).sum()}")
print(f"  of which genuinely metric (is_metric=True): "
      f"{(mismatches['minimum_boundary_distance_is_metric'] == True).sum()}")

print()
print("=" * 70)
print("DEDUPED BY UNIQUE PAIR (world_id, object_a, object_b) -- for a")
print("per-relationship rather than per-viewpoint accuracy read:")
print("=" * 70)
dedup = result_df.drop_duplicates(subset=["world_id", "object_a", "object_b"])
print(pd.crosstab(dedup["heuristic_label"], dedup["gazebo_contact"]))