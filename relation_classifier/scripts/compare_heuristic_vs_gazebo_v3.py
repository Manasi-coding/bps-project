"""
compare_heuristic_vs_gazebo_v3.py

Same join logic as v2 (world_id, scene_id, object pair -> gazebo_contact),
but carries all 8 geometric features through into the comparison output
instead of only 4, so the full feature set can be evaluated against
Gazebo ground truth.

Run from ~/bps-project/relation_classifier/.
"""
import re
import glob
from pathlib import Path
import pandas as pd

SDF_DIR = Path("../worlds_dynamic_with_contact_sensors")
DATA_DIR = Path("data")

FEATURE_COLUMNS = [
    "boundary_sharpness",
    "minimum_boundary_distance",
    "minimum_boundary_distance_is_metric",
    "surface_normal_consistency",
    "relative_height",
    "depth_gradient",
    "overlap_ratio",
    "edge_continuity",
    "occlusion_boundary_score",
]


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
            continue

        contacts_path = DATA_DIR / f"{world_id}_gazebo_contacts.csv"
        if not contacts_path.exists():
            gazebo_contact = "no_contacts_file"
        else:
            cdf = pd.read_csv(contacts_path)
            key = frozenset((name_a, name_b))
            match = cdf[cdf.apply(lambda r: frozenset((r["object_a"], r["object_b"])) == key, axis=1)]
            gazebo_contact = match["contact"].iloc[0] if not match.empty else "no_gazebo_row"

        out_row = {
            "world_id": world_id,
            "scene_id": scene_id,
            "object_a": name_a,
            "object_b": name_b,
            "heuristic_label": row["ground_truth_relation"],
            "gazebo_contact": gazebo_contact,
        }
        for col in FEATURE_COLUMNS:
            out_row[col] = row.get(col)
        all_rows.append(out_row)

result_df = pd.DataFrame(all_rows)
out_path = DATA_DIR / "heuristic_vs_gazebo_comparison_v3.csv"
result_df.to_csv(out_path, index=False)
print(f"Wrote {len(result_df)} compared rows (per scene_id, not deduped) -> {out_path}\n")

print("=" * 70)
print("PER-ROW (all viewpoints): heuristic_label x gazebo_contact")
print("=" * 70)
print(pd.crosstab(result_df["heuristic_label"], result_df["gazebo_contact"]))

print()
print("Non-null counts per feature column:")
print(result_df[FEATURE_COLUMNS].notna().sum())
