"""
build_training_table.py

Consolidates per-world feature CSVs into a single training table for the
Geometric Relation Classifier.

Expected input:
relation_classifier/data/
    world1_features.csv
    world2_features.csv
    ...

Output:
relation_classifier/data/training_table_consolidated.csv
"""

from pathlib import Path
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

FEATURE_COLUMNS = [
    "boundary_sharpness",
    "minimum_boundary_distance",
    "surface_normal_consistency",
    "relative_height",
    "depth_gradient",
    "overlap_ratio",
    "edge_continuity",
    "occlusion_boundary_score",
]

LABEL_COLUMN = "ground_truth_relation"

EXPECTED_LABELS = {
    "Contact",
    "Support",
    "Separate",
}

# These are optional for now. Once the feature extractor is complete,
# you can make them mandatory if desired.
OPTIONAL_ID_COLUMNS = [
    "world_id",
    "scene_id",
    "object_id_a",
    "object_id_b",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "relation_classifier" / "data"
OUTPUT_FILE = DATA_DIR / "training_table_consolidated.csv"

# ============================================================================
# Loading
# ============================================================================

def load_all_scenes(data_dir: Path) -> pd.DataFrame:

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{data_dir}"
        )

    files = sorted(data_dir.glob("*_features.csv"))

    if not files:
        raise FileNotFoundError(
            f"No *_features.csv files found in:\n{data_dir}"
        )

    print(f"Loaded {len(files)} feature files.\n")

    dfs = [pd.read_csv(file) for file in files]

    combined = pd.concat(dfs, ignore_index=True)

    return combined


# ============================================================================
# Validation
# ============================================================================

def validate_table(df: pd.DataFrame):

    # ------------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------------

    expected_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        c for c in expected_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns:\n{missing_columns}"
        )

    # ------------------------------------------------------------------------
    # Optional identifier columns
    # ------------------------------------------------------------------------

    missing_ids = [
        c for c in OPTIONAL_ID_COLUMNS
        if c not in df.columns
    ]

    if missing_ids:
        print(
            "WARNING: Missing identifier columns "
            "(debugging/per-object analysis may be limited):"
        )
        print(missing_ids)
        print()

    # ------------------------------------------------------------------------
    # Ensure numeric feature columns
    # ------------------------------------------------------------------------

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # ------------------------------------------------------------------------
    # Validate labels
    # ------------------------------------------------------------------------

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    unexpected_labels = (
        set(df[LABEL_COLUMN].unique())
        - EXPECTED_LABELS
    )

    if unexpected_labels:
        raise ValueError(
            f"Unexpected label values found: {unexpected_labels}"
        )

    # ------------------------------------------------------------------------
    # NaN check
    # ------------------------------------------------------------------------

    nan_counts = df[FEATURE_COLUMNS].isna().sum()

    if nan_counts.sum() > 0:
        print("WARNING: NaN values detected:\n")
        print(nan_counts[nan_counts > 0])
        print()

    # ------------------------------------------------------------------------
    # Duplicate rows
    # ------------------------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows:
        print(
            f"WARNING: {duplicate_rows} duplicate rows detected.\n"
        )

    # ------------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------------

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows               : {len(df)}")
    print(f"Feature columns    : {len(FEATURE_COLUMNS)}")
    print(f"Classes            : {df[LABEL_COLUMN].nunique()}")

    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts())

    print("\nLabel distribution (%):")
    print(
        df[LABEL_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():

    # Safe if already exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Uncomment these once to verify paths, then comment again.
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"OUTPUT_FILE  : {OUTPUT_FILE}\n")

    df = load_all_scenes(DATA_DIR)

    validate_table(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved {len(df)} rows to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()