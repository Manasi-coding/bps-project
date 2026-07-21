"""
scripts/train_relation_classifier.py

Train and evaluate multi-feature classifiers (logistic regression and/or
decision tree) to predict Gazebo-verified contact between object pairs,
using geometric features derived from depth/point cloud data. Compares
learned model performance against the existing rule-based heuristic
baseline (heuristic_label) on the same held-out test split.

Uses group-aware splitting (grouped by unique object pair) so that
multiple camera viewpoints of the same physical pair never span both
train and test sets.

Example:
    python scripts/train_relation_classifier.py --model both
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS: list[str] = [
    "boundary_sharpness",
    "minimum_boundary_distance",
    "surface_normal_consistency",
    "relative_height",
    "depth_gradient",
    "overlap_ratio",
    "edge_continuity",
    "occlusion_boundary_score",
]
LABEL_COLUMN: str = "gazebo_contact"
HEURISTIC_COLUMN: str = "heuristic_label"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train multi-feature classifiers on Gazebo contact ground truth "
            "and compare against the rule-based heuristic baseline."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_data_metric_only.csv",
        help="Path to the training data CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "tree", "both"],
        default="both",
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data (by group) to hold out for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory to save trained model(s), scaler, and summary.",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load the training CSV and drop rows with NaNs in required columns."""
    df = pd.read_csv(data_path)

    required_columns = FEATURE_COLUMNS + [
        LABEL_COLUMN,
        HEURISTIC_COLUMN,
        "world_id",
        "object_a",
        "object_b",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in CSV: {missing_columns}")

    n_before = len(df)
    df_clean = df.dropna(subset=required_columns).reset_index(drop=True)
    n_dropped = n_before - len(df_clean)
    print(f"Loaded {n_before} rows; dropped {n_dropped} rows with NaNs in required columns.")
    print(f"Remaining rows: {len(df_clean)}")

    return df_clean


def encode_heuristic(series: pd.Series) -> np.ndarray:
    """Map heuristic_label strings to 0/1 ints (Contact/Support -> 1, Separate -> 0)."""
    mapping = {"Contact": 1, "Separate": 0, "Support": 1}
    unmapped = set(series.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unexpected heuristic_label values: {unmapped}")
    return series.map(mapping).to_numpy()


def report_to_text(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Build a text block containing a classification report for a model."""
    report = classification_report(
        y_true, y_pred, target_names=["Separate", "Contact"], digits=3
    )
    return f"=== {name} ===\n{report}\n"


def print_and_collect_feature_importance(
    name: str, feature_names: list[str], importances: np.ndarray
) -> str:
    """Print and format feature importances/coefficients sorted by magnitude."""
    order = np.argsort(-np.abs(importances))
    lines = [f"Feature importance for {name} (sorted by magnitude):"]
    for idx in order:
        lines.append(f"  {feature_names[idx]:<28s} {importances[idx]: .4f}")
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def train_logistic(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_tree(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> DecisionTreeClassifier:
    """Train a depth-limited decision tree classifier."""
    model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Run the full train/evaluate/compare/save pipeline."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy().astype(int)
    heuristic_pred_all = encode_heuristic(df[HEURISTIC_COLUMN])

    # Group by unique object pair (world_id, object_a, object_b) so that
    # multiple viewpoints of the same pair never span train and test.
    groups = (
        df["world_id"].astype(str) + "_" + df["object_a"] + "_" + df["object_b"]
    ).to_numpy()

    indices = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    idx_train, idx_test = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    n_train_pairs = len(set(groups[idx_train]))
    n_test_pairs = len(set(groups[idx_test]))
    overlap = set(groups[idx_train]) & set(groups[idx_test])
    print(f"Train pairs: {n_train_pairs}, Test pairs: {n_test_pairs}, Overlap: {len(overlap)}")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    summary_blocks: list[str] = []
    summary_blocks.append(f"Data source: {args.data}\n")
    summary_blocks.append(
        f"Train pairs: {n_train_pairs}, Test pairs: {n_test_pairs}, Overlap: {len(overlap)}\n"
    )
    summary_blocks.append(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}\n\n")

    models_to_run = ["logistic", "tree"] if args.model == "both" else [args.model]
    trained_models: dict[str, Any] = {}

    if "logistic" in models_to_run:
        log_model = train_logistic(X_train_scaled, y_train, args.random_state)
        y_pred_log = log_model.predict(X_test_scaled)

        block = report_to_text("Logistic Regression", y_test, y_pred_log)
        print(block)
        summary_blocks.append(block)

        coef_block = print_and_collect_feature_importance(
            "Logistic Regression", FEATURE_COLUMNS, log_model.coef_[0]
        )
        summary_blocks.append(coef_block)

        trained_models["logistic"] = log_model

    if "tree" in models_to_run:
        tree_model = train_tree(X_train, y_train, args.random_state)
        y_pred_tree = tree_model.predict(X_test)

        block = report_to_text("Decision Tree", y_test, y_pred_tree)
        print(block)
        summary_blocks.append(block)

        importance_block = print_and_collect_feature_importance(
            "Decision Tree", FEATURE_COLUMNS, tree_model.feature_importances_
        )
        summary_blocks.append(importance_block)

        trained_models["tree"] = tree_model

    # Heuristic baseline evaluated on the same test split.
    heuristic_pred_test = heuristic_pred_all[idx_test]
    heuristic_block = report_to_text("Heuristic Baseline (rule-based)", y_test, heuristic_pred_test)
    print(heuristic_block)
    summary_blocks.append(heuristic_block)

    # Group-aware cross-validation (GroupKFold) for a more robust estimate
    # than a single split, still respecting pair grouping.
    print("=== Group-aware 5-fold Cross-Validation (F1) ===")
    cv_summary = ["=== Group-aware 5-fold Cross-Validation (F1) ===\n"]
    gkf = GroupKFold(n_splits=5)

    if "logistic" in models_to_run:
        lr_pipeline = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.random_state)
        )
        lr_scores = cross_val_score(lr_pipeline, X, y, cv=gkf, groups=groups, scoring="f1")
        line = f"Logistic: mean={lr_scores.mean():.3f}, std={lr_scores.std():.3f}, folds={np.round(lr_scores, 3)}"
        print(line)
        cv_summary.append(line + "\n")

    if "tree" in models_to_run:
        tree_cv = DecisionTreeClassifier(max_depth=4, random_state=args.random_state)
        tree_scores = cross_val_score(tree_cv, X, y, cv=gkf, groups=groups, scoring="f1")
        line = f"Tree: mean={tree_scores.mean():.3f}, std={tree_scores.std():.3f}, folds={np.round(tree_scores, 3)}"
        print(line)
        cv_summary.append(line + "\n")

    summary_blocks.append("".join(cv_summary))

    # Save models and scaler.
    for model_name, model_obj in trained_models.items():
        model_path = output_dir / f"{model_name}_model.joblib"
        joblib.dump(model_obj, model_path)
        print(f"Saved {model_name} model to {model_path}")

    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("".join(summary_blocks))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
