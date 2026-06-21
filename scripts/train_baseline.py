# train_baseline.py
"""Baseline ASD classifier: Model 1 (all), Model 2 (motion), Model 3 (bbox)."""

import sys
from pathlib import Path

import os
os.environ["MPLBACKEND"] = "Agg"
# Safely import matplotlib; fall back if unavailable
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    print(f"Warning: matplotlib import failed ({e}); confusion matrix will not be saved.")
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception as e:
    sns = None
    print(f"Warning: seaborn import failed ({e}); seaborn plots will be skipped.")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(r"c:/asd_project/outputs/features/labeled_features.csv")
REPORTS_DIR = Path(r"c:/asd_project/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["unique_video_id", "video_id", "label", "split", "dataset_path", "subject_id"]

# Model 2: Motion Features Only
MOTION_FEATURES = [
    "mean_speed",
    "max_speed",
    "std_speed",
    "total_distance",
    "motion_burst_count",
    "activity_ratio",
]

# Model 3: BBox Features Only
BBOX_FEATURES = [
    "mean_width",
    "std_width",
    "mean_height",
    "std_height",
    "mean_area",
    "std_area",
    "min_area",
    "max_area",
]


def positive_class_proba(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        if 1 in model.classes_:
            return proba[:, 0]
        return np.zeros(len(X))
    idx = list(model.classes_).index(1)
    return proba[:, idx]


def run_cross_validation(rf, X_train_scaled, y_train) -> str:
    """Run 5-fold CV and return summary text for the report file."""
    # Determine the smallest class count in the training set
    class_counts = np.bincount(y_train)
    smallest_class = class_counts[class_counts > 0].min()
    n_splits = min(5, smallest_class)
    if n_splits < 2:
        print("Skipping cross-validation (need at least 2 samples per class in train).\n")
        return ""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="accuracy")
    cv_precision = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="precision")
    cv_recall = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="recall")
    cv_f1 = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="f1")
    cv_roc_auc = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring="roc_auc")

    print("--- Cross-validation (5-fold) per-fold scores ---")
    print("CV Accuracy:", cv_accuracy)
    print("CV Precision:", cv_precision)
    print("CV Recall:", cv_recall)
    print("CV F1:", cv_f1)
    print("CV ROC-AUC:", cv_roc_auc)

    print("\n--- Cross-validation (5-fold) metrics (mean) ---")
    print(f"Mean Accuracy : {cv_accuracy.mean():.4f}")
    print(f"Mean Precision: {cv_precision.mean():.4f}")
    print(f"Mean Recall   : {cv_recall.mean():.4f}")
    print(f"Mean F1 Score : {cv_f1.mean():.4f}")
    print(f"Mean ROC-AUC  : {cv_roc_auc.mean():.4f}\n")

    return (
        f"Mean Accuracy : {cv_accuracy.mean():.4f}\n"
        f"Mean Precision: {cv_precision.mean():.4f}\n"
        f"Mean Recall   : {cv_recall.mean():.4f}\n"
        f"Mean F1 Score : {cv_f1.mean():.4f}\n"
        f"Mean ROC-AUC  : {cv_roc_auc.mean():.4f}\n"
    )


def train_model(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Train Random Forest on a chosen feature subset and write reports."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"{model_name}: missing columns in labeled_features.csv: {missing}")

    X = df[feature_cols]
    y = df["label"].astype(int)

    train_mask = df["split"].str.lower() == "train"
    test_mask = df["split"].str.lower() == "test"

    X_train, X_test = X[train_mask].reset_index(drop=True), X[test_mask].reset_index(drop=True)
    y_train, y_test = y[train_mask].reset_index(drop=True), y[test_mask].reset_index(drop=True)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:")
    print(pd.Series(y_train).value_counts())
    print("y_test distribution:")
    print(pd.Series(y_test).value_counts())

    print("=" * 60)
    print(model_name)
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print("=" * 60)
    print(f"Training set: {len(X_train)} rows")
    print(f"  train ASD: {(y_train == 1).sum()}  |  train TD: {(y_train == 0).sum()}")
    print(f"Testing set:  {len(X_test)} rows")
    print(f"  test ASD:  {(y_test == 1).sum()}  |  test TD:  {(y_test == 0).sum()}\n")

    if len(np.unique(y_train)) < 2:
        print(f"{model_name}: skipped (training set needs both ASD and TD).\n")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)
    y_proba = positive_class_proba(rf, X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if len(np.unique(y_test)) < 2:
        roc_auc = float("nan")
        print("Warning: test set has only one class; ROC-AUC skipped.")
    else:
        roc_auc = roc_auc_score(y_test, y_proba)

    conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("--- Test set metrics ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}\n")

    slug = model_name.lower().replace(" ", "_").replace(":", "")
    conf_path = REPORTS_DIR / f"confusion_matrix_{slug}.png"
    if plt is not None and sns is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["TD", "ASD"],
            yticklabels=["TD", "ASD"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(model_name)
        plt.tight_layout()
        plt.savefig(conf_path)
        plt.close()
        print(f"Confusion matrix saved to {conf_path}\n")
    else:
        print("Skipping confusion matrix plot due to missing matplotlib or seaborn.")
    print(f"Confusion matrix saved to {conf_path}\n")

    cv_msg = run_cross_validation(rf, X_train_scaled, y_train)

    importance_df = pd.DataFrame({
        "feature_name": feature_cols,
        "importance_score": rf.feature_importances_,
    }).sort_values("importance_score", ascending=False)
    imp_path = REPORTS_DIR / f"feature_importance_{slug}.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"Feature importance saved to {imp_path}\n")

    summary_path = REPORTS_DIR / f"baseline_results_{slug}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Features: {', '.join(feature_cols)}\n\n")
        f.write("--- Test set metrics ---\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")
        if cv_msg:
            f.write("--- Cross-validation (5-fold) metrics (mean) ---\n")
            f.write(cv_msg)
            f.write("\n")
        f.write("Feature importances:\n")
        for _, row in importance_df.iterrows():
            f.write(f"{row['feature_name']}: {row['importance_score']:.6f}\n")

    print(f"Results saved to {summary_path}\n")

    # --- Subject-level evaluation ---
    test_meta = df[test_mask].copy()
    test_meta["y_pred"] = y_pred

    subject_predictions_list = []
    for subj_id, group in test_meta.groupby("subject_id"):
        true_label = int(group["label"].iloc[0])
        asd_count = (group["y_pred"] == 1).sum()
        td_count = (group["y_pred"] == 0).sum()
        pred_label = 1 if asd_count >= td_count else 0
        subject_predictions_list.append({
            "subject_id": subj_id,
            "true_label": true_label,
            "predicted_label": pred_label
        })

    subject_df = pd.DataFrame(subject_predictions_list)
    num_subjects = len(subject_df)

    y_subj_true = subject_df["true_label"].values
    y_subj_pred = subject_df["predicted_label"].values

    subj_accuracy = accuracy_score(y_subj_true, y_subj_pred)
    subj_precision = precision_score(y_subj_true, y_subj_pred, zero_division=0)
    subj_recall = recall_score(y_subj_true, y_subj_pred, zero_division=0)
    subj_f1 = f1_score(y_subj_true, y_subj_pred, zero_division=0)
    subj_conf_mat = confusion_matrix(y_subj_true, y_subj_pred, labels=[0, 1])

    print("--- Subject-level metrics ---")
    print()
    print(f"Subjects evaluated: {num_subjects}")
    print()
    print(f"Accuracy : {subj_accuracy:.4f}")
    print(f"Precision: {subj_precision:.4f}")
    print(f"Recall   : {subj_recall:.4f}")
    print(f"F1 Score : {subj_f1:.4f}\n")
    print("Subject-level Confusion Matrix:")
    print(subj_conf_mat)
    print()

    # Save to current working directory
    subject_df.to_csv("subject_predictions.csv", index=False)
    
    # Save model-specific version in REPORTS_DIR
    subject_df.to_csv(REPORTS_DIR / f"subject_predictions_{slug}.csv", index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        subj_conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["TD", "ASD"],
        yticklabels=["TD", "ASD"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Subject-level Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig("subject_confusion_matrix.png")
    plt.savefig(REPORTS_DIR / f"subject_confusion_matrix_{slug}.png")
    plt.close()

    print(f"Subject predictions saved to subject_predictions.csv and reports/subject_predictions_{slug}.csv")
    print(f"Subject confusion matrix saved to subject_confusion_matrix.png and reports/subject_confusion_matrix_{slug}.png\n")

    # Append subject-level metrics to the results summary file
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n--- Subject-level metrics ---\n\n")
        f.write(f"Subjects evaluated: {num_subjects}\n\n")
        f.write(f"Accuracy : {subj_accuracy:.4f}\n")
        f.write(f"Precision: {subj_precision:.4f}\n")
        f.write(f"Recall   : {subj_recall:.4f}\n")
        f.write(f"F1 Score : {subj_f1:.4f}\n\n")
        f.write("Subject-level Confusion Matrix:\n")
        f.write(np.array2string(subj_conf_mat) + "\n\n")


def main() -> None:
    if not DATA_PATH.is_file():
        sys.exit(f"Features file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        sys.exit(f"Features file is empty: {DATA_PATH}")

    # Extract subject ID
    df["subject_id"] = df["video_id"].apply(lambda name: name.split("_part_")[0])

    train_mask = df["split"].str.lower() == "train"
    test_mask = df["split"].str.lower() == "test"

    print(f"Total clips loaded: {len(df)}")
    print(f"Training clips: {train_mask.sum()}")
    print(f"Testing clips: {test_mask.sum()}")
    print(f"Unique training subjects: {df.loc[train_mask, 'subject_id'].nunique()}")
    print(f"Unique testing subjects: {df.loc[test_mask, 'subject_id'].nunique()}\n")

    all_feature_cols = [
        c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Model 1: all numeric features
    train_model("Model 1: All Features", df, all_feature_cols)

    # Model 2: motion features only
    train_model("Model 2: Motion Features Only", df, MOTION_FEATURES)

    # Model 3: bounding-box features only
    train_model("Model 3: BBox Features Only", df, BBOX_FEATURES)


if __name__ == "__main__":
    main()
