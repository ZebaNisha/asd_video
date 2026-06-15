#!/usr/bin/env python3
"""REAL_LOSO_RF.py
Leave‑One‑Subject‑Out (LOSO) evaluation using the same feature set as the baseline
Random Forest model.

For each subject we:
  * train a RandomForestClassifier on all *other* subjects
  * predict on the held‑out subject
  * record accuracy, precision, recall, F1 and ROC‑AUC

The script produces:
  * `reports/loso_subject_results.csv` – per‑subject metrics
  * `reports/loso_summary.csv` – mean / std of metrics across subjects
  * console output that compares LOSO performance with the baseline CV and test
    metrics (computed on the same data split as `train_baseline.py`).

No majority‑class or placeholder logic is used – a real RF model is trained
for each LOSO iteration.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths & constants (mirroring train_baseline.py)
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path(r"c:/asd_project/outputs/features/labeled_features.csv")
REPORTS_DIR = pathlib.Path(r"c:/asd_project/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "unique_video_id",
    "video_id",
    "label",
    "split",
    "dataset_path",
    "subject_id",
]

# ---------------------------------------------------------------------------
# Helper – probability of the positive class (label == 1)
# ---------------------------------------------------------------------------
def positive_class_proba(model, X) -> np.ndarray:
    """Return probability of class 1.

    ``model.predict_proba`` returns an array with shape (n_samples, n_classes).
    The columns follow the ordering of ``model.classes_``.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        # Binary case where only one class was seen during training
        if 1 in model.classes_:
            return proba[:, 0]
        return np.zeros(len(X))
    idx = list(model.classes_).index(1)
    return proba[:, idx]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Feature file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if df.empty:
    raise ValueError("Feature CSV is empty")

# Ensure subject_id column exists – the baseline script creates it on the fly.
if "subject_id" not in df.columns:
    df["subject_id"] = df["video_id"].apply(lambda x: x.split("_part_")[0])

# Numeric feature columns (same logic as train_baseline.py)
all_feature_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

# ---------------------------------------------------------------------------
# 1) LOSO evaluation per subject
# ---------------------------------------------------------------------------
subject_results = []
unique_subjects = df["subject_id"].unique()

for subj in unique_subjects:
    # Masks
    train_mask = df["subject_id"] != subj
    test_mask = df["subject_id"] == subj

    X_train = df.loc[train_mask, all_feature_cols]
    y_train = df.loc[train_mask, "label"].astype(int)
    X_test = df.loc[test_mask, all_feature_cols]
    y_test = df.loc[test_mask, "label"].astype(int)

    # Skip if training data does not contain both classes
    if len(np.unique(y_train)) < 2:
        # Record NaNs for this subject
        subject_results.append({
            "subject_id": subj,
            "n_samples": len(y_test),
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
        })
        continue

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest – same hyper‑parameters as baseline
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_proba = positive_class_proba(rf, X_test_scaled)

    # Metrics (guard against single‑class test sets)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if len(np.unique(y_test)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_test, y_proba)

    subject_results.append({
        "subject_id": subj,
        "n_samples": len(y_test),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    })

# Save per-subject CSV
subject_df = pd.DataFrame(subject_results)
subject_csv_path = REPORTS_DIR / "loso_subject_results.csv"
subject_df.to_csv(subject_csv_path, index=False)
print(f"LOSO per-subject results written to {subject_csv_path}")

# ---------------------------------------------------------------------------
# 2) Summary statistics (mean / std) across subjects
# ---------------------------------------------------------------------------
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
summary = {}
for col in metric_cols:
    mean_val = subject_df[col].mean(skipna=True)
    std_val = subject_df[col].std(skipna=True)
    summary[col] = {"mean": mean_val, "std": std_val}

summary_df = pd.DataFrame(summary).T
summary_df = summary_df.rename(columns={"mean": "Mean", "std": "Std"})
summary_csv_path = REPORTS_DIR / "loso_summary.csv"
summary_df.to_csv(summary_csv_path)
print(f"LOSO summary statistics written to {summary_csv_path}")

# ---------------------------------------------------------------------------
# 3) Baseline CV (5‑fold) on the training split only
# ---------------------------------------------------------------------------
train_mask = df["split"].str.lower() == "train"
X_train_all = df.loc[train_mask, all_feature_cols]
y_train_all = df.loc[train_mask, "label"].astype(int)

# Scale whole training data once (scaler fitted on training split)
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)

# 5‑fold stratified CV – number of splits limited by smallest class count
class_counts = np.bincount(y_train_all)
smallest_class = class_counts[class_counts > 0].min()
cv_splits = min(5, smallest_class)
if cv_splits < 2:
    print("Not enough samples for CV – skipping baseline CV calculation.")
    cv_metrics = {m: (np.nan, np.nan) for m in metric_cols}
else:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_acc, cv_prec, cv_rec, cv_f1, cv_auc = [], [], [], [], []
    for train_idx, test_idx in skf.split(X_train_all_scaled, y_train_all):
        X_tr, X_te = X_train_all_scaled[train_idx], X_train_all_scaled[test_idx]
        y_tr, y_te = y_train_all.iloc[train_idx], y_train_all.iloc[test_idx]
        rf_cv = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf_cv.fit(X_tr, y_tr)
        y_pred_cv = rf_cv.predict(X_te)
        y_proba_cv = positive_class_proba(rf_cv, X_te)
        cv_acc.append(accuracy_score(y_te, y_pred_cv))
        cv_prec.append(precision_score(y_te, y_pred_cv, zero_division=0))
        cv_rec.append(recall_score(y_te, y_pred_cv, zero_division=0))
        cv_f1.append(f1_score(y_te, y_pred_cv, zero_division=0))
        # ROC‑AUC needs both classes present in the test fold
        if len(np.unique(y_te)) == 2:
            cv_auc.append(roc_auc_score(y_te, y_proba_cv))
        else:
            cv_auc.append(np.nan)
    cv_metrics = {
        "accuracy": (np.mean(cv_acc), np.std(cv_acc, ddof=1)),
        "precision": (np.mean(cv_prec), np.std(cv_prec, ddof=1)),
        "recall": (np.mean(cv_rec), np.std(cv_rec, ddof=1)),
        "f1": (np.mean(cv_f1), np.std(cv_f1, ddof=1)),
        "roc_auc": (np.nanmean(cv_auc), np.nanstd(cv_auc, ddof=1)),
    }

# ---------------------------------------------------------------------------
# 4) Baseline test metrics (train on full training split, test on test split)
# ---------------------------------------------------------------------------
test_mask = df["split"].str.lower() == "test"
X_test_all = df.loc[test_mask, all_feature_cols]
y_test_all = df.loc[test_mask, "label"].astype(int)

# Train on *all* training data
rf_full = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
rf_full.fit(X_train_all_scaled, y_train_all)
X_test_scaled = scaler_all.transform(X_test_all)
y_pred_test = rf_full.predict(X_test_scaled)
y_proba_test = positive_class_proba(rf_full, X_test_scaled)

test_metrics = {
    "accuracy": accuracy_score(y_test_all, y_pred_test),
    "precision": precision_score(y_test_all, y_pred_test, zero_division=0),
    "recall": recall_score(y_test_all, y_pred_test, zero_division=0),
    "f1": f1_score(y_test_all, y_pred_test, zero_division=0),
    "roc_auc": (roc_auc_score(y_test_all, y_proba_test)
                if len(np.unique(y_test_all)) == 2 else np.nan),
}

# ---------------------------------------------------------------------------
# 5) Comparison & conclusion
# ---------------------------------------------------------------------------
print("\n=== Baseline CV (mean ± std) ===")
for m in metric_cols:
    mean, std = cv_metrics[m]
    print(f"{m.capitalize():<9}: {mean:.4f} ± {std:.4f}")

print("\n=== Baseline Test (single evaluation) ===")
for m in metric_cols:
    val = test_metrics[m]
    print(f"{m.capitalize():<9}: {val:.4f}")

print("\n=== LOSO (mean ± std across subjects) ===")
for m in metric_cols:
    mean = summary_df.loc[m, "Mean"]
    std = summary_df.loc[m, "Std"]
    print(f"{m.capitalize():<9}: {mean:.4f} ± {std:.4f}")

# Simple inference – does LOSO hurt performance?
# We compare LOSO mean accuracy against CV mean accuracy and test accuracy.
loso_acc_mean = summary_df.loc["accuracy", "Mean"]
cv_acc_mean = cv_metrics["accuracy"][0]
test_acc = test_metrics["accuracy"]

print("\n--- Performance comparison ---")
print(f"CV mean accuracy   : {cv_acc_mean:.4f}")
print(f"Test accuracy      : {test_acc:.4f}")
print(f"LOSO mean accuracy : {loso_acc_mean:.4f}")

if loso_acc_mean < cv_acc_mean:
    print("\nObservation: LOSO accuracy is lower than CV accuracy, indicating a drop when evaluating on unseen subjects.")
else:
    print("\nObservation: LOSO accuracy is comparable or higher than CV accuracy.")

# The script ends here.
