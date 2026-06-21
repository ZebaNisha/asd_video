# xgboost_baseline.py
"""
XGBoost baseline experiment (subject‑stratified CV).

- Trains three XGBoost classifiers (all features, motion‑only, bbox‑only).
- Uses the same train/test split as the Random‑Forest baseline.
- Performs subject‑stratified cross‑validation (GroupKFold / StratifiedGroupKFold).
- Saves:
    * Clip‑level test metrics (accuracy, precision, recall, F1, ROC‑AUC)
    * Group‑CV metrics (accuracy, precision, recall, F1, ROC‑AUC)
    * Full feature‑importance CSVs (all features, sorted descending)
    * Prediction probabilities for each clip
    * Enriched subject‑level prediction CSVs (votes, confidence, mean clip probability)
    * Model pickles (for later SHAP)
    * Training / inference times
    * Consolidated comparison CSV against the Random Forest baseline
"""

import time
import pathlib
import joblib
import numpy as np
import pandas as pd
import os
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    print(f"Warning: matplotlib import failed ({e}); plots will be skipped.")
try:
    import seaborn as sns
except Exception as e:
    sns = None
    print(f"Warning: seaborn import failed ({e}); seaborn plots will be skipped.")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(r"c:/asd_project")
DATA_PATH = BASE_DIR / "outputs/features/labeled_features.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature groups (same as train_baseline.py)
# ---------------------------------------------------------------------------
META_COLS = ["unique_video_id", "video_id", "label", "split", "dataset_path", "subject_id"]

MOTION_FEATURES = [
    "mean_speed",
    "max_speed",
    "std_speed",
    "total_distance",
    "motion_burst_count",
    "activity_ratio",
]

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_subject_id(video_id: str) -> str:
    """Map Subj_10_part_1 → Subj_10."""
    return video_id.split("_part_")[0]

def positive_class_proba(model, X):
    """Return probability of class 1 (ASD)."""
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(len(X))
    idx = list(model.classes_).index(1)
    return proba[:, idx]

def run_group_cv(model, X, y, groups):
    """Subject‑stratified CV (StratifiedGroupKFold if available, else GroupKFold)."""
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splitter = cv.split(X, y, groups)
    except Exception:
        cv = GroupKFold(n_splits=5)
        splitter = cv.split(X, groups)
    acc, prec, rec, f1, roc = [], [], [], [], []
    for train_idx, test_idx in splitter:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        acc.append(accuracy_score(y_te, y_pred))
        prec.append(precision_score(y_te, y_pred, zero_division=0))
        rec.append(recall_score(y_te, y_pred, zero_division=0))
        f1.append(f1_score(y_te, y_pred, zero_division=0))
        if len(np.unique(y_te)) > 1:
            prob = positive_class_proba(model, X_te_s)
            roc.append(roc_auc_score(y_te, prob))
        else:
            roc.append(float("nan"))
    return {
        "cv_accuracy": np.mean(acc),
        "cv_precision": np.mean(prec),
        "cv_recall": np.mean(rec),
        "cv_f1": np.mean(f1),
        "cv_roc_auc": np.nanmean(roc),
    }

def aggregate_subject_predictions(df_test, y_pred, y_proba):
    """Majority voting per subject with extra columns.
    Returns a DataFrame with columns:
    subject_id, true_label, predicted_label, asd_votes, td_votes,
    confidence, mean_clip_probability.
    """
    df_test = df_test.copy()
    df_test["y_pred"] = y_pred
    df_test["y_proba"] = y_proba
    records = []
    for subj_id, grp in df_test.groupby("subject_id"):
        true_label = int(grp["label"].iloc[0])
        asd_votes = (grp["y_pred"] == 1).sum()
        td_votes = (grp["y_pred"] == 0).sum()
        total = asd_votes + td_votes
        pred_label = 1 if asd_votes >= td_votes else 0
        confidence = asd_votes / total if total > 0 else 0.0
        mean_prob = grp["y_proba"].mean()
        records.append({
            "subject_id": subj_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "asd_votes": asd_votes,
            "td_votes": td_votes,
            "confidence": confidence,
            "mean_clip_probability": mean_prob,
        })
    return pd.DataFrame(records)

def save_feature_importance(model, feature_names, slug):
    imp = model.feature_importances_
    df_imp = pd.DataFrame({"feature_name": feature_names, "importance_score": imp})
    df_imp = df_imp.sort_values("importance_score", ascending=False)
    csv_path = REPORTS_DIR / f"xgboost_feature_importance_{slug}.csv"
    df_imp.to_csv(csv_path, index=False)
    print(f"Top 10 features for {slug}:")
    print(df_imp.head(10).to_string(index=False))

def save_confusion_matrix(cm, slug):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(slug)
    plt.tight_layout()
    path = REPORTS_DIR / f"confusion_matrix_{slug}.png"
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")

def train_and_evaluate(model_name, df, feature_cols, model):
    # Verify columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{model_name}: missing columns {missing}")
    X = df[feature_cols].values
    y = df["label"].astype(int).values
    train_mask = df["split"].str.lower() == "train"
    test_mask = df["split"].str.lower() == "test"
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    # Timing
    start_train = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    train_time = time.time() - start_train
    # Inference
    start_inf = time.time()
    y_pred = model.predict(X_test_s)
    y_proba = positive_class_proba(model, X_test_s)
    inference_time = time.time() - start_inf
    # Clip‑level metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    # Subject‑level
    subject_df = aggregate_subject_predictions(df[test_mask], y_pred, y_proba)
    subj_acc = accuracy_score(subject_df["true_label"], subject_df["predicted_label"])
    subj_prec = precision_score(subject_df["true_label"], subject_df["predicted_label"], zero_division=0)
    subj_rec = recall_score(subject_df["true_label"], subject_df["predicted_label"], zero_division=0)
    subj_f1 = f1_score(subject_df["true_label"], subject_df["predicted_label"], zero_division=0)
    subj_cm = confusion_matrix(subject_df["true_label"], subject_df["predicted_label"], labels=[0, 1])
    # Group‑CV
    groups = np.array([extract_subject_id(v) for v in df["video_id"]])
    print("Unique train subjects:", len(np.unique(groups[train_mask])))
    cv_metrics = run_group_cv(model, X_train, y_train, groups[train_mask])
    # Save artefacts
    slug = model_name.lower().replace(" ", "_").replace(":", "")
    report_path = REPORTS_DIR / f"xgboost_results_{slug}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")
        f.write("--- Clip‑level Test Metrics ---\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC‑AUC  : {roc:.4f}\n")
        f.write(f"Training time (s): {train_time:.2f}\n")
        f.write(f"Inference time (s): {inference_time:.2f}\n\n")
        f.write("--- Subject‑level Metrics ---\n")
        f.write(f"Accuracy : {subj_acc:.4f}\n")
        f.write(f"Precision: {subj_prec:.4f}\n")
        f.write(f"Recall   : {subj_rec:.4f}\n")
        f.write(f"F1 Score : {subj_f1:.4f}\n\n")
        f.write("--- Subject‑level Confusion Matrix ---\n")
        f.write(str(subj_cm) + "\n\n")
        f.write("--- Subject‑Stratified Cross‑Validation (5‑fold) ---\n")
        f.write(f"Mean Accuracy : {cv_metrics['cv_accuracy']:.4f}\n")
        f.write(f"Mean Precision: {cv_metrics['cv_precision']:.4f}\n")
        f.write(f"Mean Recall   : {cv_metrics['cv_recall']:.4f}\n")
        f.write(f"Mean F1 Score : {cv_metrics['cv_f1']:.4f}\n")
        f.write(f"Mean ROC‑AUC  : {cv_metrics['cv_roc_auc']:.4f}\n")
    print(f"Report saved to {report_path}")
    # Probabilities CSV
    prob_path = REPORTS_DIR / f"xgboost_prediction_probabilities_{slug}.csv"
    pd.DataFrame({
        "unique_video_id": df.loc[test_mask, "unique_video_id"],
        "video_id": df.loc[test_mask, "video_id"],
        "probability_asd": y_proba,
    }).to_csv(prob_path, index=False)
    print(f"Probability CSV saved to {prob_path}")
    # Feature importance full
    save_feature_importance(model, feature_cols, slug)
    # Confusion matrix image
    save_confusion_matrix(confusion_matrix(y_test, y_pred, labels=[0, 1]), slug)
    # Subject predictions enriched CSV
    subj_path = REPORTS_DIR / f"subject_predictions_{slug}.csv"
    subject_df.to_csv(subj_path, index=False)
    print(f"Subject predictions saved to {subj_path}")
    # Model pickle for SHAP later
    model_path = REPORTS_DIR / f"xgboost_model_{slug}.pkl"
    joblib.dump(model, model_path)
    print(f"Model pickle saved to {model_path}")
    # Return summary for comparison table
    return {
        "Model": f"XGB {model_name}",
        "Clip Accuracy": acc,
        "Clip ROC-AUC": roc,
        "Subject Accuracy": subj_acc,
        "Subject F1": subj_f1,
        "CV Accuracy": cv_metrics["cv_accuracy"],
        "CV ROC-AUC": cv_metrics["cv_roc_auc"],
    }

def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["subject_id"] = df["video_id"].apply(extract_subject_id)
    # Random Forest baseline for comparison
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_summary = train_and_evaluate(
        "RF All Features",
        df,
        [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])],
        rf,
    )
    # XGBoost models
    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    # Model 1 – All features
    all_features = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    xgb_all = XGBClassifier(**xgb_params)
    xgb1 = train_and_evaluate("Model 1: All Features", df, all_features, xgb_all)
    # Model 2 – Motion only
    xgb_motion = XGBClassifier(**xgb_params)
    xgb2 = train_and_evaluate("Model 2: Motion Features Only", df, MOTION_FEATURES, xgb_motion)
    # Model 3 – BBox only
    xgb_bbox = XGBClassifier(**xgb_params)
    xgb3 = train_and_evaluate("Model 3: BBox Features Only", df, BBOX_FEATURES, xgb_bbox)
    # Comparison CSV
    comp_df = pd.DataFrame([rf_summary, xgb1, xgb2, xgb3])
    comp_df = comp_df[["Model", "Subject Accuracy", "Clip Accuracy", "Clip ROC-AUC", "CV Accuracy", "CV ROC-AUC"]]
    comp_path = REPORTS_DIR / "rf_vs_xgboost_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Comparison CSV saved to {comp_path}")

if __name__ == "__main__":
    main()
