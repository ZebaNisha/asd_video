import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json

# Paths
FEATURE_CSV = r'c:/asd_project/outputs/features/features_v2.csv'
REPORT_MD = r'c:/asd_project/model_evaluation_v2.md'
MODEL_DIR = r'c:/asd_project/models'

# Load data
df = pd.read_csv(FEATURE_CSV)
X = df.drop(columns=['video_id', 'label', 'split', 'dataset_path'])
y = df['label']

# Train‑test split according to original split column
train_idx = df['split'] == 'train'
X_train, X_test = X[train_idx], X[~train_idx]
y_train, y_test = y[train_idx], y[~train_idx]

# Feature families
DYNAMICS = [c for c in X.columns if any(k in c for k in ['mean_height','mean_width','mean_area','mean_speed','max_speed','std_speed','total_distance','activity_ratio','acceleration_mean','jerk_mean','direction_change_rate'])]
PERIODICITY = ['fft_peak_freq','fft_power_ratio','fft_entropy','acf_lag1','acf_decay','acf_entropy']
SPATIAL = ['spatial_occupancy','bbox_center_variance']
BURST = ['motion_burst_count','burst_duration_mean','burst_duration_std','motion_entropy']

FAMILIES = {
    'Dynamics': DYNAMICS,
    'Periodicity': PERIODICITY,
    'Spatial': SPATIAL,
    'Burst': BURST,
    'All': X.columns.tolist()
}

os.makedirs(MODEL_DIR, exist_ok=True)

report_lines = []

for family_name, cols in FAMILIES.items():
    if not cols:
        continue
    # Scale features
    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train[cols])
    X_test_f = scaler.transform(X_test[cols])

    # ---- Logistic Regression baseline ----
    logreg = LogisticRegression(max_iter=1000, n_jobs=1)
    logreg.fit(X_train_f, y_train)
    y_pred_lr = logreg.predict(X_test_f)
    y_proba_lr = logreg.predict_proba(X_test_f)[:,1] if len(logreg.classes_)==2 else None

    # ---- Random Forest ----
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_f, y_train)
    y_pred_rf = rf.predict(X_test_f)
    y_proba_rf = rf.predict_proba(X_test_f)[:,1] if len(rf.classes_)==2 else None

    # ---- XGBoost ----
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_f, y_train)
    y_pred_xgb = xgb.predict(X_test_f)
    y_proba_xgb = xgb.predict_proba(X_test_f)[:,1] if len(xgb.classes_)==2 else None

    # Store models
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{family_name}.pkl'))
    joblib.dump(logreg, os.path.join(MODEL_DIR, f'logreg_{family_name}.pkl'))
    joblib.dump(rf, os.path.join(MODEL_DIR, f'rf_{family_name}.pkl'))
    joblib.dump(xgb, os.path.join(MODEL_DIR, f'xgb_{family_name}.pkl'))

    # Evaluation helper
    def eval_metrics(y_true, y_pred, y_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None and len(set(y_true))==2 else 'N/A'
        cm = confusion_matrix(y_true, y_pred)
        return acc, prec, rec, f1, auc, cm

    metrics_lr = eval_metrics(y_test, y_pred_lr, y_proba_lr)
    metrics_rf = eval_metrics(y_test, y_pred_rf, y_proba_rf)
    metrics_xgb = eval_metrics(y_test, y_pred_xgb, y_proba_xgb)

    # Append to report
    report_lines.append(f"## {family_name} Feature Set")
    for model_name, mets in zip(['Logistic Regression','Random Forest','XGBoost'], [metrics_lr, metrics_rf, metrics_xgb]):
        acc, prec, rec, f1, auc, cm = mets
        report_lines.append(f"### {model_name}")
        report_lines.append(f"- Accuracy: {acc:.4f}")
        report_lines.append(f"- Precision (weighted): {prec:.4f}")
        report_lines.append(f"- Recall (weighted): {rec:.4f}")
        report_lines.append(f"- F1‑score (weighted): {f1:.4f}")
        report_lines.append(f"- ROC‑AUC: {auc if isinstance(auc,float) else auc}")
        report_lines.append(f"- Confusion Matrix:\n```\n{cm}\n```\n")

# Write markdown report
with open(REPORT_MD, 'w') as f:
    f.write('# Model Evaluation V2\n\n')
    f.write('\n'.join(report_lines))

print(f"Evaluation report saved to {REPORT_MD}")
