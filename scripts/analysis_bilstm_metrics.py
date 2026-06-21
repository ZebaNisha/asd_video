# analysis_bilstm_metrics.py
"""Compute detailed BiLSTM training analysis.
Generates:
- reports/bilstm_metric_analysis.json (structured data)
- reports/bilstm_metric_analysis.md (human‑readable report)
"""
import json, pathlib, numpy as np

# Paths
base_dir = pathlib.Path('c:/asd_project')
report_json = base_dir / 'reports' / 'bilstm_metric_analysis.json'
report_md = base_dir / 'reports' / 'bilstm_metric_analysis.md'

# Load JSON report from training run
train_report_path = base_dir / 'outputs/vgg16_lstm/subset_allsubjects_20videos/child_vgg16_lstm_report.json'
with train_report_path.open() as f:
    rpt = json.load(f)

# Helper to load npz and compute variance (masked by lengths)
def load_npz(split):
    npz_path = base_dir / f'outputs/vgg16_lstm/subset_allsubjects_20videos/vgg16_child_{split}.npz'
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']  # shape (samples, seq_len, feat_dim)
    lengths = data['lengths']
    # mask padded frames
    mask = np.arange(X.shape[1])[None, :] < lengths[:, None]
    masked = X[mask]
    return masked

train_feats = load_npz('train')
test_feats = load_npz('test')

# Feature variance statistics (overall variance across all frames)
var_train = np.var(train_feats, axis=0)
var_test = np.var(test_feats, axis=0)

var_stats = {
    'train': {
        'mean_variance': float(var_train.mean()),
        'std_variance': float(var_train.std()),
        'min_variance': float(var_train.min()),
        'max_variance': float(var_train.max()),
    },
    'test': {
        'mean_variance': float(var_test.mean()),
        'std_variance': float(var_test.std()),
        'min_variance': float(var_test.min()),
        'max_variance': float(var_test.max()),
    },
}

# Model parameter count (load SavedModel)
import tensorflow as tf
model_path = base_dir / 'outputs/vgg16_lstm/subset_allsubjects_20videos/child_vgg16_lstm'
model = tf.keras.models.load_model(model_path)
trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])

# Assemble analysis dict
analysis = {
    'timestamp': rpt.get('READY_FOR_BILSTM_TRAINING'),
    'data_dir': rpt.get('data_dir'),
    'train_metrics': rpt['clip_metrics']['train'],
    'test_metrics': rpt['clip_metrics']['test'],
    'train_subject_metrics': rpt['subject_metrics']['train'],
    'test_subject_metrics': rpt['subject_metrics']['test'],
    'train_roc_auc': rpt['clip_metrics']['train'].get('roc_auc'),
    'test_roc_auc': rpt['clip_metrics']['test'].get('roc_auc'),
    'train_accuracy': rpt['clip_metrics']['train']['accuracy'],
    'test_accuracy': rpt['clip_metrics']['test']['accuracy'],
    'trainable_parameters': int(trainable_params),
    'feature_variance': var_stats,
    'class_distribution': {
        'train': rpt['train_class_counts'],
        'test': rpt['test_class_counts'],
    },
}

# Write JSON
with report_json.open('w') as f:
    json.dump(analysis, f, indent=2)

# Write Markdown report
md = f"""# BiLSTM Detailed Metric Analysis
**Dataset:** `{analysis['data_dir']}`
**Run timestamp placeholder**: N/A (report generated later)

## Train vs Test Performance
| Metric | Train | Test |
|---|---:|---:|
| Accuracy | {analysis['train_accuracy']:.4f} | {analysis['test_accuracy']:.4f} |
| ROC‑AUC | {analysis['train_roc_auc']:.4f} | {analysis['test_roc_auc']:.4f} |

## Subject‑level Test Metrics (official)
- Accuracy: {analysis['test_subject_metrics']['accuracy']:.4f}
- Precision: {analysis['test_subject_metrics']['precision']:.4f}
- Recall: {analysis['test_subject_metrics']['recall']:.4f}
- F1: {analysis['test_subject_metrics']['f1']:.4f}

## Model Complexity
- Trainable parameters: {analysis['trainable_parameters']:,}

## Feature Variance (per‑feature variance across all frames)
### Train
- Mean variance: {var_stats['train']['mean_variance']:.6f}
- Std variance: {var_stats['train']['std_variance']:.6f}
- Min variance: {var_stats['train']['min_variance']:.6f}
- Max variance: {var_stats['train']['max_variance']:.6f}
### Test
- Mean variance: {var_stats['test']['mean_variance']:.6f}
- Std variance: {var_stats['test']['std_variance']:.6f}
- Min variance: {var_stats['test']['min_variance']:.6f}
- Max variance: {var_stats['test']['max_variance']:.6f}

## Class Distribution
- Train: {analysis['class_distribution']['train']}
- Test: {analysis['class_distribution']['test']}

---
*The official final test results are the **clip‑level test metrics** and **subject‑level test metrics** shown above.*
"""
with report_md.open('w') as f:
    f.write(md)

print('Analysis completed')
