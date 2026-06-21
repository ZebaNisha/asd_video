# analysis_bilstm_metrics2.py
import pathlib, json, os, sys
import numpy as np
import tensorflow as tf

# Paths
base_dir = pathlib.Path(r'c:/asd_project/outputs/vgg16_lstm/subset_allsubjects_20videos')
train_npz = base_dir / 'vgg16_child_train.npz'
model_dir = base_dir / 'child_vgg16_lstm'
report_json = base_dir / 'child_vgg16_lstm_report.json'

result = {}
# 1. Feature variance (train split)
if train_npz.is_file():
    data = np.load(train_npz, allow_pickle=True)
    X = data['X']  # shape (samples, seq_len, feat_dim)
    # flatten across samples and timesteps
    X_flat = X.reshape(-1, X.shape[-1])
    variances = X_flat.var(axis=0)
    result['feature_variance'] = variances.tolist()
    result['feature_mean'] = X_flat.mean(axis=0).tolist()
else:
    result['feature_variance'] = None

# 2. Model trainable parameters count
try:
    loaded = tf.saved_model.load(str(model_dir))
    # Count variables
    total_params = 0
    for var in loaded.variables:
        total_params += tf.size(var).numpy()
    result['trainable_parameters'] = int(total_params)
except Exception as e:
    result['trainable_parameters'] = None
    result['model_load_error'] = str(e)

# 3. File timestamps
def ts(p):
    stat = p.stat()
    return {'created': stat.st_ctime, 'modified': stat.st_mtime}
result['timestamps'] = {
    'report_json': ts(report_json),
    'train_npz': ts(train_npz),
    'model_dir': ts(model_dir),
}

print(json.dumps(result, indent=2))
