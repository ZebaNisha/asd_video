import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

def load_data_subset(npy_dir, csv_path, limit):
    # Load CSV to get labels
    df = pd.read_csv(csv_path)
    # Create binary label column (ASD=1, TD=0). Assuming label column contains 0 for TD and 1 for ASD.
    if 'label' in df.columns:
        # Create binary label: any non-zero value is ASD (1), zero is TD (0)
        df['label_binary'] = (df['label'] != 0).astype(int)
    else:
        # Fallback if label column is missing
        raise ValueError('Label column not found in CSV')

    X = []
    y = []
    # Walk through .npy files in sorted order
    for fname in sorted(os.listdir(npy_dir)):
        if not fname.endswith('.npy'):
            continue
        # Derive video_id from filename (strip suffix and possible '_pose')
        base_name = fname.replace('_pose.npy', '').replace('.npy', '')
        if base_name not in df['video_id'].values:
            continue
        label = df.loc[df['video_id'] == base_name, 'label_binary'].values[0]
        arr = np.load(os.path.join(npy_dir, fname))
        # Replace any NaNs in the pose array with zeros
        arr = np.nan_to_num(arr, nan=0.0)
        # Skip empty arrays after cleaning
        if arr.size == 0:
            continue
        X.append(arr)
        y.append(int(label))
    # Apply limit if specified
    if limit is not None and limit < len(X):
        X = X[:limit]
        y = y[:limit]
    if not X:
        return np.array([]), np.array([])
    # Determine feature dimension from first sample
    feature_dim = X[0].shape[1] if X[0].ndim > 1 else 1
    # Pad sequences manually
    max_len = max(s.shape[0] for s in X)
    X_padded = np.zeros((len(X), max_len, feature_dim))
    for i, s in enumerate(X):
        X_padded[i, :s.shape[0], :] = s
    return X_padded, np.array(y)

def build_model(input_shape):
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    npy_dir = r"C:\\asd_project\\outputs\\pose_sequences"
    csv_path = r"C:\\asd_project\\outputs\\features\\features.csv"
    # Determine limit from environment variable or default to 4
    limit = int(os.getenv('LSTM_LIMIT', '4'))
    X, y = load_data_subset(npy_dir, csv_path, limit)
    if len(X) == 0:
        print('No data loaded. Check paths and files.')
        exit(1)
    # Stratified train-test split to ensure both classes in test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.1, verbose=2)
    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    probas = model.predict(X_test).flatten()
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probas)
    }
    out_dir = r"C:\\asd_project\\outputs\\lstm_subset"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"lstm_report_{limit}.txt")
    with open(report_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print('Training complete. Metrics saved to', report_path)
