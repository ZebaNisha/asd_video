import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(npy_dir, csv_path):
    df = pd.read_csv(csv_path)
    X_raw, y = [], []
    for f in sorted(os.listdir(npy_dir)):
        if f.endswith('.npy'):
            subj = f.replace('.npy', '')
            seq = np.load(os.path.join(npy_dir, f))
            X_raw.append(seq)
            # Assuming subject ID matches prefix of video_id in CSV
            label_row = df[df['video_id'].str.startswith(subj)]
            if not label_row.empty:
                label = label_row['label'].iloc[0]
                y.append(1 if label == 'ASD' else 0)
    if not X_raw:
        return np.array([]), np.array([])
    # Pad sequences to have same number of frames
    max_len = max(s.shape[0] for s in X_raw)
    X = np.array([np.pad(s, ((0, max_len - s.shape[0]), (0, 0)), mode='constant') for s in X_raw])
    return X, np.array(y)

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
    X, y = load_data(npy_dir, csv_path)
    if len(X) == 0:
        print('No data found. Ensure .npy files are present in the pose_sequences folder.')
        exit(1)
    # Simple train/test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=2)
    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    probas = model.predict(X_test).flatten()
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probas)
    }
    out_dir = r"C:\\asd_project\\outputs\\lstm"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'model.h5')
    report_path = os.path.join(out_dir, 'lstm_report.txt')
    model.save(model_path)
    with open(report_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print('Training complete. Metrics saved to', report_path)
