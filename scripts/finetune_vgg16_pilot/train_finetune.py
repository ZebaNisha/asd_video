import json
import pathlib
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = pathlib.Path('c:/asd_project')
OUTPUT_DIR = BASE_DIR / 'outputs' / 'finetune_vgg16_pilot'
CROPS_DIR = OUTPUT_DIR / 'crops'
REPORTS_DIR = OUTPUT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 'train'
TEST_SPLIT = 'test'

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image sequences and labels for a given split.

    Expected directory layout:
        crops/<split>/*.jpg  # filenames like <video_id>_frame<idx>.jpg
    The CSV `selected_subjects.csv` provides the label for each subject.
    """
    split_dir = CROPS_DIR / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    # Load label mapping from selected_subjects.csv (subject_id -> label)
    subjects_csv = REPORTS_DIR / 'selected_subjects.csv'
    subj_df = pd.read_csv(subjects_csv)
    label_map = dict(zip(subj_df['subject_id'], subj_df['label']))
    # Group frames by video_id (prefix before first '_frame')
    all_files = sorted(split_dir.glob('*.jpg'))
    video_groups = {}
    for path in all_files:
        name = path.stem  # e.g., video123_frame45
        if '_frame' not in name:
            continue
        video_id = name.split('_frame')[0]
        frame_idx = int(name.split('_frame')[1])
        video_groups.setdefault(video_id, []).append((frame_idx, path))
    sequences = []
    labels = []
    for video_id, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        imgs = []
        for _, p in frames:
            img = tf.keras.preprocessing.image.load_img(p, target_size=(224, 224))
            arr = tf.keras.preprocessing.image.img_to_array(img)
            imgs.append(arr)
        seq_arr = np.stack(imgs).astype('float32')
        seq_arr = tf.keras.applications.vgg16.preprocess_input(seq_arr)
        sequences.append(seq_arr)
        # Infer label from video_id prefix matching subject_id
        label = 0
        for subj, lbl in label_map.items():
            if video_id.startswith(subj):
                label = lbl
                break
        labels.append(label)
    if not sequences:
        return np.empty((0, 0, 224, 224, 3), dtype='float32'), np.array([], dtype='int32')
    max_len = max(seq.shape[0] for seq in sequences)
    padded = np.zeros((len(sequences), max_len, 224, 224, 3), dtype='float32')
    for i, seq in enumerate(sequences):
        l = seq.shape[0]
        padded[i, :l] = seq
    labels = np.array(labels, dtype='int32')
    return padded, labels


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

def build_model():
    # Base VGG16 (ImageNet weights)
    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')
    # Freeze all layers first
    for layer in base.layers:
        layer.trainable = False
    # Unfreeze block5 layers only
    for layer in base.layers:
        if any(name in layer.name for name in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']):
            layer.trainable = True
    # Input: (batch, timesteps, 224, 224, 3)
    input_seq = tf.keras.Input(shape=(None, 224, 224, 3), name='video_frames')
    # Apply VGG16 to each frame
    td = tf.keras.layers.TimeDistributed(base, name='time_distributed_vgg16')(input_seq)
    # BiLSTM
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), name='bilstm')(td)
    dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')(bi_lstm)
    drop = tf.keras.layers.Dropout(0.5, name='dropout')(dense1)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(drop)
    model = tf.keras.Model(inputs=input_seq, outputs=output)
    return model

# -----------------------------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------------------------

def train_and_evaluate():
    print('Loading training data...')
    x_train, y_train = load_split(TRAIN_SPLIT)
    print(f'Train samples: {x_train.shape[0]}, timesteps (max): {x_train.shape[1]}')
    print('Loading test data...')
    x_test, y_test = load_split(TEST_SPLIT)
    print(f'Test samples: {x_test.shape[0]}, timesteps (max): {x_test.shape[1]}')

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7, monitor='val_loss')
    ]

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        batch_size=4,
                        callbacks=callbacks,
                        verbose=2)

    # Save model
    model_path = OUTPUT_DIR / 'model.keras'
    model.save(model_path)
    print(f'Model saved to {model_path}')

    # Predictions on test set
    probs = model.predict(x_test, batch_size=4).flatten()
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    conf_mat = confusion_matrix(y_test, preds).tolist()

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'true_label': y_test,
        'pred_prob': probs,
        'pred_label': preds
    })
    pred_path = OUTPUT_DIR / 'predictions.csv'
    pred_df.to_csv(pred_path, index=False)

    # Training log CSV (epoch, loss, acc, val_loss, val_acc)
    log_path = OUTPUT_DIR / 'training_log.csv'
    pd.DataFrame(history.history).to_csv(log_path, index_label='epoch')

    # Report JSON
    report = {
        'train_metrics': {
            'accuracy': float(history.history['accuracy'][-1]),
            'precision': None,  # not computed on train set here
            'recall': None,
            'f1': None,
            'roc_auc': None
        },
        'test_metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc,
            'confusion_matrix': conf_mat
        },
        'generalization_gap': {
            'accuracy_gap': float(history.history['accuracy'][-1] - acc)
        },
        'success': {
            'roc_auc_threshold_met': roc >= 0.65
        }
    }
    report_path = OUTPUT_DIR / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report written to {report_path}')

    # Comparison markdown for quick view
    comp_md = f"""
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | {history.history['accuracy'][-1]:.3f} | {acc:.3f} |
| F1 | - | {f1:.3f} |
| ROC-AUC | - | {roc:.3f} |
| Generalization Gap (Acc) | - | {history.history['accuracy'][-1] - acc:.3f} |
"""
    comp_path = OUTPUT_DIR / 'comparison.md'
    comp_path.write_text(comp_md)

    # Recommendation based on outcome
    if roc < 0.5:
        recommendation = "VGG16 visual features likely do not contain ASD signal. Stop further work."
    elif 0.5 <= roc < 0.65:
        recommendation = "Signal is weak. Consider scaling to 10 ASD + 10 TD for further investigation."
    else:
        recommendation = "Fine‑tuning shows promising improvement. Proceed with larger experiment."
    rec_path = OUTPUT_DIR / 'reports' / 'recommendation.md'
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(recommendation)
    print('Recommendation written.')

if __name__ == '__main__':
    # Ensure reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    train_and_evaluate()
