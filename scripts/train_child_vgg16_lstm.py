#!/usr/bin/env python
"""Train an LSTM classifier on child-only VGG16 feature sequences."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM on child-only VGG16 feature NPZ files.")
    parser.add_argument("--data-dir", default="outputs/vgg16_lstm/subset_2subjects_10videos")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def load_split(data_dir: Path, split: str):
    path = data_dir / f"vgg16_child_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int64), data["lengths"].astype(np.int64), data["video_ids"]


def class_counts(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    x_train, y_train, train_lengths, train_ids = load_split(data_dir, "train")
    x_test, y_test, test_lengths, test_ids = load_split(data_dir, "test")

    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError(f"Empty train/test tensors: train={x_train.shape}, test={x_test.shape}")
    if len(class_counts(y_train)) < 2 or len(class_counts(y_test)) < 2:
        raise RuntimeError(
            f"Need both classes in train and test. train={class_counts(y_train)}, test={class_counts(y_test)}"
        )

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for this script. Install/use an environment with tensorflow."
        ) from exc

    tf.random.set_seed(42)
    np.random.seed(42)

    # Standardize VGG features using train frames only, ignoring padded zeros.
    mask = np.arange(x_train.shape[1])[None, :] < train_lengths[:, None]
    train_values = x_train[mask]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train[~mask] = 0.0
    test_mask = np.arange(x_test.shape[1])[None, :] < test_lengths[:, None]
    x_test[~test_mask] = 0.0

    model = models.Sequential(
        [
            layers.Input(shape=x_train.shape[1:]),
            layers.Masking(mask_value=0.0),
            layers.LSTM(args.hidden_dim, dropout=args.dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(args.dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(np.int64)
    metrics = compute_metrics(y_test, y_pred)

    report = {
        "data_dir": str(data_dir),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "train_video_examples": train_ids[:5].tolist(),
        "test_video_examples": test_ids[:5].tolist(),
        "final_metrics": metrics,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }

    report_path = Path(args.report) if args.report else data_dir / "child_vgg16_lstm_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    model.save(data_dir / "child_vgg16_lstm.keras")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
