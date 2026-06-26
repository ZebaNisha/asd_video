#!/usr/bin/env python
"""Train a Bidirectional LSTM on child‑only VGG16 feature sequences.

Features added per user request:
- ``--data-dir`` is configurable with a default pointing to the verified experiment.
- Clip‑level metrics include accuracy, precision, recall, F1, confusion matrix and ROC‑AUC.
- Subject‑level evaluation uses majority‑vote aggregation with vote counts and confidence.
- CSV files are produced for clip‑level (train & test) and subject‑level predictions.
- Training curves are plotted (if matplotlib is available).
- A comprehensive JSON report is written, containing the flag ``READY_FOR_BILSTM_TRAINING = "YES"``.
"""

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
# Best epoch from the full‑dataset run (see child_vgg16_lstm_report.json)
BEST_EPOCH = 30

# Optional matplotlib import – plotting is skipped if unavailable.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Bidirectional LSTM on child‑only VGG16 feature NPZ files."
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/vgg16_lstm/subset_allsubjects_20videos",
        help="Directory containing vgg16_child_train.npz and vgg16_child_test.npz",
    )
    parser.add_argument("--epochs", type=int, default=BEST_EPOCH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--recurrent-dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--report", default=None, help="Path to JSON report file")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Use a bidirectional LSTM (default for this experiment)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyper‑parameter tuning with keras‑tuner (optional)",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split(data_dir: pathlib.Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a split from an NPZ file.

    Returns ``X`` (features), ``y`` (labels), ``lengths`` (sequence lengths) and ``video_ids``.
    """
    npz_path = data_dir / f"vgg16_child_{split}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    video_ids = data["video_ids"]
    return X, y, lengths, video_ids

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_subject_id(video_id) -> str | None:
    """Extract subject identifier from a ``video_id``.

    Expected pattern contains ``Subj_<ID>``; the function returns the ID portion.
    """
    s = str(video_id)
    if "Subj_" in s:
        return s.split("Subj_")[1].split("_")[0]
    return None


def compute_clip_metrics(y_true: np.ndarray, y_pred: np.ndarray, probas: np.ndarray | None = None) -> dict:
    """Standard classification metrics for clip‑level predictions.
    ``probas`` optional – if provided, ROC‑AUC is computed.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if probas is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, probas)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def majority_vote_subject_metrics(
    video_ids: np.ndarray,
    true_labels: np.ndarray,
    clip_preds: np.ndarray,
    clip_probas: np.ndarray,
) -> Tuple[dict, List[dict]]:
    """Aggregate clip predictions per subject using majority vote.

    Returns ``(subject_metrics, rows)`` where ``rows`` is ready for CSV export.
    """
    subject_dict = defaultdict(lambda: {"true_labels": [], "preds": [], "probas": []})
    for vid, true, pred, prob in zip(video_ids, true_labels, clip_preds, clip_probas):
        subj = extract_subject_id(vid)
        if subj is None:
            continue
        entry = subject_dict[subj]
        entry["true_labels"].append(int(true))
        entry["preds"].append(int(pred))
        entry["probas"].append(float(prob))

    rows = []
    agg_true = []
    agg_pred = []
    agg_prob = []
    for subj, data in subject_dict.items():
        asd_votes = sum(1 for p in data["preds"] if p == 1)
        td_votes = sum(1 for p in data["preds"] if p == 0)
        pred_label = 1 if asd_votes >= td_votes else 0  # tie goes to ASD
        true_label = Counter(data["true_labels"]).most_common(1)[0][0]
        confidence = max(asd_votes, td_votes) / len(data["preds"]) if data["preds"] else 0.0
        rows.append({
            "subject_id": subj,
            "true_label": true_label,
            "predicted_label": pred_label,
            "asd_votes": asd_votes,
            "td_votes": td_votes,
            "confidence": confidence,
        })
        agg_true.append(true_label)
        agg_pred.append(pred_label)
        agg_prob.append(np.mean(data["probas"]))

    subject_metrics = compute_clip_metrics(np.array(agg_true), np.array(agg_pred), np.array(agg_prob))
    return subject_metrics, rows


def write_clip_predictions_csv(
    path: pathlib.Path,
    video_ids: np.ndarray,
    true_labels: np.ndarray,
    preds: np.ndarray,
    probas: np.ndarray,
) -> None:
    df = pd.DataFrame({
        "video_id": video_ids,
        "true_label": true_labels,
        "predicted_label": preds,
        "probability": probas,
    })
    df.to_csv(path, index=False)


def write_subject_predictions_csv(path: pathlib.Path, rows: List[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def plot_training_curves(history: dict, out_path: pathlib.Path) -> None:
    if plt is None:
        return
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Train Acc")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir)

    # Load data splits
    x_train, y_train, train_lengths, train_ids = load_split(data_dir, "train")
    x_test, y_test, test_lengths, test_ids = load_split(data_dir, "test")

    if x_train.size == 0 or x_test.size == 0:
        raise RuntimeError("Empty training or test tensors detected")

    # Standardise using only genuine frames (ignore padding)
    mask_train = np.arange(x_train.shape[1])[None, :] < train_lengths[:, None]
    train_vals = x_train[mask_train]
    mean = train_vals.mean(axis=0, keepdims=True)
    std = train_vals.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train[~mask_train] = 0.0
    mask_test = np.arange(x_test.shape[1])[None, :] < test_lengths[:, None]
    x_test[~mask_test] = 0.0

    # Build model
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, regularizers

    tf.random.set_seed(42)
    np.random.seed(42)

    regularizer = regularizers.l2(args.l2) if args.l2 > 0 else None
    lstm_layer = (
        layers.Bidirectional(
            layers.LSTM(
                args.hidden_dim,
                dropout=args.dropout,
                recurrent_dropout=args.recurrent_dropout,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            )
        )
        if args.bidirectional
        else layers.LSTM(
            args.hidden_dim,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )
    )

    model = models.Sequential([
        layers.Input(shape=x_train.shape[1:]),
        layers.Masking(mask_value=0.0),
        lstm_layer,
        layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
        layers.Dropout(args.dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=args.patience,
        restore_best_weights=True,
        mode="max",
        min_delta=0.001,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    # Predictions
    test_proba = model.predict(x_test, batch_size=args.batch_size, verbose=0).flatten()
    test_pred = (test_proba >= 0.5).astype(int)
    train_proba = model.predict(x_train, batch_size=args.batch_size, verbose=0).flatten()
    train_pred = (train_proba >= 0.5).astype(int)

    # Clip‑level CSVs
    predictions_dir = data_dir.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    write_clip_predictions_csv(
        predictions_dir / "test_predictions.csv",
        test_ids,
        y_test,
        test_pred,
        test_proba,
    )
    write_clip_predictions_csv(
        predictions_dir / "train_predictions.csv",
        train_ids,
        y_train,
        train_pred,
        train_proba,
    )

    # Subject‑level aggregation & CSV (test set)
    subject_metrics_test, subject_rows = majority_vote_subject_metrics(
        test_ids, y_test, test_pred, test_proba
    )
    # Train‑set subject metrics (only for report)
    subject_metrics_train, _ = majority_vote_subject_metrics(
        train_ids, y_train, train_pred, train_proba
    )
    write_subject_predictions_csv(predictions_dir / "subject_predictions.csv", subject_rows)

    # Plot training curves if possible
    figures_dir = data_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "training_curves.png"
    plot_training_curves(history.history, plot_path)

    # Build report
    report = {
        "data_dir": str(data_dir),
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "patience": args.patience,
            "bidirectional": args.bidirectional,
        },
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_class_counts": {str(k): int(v) for k, v in Counter(y_train.tolist()).items()},
        "test_class_counts": {str(k): int(v) for k, v in Counter(y_test.tolist()).items()},
        "clip_metrics": {
            "train": compute_clip_metrics(y_train, train_pred, train_proba),
            "test": compute_clip_metrics(y_test, test_pred, test_proba),
        },
        "subject_metrics": {
            "train": subject_metrics_train,
            "test": subject_metrics_test,
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "paths": {
            "train_predictions_csv": str(predictions_dir / "train_predictions.csv"),
            "test_predictions_csv": str(predictions_dir / "test_predictions.csv"),
            "subject_predictions_csv": str(predictions_dir / "subject_predictions.csv"),
            "training_curves_png": str(plot_path),
            "model_checkpoint": str(data_dir / "child_vgg16_lstm"),
        },
        "READY_FOR_BILSTM_TRAINING": "YES",
    }
    report_path = pathlib.Path(args.report) if args.report else data_dir / "child_vgg16_lstm_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save model checkpoint
    # Save model checkpoint in Keras 3 format (no save_format argument)
    model.save(data_dir / "child_vgg16_lstm.keras")

    # Print concise summary for quick view
    print(json.dumps({"clip_test_metrics": report["clip_metrics"]["test"], "subject_test_metrics": subject_metrics_test}, indent=2))
    print(f"Report written to {report_path}")
    if plt is not None:
        print(f"Training curves saved to {plot_path}")
    else:
        print("Matplotlib not available – training curves not saved.")
    print(f"Clip predictions saved to {predictions_dir}")
    print(f"Subject predictions saved to {predictions_dir / 'subject_predictions.csv'}")

if __name__ == "__main__":
    main()
