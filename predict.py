#!/usr/bin/env python
"""
ASD Video Inference Pipeline
=============================
Processes an anonymized OpenPose stickman video end‑to‑end and outputs predictions:
1. Preprocessing (Skeleton detection, Tracking, Child selection, Sequence extraction)
2. Feature extraction (Crops from original video, pre‑trained VGG16 features)
3. Standardisation (Using cached training set stats)
4. Model Loading (Backwards‑compatible Keras 2 loader for Keras 3 weights)
5. Inference (Bidirectional LSTM prediction)

Usage:
    python predict.py --video path/to/video.mp4 [options]

Added options allow you to loosen the child‑selection thresholds (useful when the tracker
fails on a new video) and to increase the centroid‑tracker tolerance.
"""

import json
import argparse
import sys
import shutil
import tempfile
import zipfile
import subprocess
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
# Define python executable path early for reuse
python_exe = sys.executable

# Add scripts directory to path to reuse training scripts logic
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "scripts"))

# Import preprocessing / feature extraction helpers
from extract_child_vgg16_features import load_vgg16, extract_video_features, load_child_boxes

class InferenceLogger:
    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.terminal = sys.stdout



    def _sanitize(self, message: str) -> str:
        """Sanitize a log message for Windows console encoding (cp1252).
        Characters that cannot be encoded are replaced with a placeholder.
        """
        try:
            return message.encode('cp1252', errors='replace').decode('cp1252')
        except Exception:
            # Fallback to utf-8 safe replace if something unexpected occurs
            return message.encode('utf-8', errors='replace').decode('utf-8')

    def info(self, message: str):
        safe_msg = self._sanitize(message)
        self.terminal.write(safe_msg + "\n")
        self.log_file.write(safe_msg + "\n")
        self.log_file.flush()

    def error(self, message: str):
        safe_msg = self._sanitize(message)
        sys.stderr.write(safe_msg + "\n")
        self.log_file.write("[ERROR] " + safe_msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()




def run_command(cmd: list[str], logger: InferenceLogger) -> bool:
    """Run a shell command as a subprocess, logging its stdout and stderr."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode})")
        logger.error(f"Stdout:\n{result.stdout}")
        logger.error(f"Stderr:\n{result.stderr}")
        return False
    logger.info("Command succeeded.")
    return True


def get_training_scaling_stats(train_npz_path: Path, cache_path: Path, logger: InferenceLogger) -> tuple[np.ndarray, np.ndarray]:
    """Load or compute mean/std stats from the training split (ignoring padding)."""
    if cache_path.is_file():
        logger.info(f"Loading cached scaling params from {cache_path}")
        data = np.load(cache_path)
        return data["mean"], data["std"]
    logger.info(f"Cache missing – computing stats from {train_npz_path}")
    if not train_npz_path.is_file():
        raise FileNotFoundError(f"Training split not found at {train_npz_path}")
    data = np.load(train_npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    lengths = data["lengths"].astype(np.int64)
    mask = np.arange(X.shape[1])[None, :] < lengths[:, None]
    vals = X[mask]
    mean = vals.mean(axis=0, keepdims=True)
    std = vals.std(axis=0, keepdims=True) + 1e-6
    np.savez_compressed(cache_path, mean=mean, std=std)
    logger.info(f"Saved new scaling params to {cache_path}")
    return mean, std


def build_reconstructed_model(weights_zip_path: Path) -> Any:
    """Recreate the Bi‑LSTM architecture and load weights from the Keras 3 zip archive."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    import h5py

    l2_val = 1e-4
    reg = regularizers.l2(l2_val)
    lstm = layers.Bidirectional(
        layers.LSTM(64, dropout=0.5, recurrent_dropout=0.2, kernel_regularizer=reg, recurrent_regularizer=reg)
    )
    model = models.Sequential([
        layers.Input(shape=(30, 512)),
        layers.Masking(mask_value=0.0),
        lstm,
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model(np.zeros((1, 30, 512), dtype=np.float32))
    with zipfile.ZipFile(weights_zip_path) as zipf:
        temp_dir = tempfile.gettempdir()
        weights_path = os.path.join(temp_dir, 'model.weights.h5')
        with open(weights_path, 'wb') as f_out:
            f_out.write(zipf.read('model.weights.h5'))
    f_h5 = h5py.File(weights_path, 'r')
    try:
        bidi_weights = [
            np.array(f_h5['layers/bidirectional/forward_layer/cell/vars/0']),
            np.array(f_h5['layers/bidirectional/forward_layer/cell/vars/1']),
            np.array(f_h5['layers/bidirectional/forward_layer/cell/vars/2']),
            np.array(f_h5['layers/bidirectional/backward_layer/cell/vars/0']),
            np.array(f_h5['layers/bidirectional/backward_layer/cell/vars/1']),
            np.array(f_h5['layers/bidirectional/backward_layer/cell/vars/2'])
        ]
        model.layers[1].set_weights(bidi_weights)
        dense_w = [np.array(f_h5['layers/dense/vars/0']), np.array(f_h5['layers/dense/vars/1'])]
        model.layers[2].set_weights(dense_w)
        dense1_w = [np.array(f_h5['layers/dense_1/vars/0']), np.array(f_h5['layers/dense_1/vars/1'])]
        model.layers[4].set_weights(dense1_w)
    finally:
        f_h5.close()
        try:
            os.remove(weights_path)
        except OSError:
            pass
    return model


def fallback_child_report(detections_csv: Path, child_report_csv: Path, logger: InferenceLogger) -> None:
    """If the child report contains only zero‑area boxes, copy the first detection row.
    This provides a crude fallback when the child‑selection filter is too strict.
    """
    df_child = pd.read_csv(child_report_csv)
    # Guard against missing 'bbox_area' column (new child_report format)
    if 'bbox_area' not in df_child.columns or (df_child['bbox_area'] == 0).all():
        logger.info("Child report missing 'bbox_area' or contains only zero‑area boxes – applying fallback using first detection.")
        df_det = pd.read_csv(detections_csv)
        if df_det.empty:
            logger.error("Detections CSV empty – cannot fallback.")
            return
        first_valid = df_det[df_det['bbox_area'] > 0].head(1)
        if not first_valid.empty:
            first_valid.to_csv(child_report_csv, index=False)
            logger.info("Fallback child report written using first valid detection.")
        else:
            logger.error("No valid detection with non‑zero bbox for fallback.")


def main():
    parser = argparse.ArgumentParser(description="Run end‑to‑end ASD inference on a video.")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--output-dir", help="Directory to store pipeline artefacts")
    parser.add_argument("--min-bbox-area", type=float, default=0.0, help="Minimum bbox area for child selection (default 0)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum OpenPose confidence for child selection (default 0)")
    parser.add_argument("--max-distance", type=int, default=150, help="Centroid‑tracker max_distance")
    parser.add_argument("--max-disappeared", type=int, default=10, help="Centroid‑tracker max_disappeared")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        print(f"Error: video not found at {video_path}")
        sys.exit(1)
    video_name = video_path.stem
    output_dir = Path(args.output_dir).resolve() if args.output_dir else PROJECT_ROOT / "outputs" / "inference" / video_name
    openpose_dir = output_dir / "openpose"
    openpose_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs.txt"
    logger = InferenceLogger(log_path)

    logger.info("=" * 40)
    logger.info("Starting ASD Video Inference Pipeline")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 40)

    # 0. Preserve raw video for debugging
    raw_copy = openpose_dir / f"{video_name}_raw.mp4"
    shutil.copy(str(video_path), str(raw_copy))
    logger.info(f"Raw video copied to {raw_copy}")

    # 1. Generate stickman video
    stickman_video = openpose_dir / f"{video_name}_stickman.mp4"
    cmd_stick = [python_exe, str(PROJECT_ROOT / "scripts" / "stickmen.py"),
                 "--input", str(video_path),
                 "--output", str(stickman_video),
                 "--model", str(PROJECT_ROOT / "pose_landmarker_lite.task")]
    if not run_command(cmd_stick, logger) or not stickman_video.is_file():
        logger.error("Stickman generation failed.")
        fail_gracefully(output_dir, "Stickman generation failed.", logger)
    logger.info(f"Stickman video saved to {stickman_video}")

    # 1. Pre‑processing – OpenPose & tracking
    logger.info("[1/5] Running OpenPose processing…")
    detections_csv = openpose_dir / f"{video_name}_detections.csv"
    bbox_video = openpose_dir / f"{video_name}_bbox.mp4"
    tracked_csv = openpose_dir / f"{video_name}_tracked.csv"
    child_report_csv = openpose_dir / f"{video_name}_child_report.csv"
    child_seq_csv = openpose_dir / f"{video_name}_child_sequence.csv"

    # Detect skeletons
    cmd_detect = [python_exe, str(PROJECT_ROOT / "scripts" / "detect_skeletons.py"),
                  "--input", str(stickman_video),
                  "--unique-id", video_name,
                  "--output", str(bbox_video),
                  "--csv", str(detections_csv),
                  "--model", str(PROJECT_ROOT / "pose_landmarker_lite.task")]
    if not run_command(cmd_detect, logger) or not detections_csv.is_file():
        logger.error("Skeleton detection failed.")
        fail_gracefully(output_dir, "Skeleton detection failed.", logger)

    # Centroid tracker with user‑provided tolerances
    cmd_track = [python_exe, str(PROJECT_ROOT / "scripts" / "centroid_tracker.py"),
                 "--input", str(detections_csv),
                 "--output", str(tracked_csv),
                 "--max-distance", str(args.max_distance),
                 "--max-disappeared", str(args.max_disappeared)]
    if not run_command(cmd_track, logger) or not tracked_csv.is_file():
        logger.error("Centroid tracking failed.")
        fail_gracefully(output_dir, "Tracking failed.", logger)

    # Child track extraction (fallback applied later)
    cmd_child = [python_exe, str(PROJECT_ROOT / "scripts" / "extract_child_track.py"),
                 "--input", str(tracked_csv),
                 "--output", str(child_report_csv)]
    if not run_command(cmd_child, logger) or not child_report_csv.is_file():
        logger.error("Child track identification failed.")
        fail_gracefully(output_dir, "Child track identification failed.", logger)

    fallback_child_report(detections_csv, child_report_csv, logger)

    # Extract child sequence
    cmd_seq = [python_exe, str(PROJECT_ROOT / "scripts" / "extract_child_sequence.py"),
               "--tracked", str(tracked_csv),
               "--report", str(child_report_csv),
               "--output-dir", str(openpose_dir)]
    if not run_command(cmd_seq, logger) or not child_seq_csv.is_file():
        logger.error("Child sequence extraction failed.")
        fail_gracefully(output_dir, "Child sequence extraction failed.", logger)

    # 2. Feature extraction (VGG16)
    logger.info("[2/5] Extracting visual features…")
    try:
        model_vgg, preprocess_input = load_vgg16()
        boxes = load_child_boxes(child_seq_csv)
        if not boxes:
            raise ValueError("No bounding boxes after child extraction.")
        features, length = extract_video_features(
            video_path, boxes, max_frames=30, margin=0.25,
            model=model_vgg, preprocess_input=preprocess_input, batch_size=32)
        if length == 0:
            raise ValueError("Feature extraction yielded 0 frames.")
        logger.info(f"Features shape: {features.shape}, active frames: {length}")
        feat_csv = output_dir / "features.csv"
        pd.DataFrame(features, columns=[f"vgg_{i}" for i in range(features.shape[1])]).to_csv(feat_csv, index=False)
        logger.info(f"Saved raw features to {feat_csv}")
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        fail_gracefully(output_dir, f"Feature extraction failed: {e}", logger)

    # 3. Scaling / normalisation
    logger.info("[3/5] Scaling features…")
    try:
        train_npz = PROJECT_ROOT / "outputs" / "vgg16_lstm" / "subset_allsubjects_20videos" / "vgg16_child_train.npz"
        cache_npz = PROJECT_ROOT / "outputs" / "vgg16_lstm" / "subset_allsubjects_20videos" / "vgg16_scaling_params.npz"
        mean, std = get_training_scaling_stats(train_npz, cache_npz, logger)
        features_scaled = (features - mean) / std
        features_scaled[length:] = 0.0
        features_batch = np.expand_dims(features_scaled, axis=0)
        logger.info("Features normalised.")
    except Exception as e:
        logger.error(f"Scaling error: {e}")
        fail_gracefully(output_dir, f"Scaling failed: {e}", logger)

    # 4. Load model
    logger.info("[4/5] Loading model…")
    try:
        model_path = PROJECT_ROOT / "outputs" / "vgg16_lstm" / "subset_allsubjects_20videos" / "child_vgg16_lstm.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = build_reconstructed_model(model_path)
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        fail_gracefully(output_dir, f"Model loading failed: {e}", logger)

    # 5. Inference
    logger.info("[5/5] Running inference…")
    try:
        pred_prob = float(model.predict(features_batch, verbose=0).flatten()[0])
        pred_class = "ASD" if pred_prob >= 0.5 else "TD"
        confidence = pred_prob if pred_prob >= 0.5 else (1.0 - pred_prob)
        logger.info("=" * 40)
        logger.info(f"Prediction: {pred_class}")
        logger.info(f"Probability (Confidence): {confidence:.4f}")
        logger.info(f"Raw activation: {pred_prob:.4f}")
        logger.info("=" * 40)
        result = {
            "video_id": video_name,
            "prediction": pred_class,
            "confidence": confidence,
            "raw_activation": pred_prob,
            "sequence_length": length,
            "status": "success",
            "error_message": None
        }
        pred_json = output_dir / "prediction.json"
        pred_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Prediction saved to {pred_json}")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        fail_gracefully(output_dir, f"Inference failed: {e}", logger)

    logger.info("Inference pipeline finished.")
    logger.close()

def fail_gracefully(output_dir: Path, message: str, logger: InferenceLogger) -> None:
    logger.error(message)
    result = {
        "video_id": output_dir.name,
        "prediction": None,
        "confidence": 0.0,
        "raw_activation": 0.0,
        "sequence_length": 0,
        "status": "failed",
        "error_message": message
    }
    (output_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.close()
    sys.exit(1)

if __name__ == "__main__":
    main()
