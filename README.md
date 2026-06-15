# ASD Video Analysis Project

## Overview

This repository contains a full pipeline for analyzing autism‑related video data. It extracts pose landmarks using MediaPipe, engineers both absolute and scale‑invariant features, trains a PyTorch LSTM model, and evaluates performance through a comprehensive report.

## Repository Structure
```
.
├─ LSTM_EVALUATION.md          # Model evaluation report
├─ README.md                   # This file
├─ autism_data_anonymized/    # Raw video frames (anonymized)
├─ dataset_stats.json          # Statistics about the dataset
├─ features.csv                # Baseline feature CSV for training
├─ outputs/
│   ├─ child_sequences/      # Per‑video CSV sequences of features
│   └─ features/              # Feature CSV used by training scripts
├─ models/                    # Saved model checkpoints
├─ scripts/                   # Helper scripts (e.g., frame extraction)
├─ train_lstm.py               # Main training script for both stages
├─ test_mediapipe.py           # MediaPipe pose extraction demo
└─ ... (other utilities)
```

## Data Preparation
1. **Video Frames** – Stored in `autism_data_anonymized/`.
2. **Pose Extraction** – `test_mediapipe.py` uses MediaPipe to generate CSV files of landmark coordinates for each frame, saved under `outputs/child_sequences/`.
3. **Feature Engineering** – `feature_extraction_v2.py` creates two feature sets:
   - **Stage 1 (Absolute)** – Includes raw coordinates, bounding‑box sizes, speeds.
   - **Stage 2 (Scale‑Invariant)** – Normalizes out camera distance and size, keeping only kinematic ratios, angles, and normalized speed.

## Model Training
Run the main script:
```bash
python train_lstm.py
```
- **Stage 1** trains on absolute features.
- **Stage 2** trains on scale‑invariant features.
- Both stages share the same LSTM architecture (bidirectional, 2 layers, hidden size 64).
- Early stopping based on test loss is applied, and the best model checkpoint is saved to `models/lstm/`.

## Evaluation
After training, a comparative markdown report (`LSTM_EVALUATION.md`) is generated with:
- Accuracy, precision, recall, F1‑score, ROC‑AUC for each stage.
- Confusion matrices.
- Analysis of why scale‑invariant features may generalize better.

## Usage
1. **Prepare data** – Ensure the `outputs/child_sequences/` directory contains CSVs for each video (`<video_id>_child_sequence.csv`).
2. **Edit paths** – If you move the repository, update the hard‑coded paths at the top of `train_lstm.py`.
3. **Run training** – `python train_lstm.py` will automatically split data, train both stages, and write the report.
4. **Inspect results** – Open `LSTM_EVALUATION.md` in any markdown viewer.

---
*Feel free to open issues or pull requests for improvements!*
