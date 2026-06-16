# ASD Video Analysis Project

## Overview

This repository contains a child-focused video analysis pipeline for anonymized OpenPose-style stickman videos. The current modeling path tracks the child, crops the child-only region from each frame, extracts VGG16 visual features from those crops, and trains an LSTM over the resulting frame-feature sequence.

The old landmark-based graph experiments have been removed because they were unreliable on rendered stickman videos and the current goal is to avoid learning from parents or examiners in the full frame.

## Current Pipeline

1. Run or reuse the child tracking pipeline to produce:
   `outputs/child_sequences/<unique_video_id>_child_sequence.csv`
2. Create a balanced subset metadata file:
   `scripts/create_balanced_subset_metadata.py`
3. Extract child-only VGG16 feature sequences:
   `scripts/extract_child_vgg16_features.py`
4. Train the LSTM classifier:
   `scripts/train_child_vgg16_lstm.py`

## Useful Commands

Create a balanced 2-subject-per-group subset with 30 clips per subject:

```powershell
C:\Users\moham\miniconda3\python.exe scripts\create_balanced_subset_metadata.py --subjects-per-group 2 --max-videos-per-subject 30 --output outputs\vgg16_lstm\subset_2subjects_30videos_metadata.csv
```

Extract child-only VGG16 features:

```powershell
C:\Users\moham\miniconda3\python.exe scripts\extract_child_vgg16_features.py --metadata outputs\vgg16_lstm\subset_2subjects_30videos_metadata.csv --child-seq-dir outputs\child_sequences --out-dir outputs\vgg16_lstm\subset_2subjects_30videos --max-frames 30 --batch-size 32 --force
```

Train the VGG16+LSTM model:

```powershell
C:\Users\moham\miniconda3\python.exe scripts\train_child_vgg16_lstm.py --data-dir outputs\vgg16_lstm\subset_2subjects_30videos --epochs 30 --batch-size 16 --hidden-dim 128 --dropout 0.3
```

## Latest Smoke-Test Result

The `subset_2subjects_30videos` run used 120 train clips and 120 test clips, balanced across ASD and TD.

Result:

```text
Accuracy:  0.7833
Precision: 0.8269
Recall:    0.7167
F1-score:  0.7679
```

The report is saved at:
`outputs/vgg16_lstm/subset_2subjects_30videos/child_vgg16_lstm_report.json`
