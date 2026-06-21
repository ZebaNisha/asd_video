#!/usr/bin/env python
"""Prepare data for the fine‑tuned VGG16 + BiLSTM pilot.

* Randomly selects 3 ASD and 3 TD subjects for training, and 3 + 3 for testing (seed 42).
* Extracts 30 frames per clip, resizes to 224×224, and stores under `outputs/finetune_vgg16_pilot/crops/{train,test}/`.
* Saves a summary of selected subjects to `reports/selected_subjects.csv`.
* Saves 20 random crop images to `reports/sample_crops/` for visual sanity check.
* Generates `pilot_metadata.csv` (columns: video_path, label, split) for the training script.
"""

import pathlib
import random
import csv
import json
import shutil
import os
from collections import defaultdict
import cv2
import numpy as np

# ------------------------- Configuration -------------------------
SEED = 42
TRAIN_ASD = 3
TRAIN_TD = 3
TEST_ASD = 3
TEST_TD = 3
MAX_FRAMES_PER_CLIP = 30
CROP_SIZE = (224, 224)
MARGIN = 0.25  # relative margin around bounding box

# Paths – adjust if your dataset layout differs
METADATA_CSV = pathlib.Path('c:/asd_project/outputs/vgg16_lstm/subset_2subjects_10videos_metadata.csv')
CHILD_SEQ_DIR = pathlib.Path('c:/asd_project/outputs/child_sequences')
VIDEO_ROOT = pathlib.Path('c:/asd_project/autism_data_anonymized')  # root where original videos reside
OUT_ROOT = pathlib.Path('c:/asd_project/outputs/finetune_vgg16_pilot')
CROPS_ROOT = OUT_ROOT / 'crops'
REPORTS_ROOT = OUT_ROOT / 'reports'
SAMPLE_CROPS_ROOT = REPORTS_ROOT / 'sample_crops'
SELECTED_CSV = REPORTS_ROOT / 'selected_subjects.csv'
PILOT_METADATA = OUT_ROOT / 'pilot_metadata.csv'

random.seed(SEED)

# -----------------------------------------------------------------
def load_metadata(csv_path: pathlib.Path):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def group_by_subject(rows):
    """Return list of (subject_id, label, rows) tuples."""
    by_subject = defaultdict(list)
    for r in rows:
        sid = r.get('subject_id') or r.get('unique_subject_id') or r.get('unique_video_id')
        if not sid:
            continue
        by_subject[sid].append(r)
    subjects = []
    for sid, recs in by_subject.items():
        # Assume label is consistent within a subject
        label = recs[0].get('label') or recs[0].get('diagnosis')
        subjects.append((sid, label, recs))
    return subjects

def sample_subjects(pool, n_asd, n_td):
    asd = [s for s in pool if str(s[1]).lower() in {'1', 'asd', 'autism'}]
    td = [s for s in pool if str(s[1]).lower() in {'0', 'td', 'typical'}]
    random.shuffle(asd)
    random.shuffle(td)
    return asd[:n_asd] + td[:n_td]

def load_child_boxes(seq_path: pathlib.Path):
    """Read bounding box CSV (same format as used elsewhere)."""
    boxes = {}
    with seq_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row['frame_number']))
            cx = float(row['centroid_x'])
            cy = float(row['centroid_y'])
            w = float(row['bbox_width'])
            h = float(row['bbox_height'])
            boxes[frame] = (cx, cy, w, h)
    return boxes

def crop_frame(frame: np.ndarray, box, margin: float):
    h_img, w_img = frame.shape[:2]
    cx, cy, bw, bh = box
    pad_w = bw * margin
    pad_h = bh * margin
    x1 = max(0, int(round(cx - bw / 2 - pad_w)))
    y1 = max(0, int(round(cy - bh / 2 - pad_h)))
    x2 = min(w_img, int(round(cx + bw / 2 + pad_w)))
    y2 = min(h_img, int(round(cy + bh / 2 + pad_h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((*CROP_SIZE, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA)

def select_frames(total_frames, max_frames):
    if len(total_frames) <= max_frames:
        return total_frames
    indices = np.linspace(0, len(total_frames) - 1, max_frames).astype(int)
    return [total_frames[i] for i in indices]

def extract_crops_for_subject(subject_rows, split_name):
    """Extract crops for all videos belonging to a subject.
    Returns a list of (crop_path, label) tuples.
    """
    samples = []
    for rec in subject_rows:
        video_id = rec.get('unique_video_id') or rec.get('video_id')
        label = rec.get('label')
        video_path = VIDEO_ROOT / rec.get('dataset_path')  # assumes column exists
        seq_path = CHILD_SEQ_DIR / f"{video_id}_child_sequence.csv"
        if not seq_path.is_file():
            continue
        boxes = load_child_boxes(seq_path)
        if not boxes:
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_numbers = sorted(boxes.keys())
        selected = select_frames(frame_numbers, MAX_FRAMES_PER_CLIP)
        selected_set = set(selected)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in selected_set:
                crop = crop_frame(frame, boxes[frame_idx], MARGIN)
                # Save crop image
                out_dir = CROPS_ROOT / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"{video_id}_frame{frame_idx}.jpg"
                crop_path = out_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                samples.append((crop_path, int(label)))
            frame_idx += 1
        cap.release()
    return samples

def save_sample_crops(sample_paths):
    SAMPLE_CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(sample_paths, min(20, len(sample_paths)))
    for src in chosen:
        dst = SAMPLE_CROPS_ROOT / src.name
        shutil.copy2(src, dst)

def write_pilot_metadata(samples):
    with PILOT_METADATA.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label', 'split'])
        for path, label, split in samples:
            writer.writerow([str(path), label, split])

def main():
    # -----------------------------------------------------------------
    rows = load_metadata(METADATA_CSV)
    subjects = group_by_subject(rows)

    # ----- Train split -----
    train_pool = subjects
    train_sel = sample_subjects(train_pool, TRAIN_ASD, TRAIN_TD)
    # Remove train subjects from pool before sampling test
    remaining = [s for s in subjects if s not in train_sel]
    test_sel = sample_subjects(remaining, TEST_ASD, TEST_TD)

    # -----------------------------------------------------------------
    # Extract crops
    all_samples = []  # (crop_path, label, split)
    for split_name, sel in [('train', train_sel), ('test', test_sel)]:
        for subj_id, label, subj_rows in sel:
            crops = extract_crops_for_subject(subj_rows, split_name)
            for crop_path, lbl in crops:
                all_samples.append((crop_path, lbl, split_name))

    # Save sample crops for sanity check
    crop_paths = [p for p, _, _ in all_samples]
    if crop_paths:
        save_sample_crops(crop_paths)

    # -----------------------------------------------------------------
    # Write selected_subjects.csv
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    with SELECTED_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'label', 'split', 'num_clips'])
        writer.writeheader()
        for split_name, sel in [('train', train_sel), ('test', test_sel)]:
            for sid, label, subj_rows in sel:
                writer.writerow({
                    'subject_id': sid,
                    'label': label,
                    'split': split_name,
                    'num_clips': len(subj_rows),
                })

    # -----------------------------------------------------------------
    # Write pilot metadata for training script
    write_pilot_metadata(all_samples)
    print('Data preparation complete. Crops stored under', CROPS_ROOT)
    print('Metadata files written to', OUT_ROOT)

if __name__ == '__main__':
    main()
