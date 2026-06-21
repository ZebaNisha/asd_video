#!/usr/bin/env python
"""Prepare data for Phase 1.5 fine‑tuned VGG16 experiment.

- Loads `selected_subjects.csv` created by the selection script.
- For each subject/video, attempts to copy existing cropped frames from the pilot experiment
  (`outputs/finetune_vgg16_pilot/crops/{train,test}`).
- If a crop is missing, it falls back to extracting frames from the original video (same
  logic as the pilot script).
- Writes a flat `pilot_metadata.csv` (compatible with the training script) and copies 20
  random crops to `reports/sample_crops/` for visual sanity‑check.
"""

import pathlib, random, csv, shutil, cv2, numpy as np
from collections import defaultdict

SEED = 42
MAX_FRAMES_PER_CLIP = 30
CROP_SIZE = (224, 224)
MARGIN = 0.25

# --- Paths ------------------------------------------------------------
BASE = pathlib.Path('c:/asd_project')
PHASE_ROOT = BASE / 'outputs' / 'finetune_vgg16_phase15'
CROPS_ROOT = PHASE_ROOT / 'crops'
REPORTS_ROOT = PHASE_ROOT / 'reports'
SELECTED_CSV = REPORTS_ROOT / 'selected_subjects.csv'
METADATA_CSV = BASE / 'outputs' / 'vgg16_lstm' / 'subset_2subjects_10videos_metadata.csv'
CHILD_SEQ_DIR = BASE / 'outputs' / 'child_sequences'
VIDEO_ROOT = BASE / 'autism_data_anonymized'
PILOT_CROPS_ROOT = BASE / 'outputs' / 'finetune_vgg16_pilot' / 'crops'
PILOT_METADATA = PHASE_ROOT / 'pilot_metadata.csv'

random.seed(SEED)

def load_selected(csv_path):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def load_child_boxes(seq_path: pathlib.Path):
    boxes = {}
    with seq_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
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
    x1 = max(0, int(round(cx - bw/2 - pad_w)))
    y1 = max(0, int(round(cy - bh/2 - pad_h)))
    x2 = min(w_img, int(round(cx + bw/2 + pad_w)))
    y2 = min(h_img, int(round(cy + bh/2 + pad_h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((*CROP_SIZE, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA)

def select_frames(total_frames, max_frames):
    if len(total_frames) <= max_frames:
        return total_frames
    idx = np.linspace(0, len(total_frames)-1, max_frames).astype(int)
    return [total_frames[i] for i in idx]

def copy_or_extract(video_id, split_name, out_dir, all_samples):
    # First try to copy from pilot crops
    pilot_dir = PILOT_CROPS_ROOT / split_name
    pattern = f"{video_id}_frame"
    copied = False
    if pilot_dir.is_dir():
        for src in pilot_dir.glob(f"{video_id}_frame*.jpg"):
            dst = out_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            all_samples.append((dst, int(label_map.get(video_id, 0)), split_name))
            copied = True
    return copied

def main():
    # Load selected subjects
    selected = load_selected(SELECTED_CSV)
    # Load full metadata to map subject_id to video details
    metadata_rows = []
    with METADATA_CSV.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            metadata_rows.append(r)
    # Build a map from subject_id to its metadata record
    meta_map = {row['unique_video_id']: row for row in metadata_rows}
    # Build label map from selected (subject_id -> label)
    label_map = {row['subject_id']: int(row['label']) for row in selected}

    # Prepare output dirs
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_samples = []  # (path, label, split)
    for split in ['train', 'test']:
        split_dir = CROPS_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # Find videos belonging to this split
        vids = [r for r in selected if r['split'] == split]
        for rec in vids:
            video_id = rec['subject_id']
            label = int(rec['label'])
            # Try to reuse pilot crops
            copied = False
            pilot_dir = PILOT_CROPS_ROOT / split
            if pilot_dir.is_dir():
                for src in pilot_dir.glob(f"{video_id}_frame*.jpg"):
                    dst = split_dir / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    all_samples.append((dst, label, split))
                    copied = True
            if not copied:
                # Need to extract from original video
                # Retrieve dataset_path from metadata map
                video_path = VIDEO_ROOT / meta_map[video_id]['dataset_path']
                seq_path = CHILD_SEQ_DIR / f"{video_id}_child_sequence.csv"
                if not seq_path.is_file():
                    continue
                boxes = load_child_boxes(seq_path)
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                frame_nums = sorted(boxes.keys())
                selected_frames = select_frames(frame_nums, MAX_FRAMES_PER_CLIP)
                sel_set = set(selected_frames)
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx in sel_set:
                        crop = crop_frame(frame, boxes[idx], MARGIN)
                        out_name = f"{video_id}_frame{idx}.jpg"
                        out_path = split_dir / out_name
                        cv2.imwrite(str(out_path), crop)
                        all_samples.append((out_path, label, split))
                    idx += 1
                cap.release()

    # Write pilot_metadata.csv for training script
    with PILOT_METADATA.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video_path', 'label', 'split'])
        for p, lbl, sp in all_samples:
            w.writerow([str(p), lbl, sp])

    # Save 20 random sample crops for visual check
    SAMPLE_ROOT = REPORTS_ROOT / 'sample_crops'
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = [p for p, _, _ in all_samples]
    for src in random.sample(paths, min(20, len(paths))):
        shutil.copy2(src, SAMPLE_ROOT / src.name)

    print('Data preparation complete. Crops stored under', CROPS_ROOT)

if __name__ == '__main__':
    main()
