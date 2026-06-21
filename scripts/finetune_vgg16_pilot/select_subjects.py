#!/usr/bin/env python
"""Select random subjects for the fine‑tuning pilot.

Creates train/test CSV files and a summary CSV of the selected subjects.
"""

import pathlib
import random
import csv
import json
from collections import defaultdict

SEED = 42
TRAIN_ASD = 3
TRAIN_TD = 3
TEST_ASD = 3
TEST_TD = 3

# Paths – adjust if your dataset layout differs
METADATA_CSV = pathlib.Path('c:/asd_project/outputs/vgg16_lstm/subset_2subjects_10videos_metadata.csv')
OUT_DIR = pathlib.Path('c:/asd_project/outputs/finetune_vgg16_pilot')
REPORTS_DIR = OUT_DIR / 'reports'
SELECTED_CSV = REPORTS_DIR / 'selected_subjects.csv'

def load_metadata(csv_path: pathlib.Path):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def split_subjects(rows):
    # Group rows by subject id (assumes column 'subject_id' exists) and label (ASD/TD)
    by_subject = defaultdict(list)
    for r in rows:
        subj = r.get('subject_id') or r.get('unique_subject_id') or r.get('unique_video_id')
        if not subj:
            continue
        by_subject[subj].append(r)
    # Determine label per subject (assumes all rows of a subject share the same label)
    subjects = []
    for subj, recs in by_subject.items():
        label = recs[0].get('label') or recs[0].get('diagnosis')
        subjects.append((subj, label, recs))
    return subjects

def sample_subjects(subjects, n_asd, n_td):
    rng = random.Random(SEED)
    asd = [s for s in subjects if str(s[1]).lower() in {'1', 'asd', 'autism'}]
    td = [s for s in subjects if str(s[1]).lower() in {'0', 'td', 'typical'}]
    rng.shuffle(asd)
    rng.shuffle(td)
    return asd[:n_asd] + td[:n_td]

def write_split(split_name, selected, split_path):
    with split_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['unique_video_id','label','split'])
        writer.writeheader()
        for subj, label, rows in selected:
            for r in rows:
                writer.writerow({
                    'unique_video_id': r.get('unique_video_id') or r.get('video_id'),
                    'label': label,
                    'split': split_name,
                })

def main():
    rows = load_metadata(METADATA_CSV)
    subjects = split_subjects(rows)

    train_sel = sample_subjects(subjects, TRAIN_ASD, TRAIN_TD)
    # Remove already selected subjects from pool before sampling test
    remaining = [s for s in subjects if s not in train_sel]
    test_sel = sample_subjects(remaining, TEST_ASD, TEST_TD)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSVs for downstream scripts
    write_split('train', train_sel, OUT_DIR / 'train_split.csv')
    write_split('test', test_sel, OUT_DIR / 'test_split.csv')

    # Save selected_subjects summary
    with SELECTED_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id','label','split','num_clips'])
        writer.writeheader()
        for split_name, sel in [('train', train_sel), ('test', test_sel)]:
            for subj, label, rows in sel:
                writer.writerow({
                    'subject_id': subj,
                    'label': label,
                    'split': split_name,
                    'num_clips': len(rows),
                })
    print('Subject selection complete. Files written to', OUT_DIR)

if __name__ == '__main__':
    main()
