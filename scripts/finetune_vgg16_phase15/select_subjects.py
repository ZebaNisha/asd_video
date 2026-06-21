#!/usr/bin/env python
"""Select random subjects for Phase 1.5 fast validation.

Creates train/test split CSVs and a summary CSV (selected_subjects.csv) in
outputs/finetune_vgg16_phase15/reports/.
"""

import pathlib
import random
import csv
from collections import defaultdict

SEED = 42
TRAIN_ASD = 5
TRAIN_TD = 5
TEST_ASD = 5
TEST_TD = 5

# Paths – adjust as needed
METADATA_CSV = pathlib.Path('c:/asd_project/outputs/vgg16_lstm/subset_2subjects_10videos_metadata.csv')
OUT_ROOT = pathlib.Path('c:/asd_project/outputs/finetune_vgg16_phase15')
REPORTS_ROOT = OUT_ROOT / 'reports'
SELECTED_CSV = REPORTS_ROOT / 'selected_subjects.csv'

random.seed(SEED)

def load_metadata(csv_path: pathlib.Path):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def group_by_subject(rows):
    by_subject = defaultdict(list)
    for r in rows:
        sid = r.get('subject_id') or r.get('unique_subject_id') or r.get('unique_video_id')
        if sid:
            by_subject[sid].append(r)
    subjects = []
    for sid, recs in by_subject.items():
        label = recs[0].get('label') or recs[0].get('diagnosis')
        subjects.append((sid, label, recs))
    return subjects

def sample_subjects(pool, n_asd, n_td):
    asd = [s for s in pool if str(s[1]).lower() in {'1', 'asd', 'autism'}]
    td = [s for s in pool if str(s[1]).lower() in {'0', 'td', 'typical'}]
    random.shuffle(asd)
    random.shuffle(td)
    return asd[:n_asd] + td[:n_td]

def write_split(split_name: str, selected, split_path: pathlib.Path):
    with split_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['unique_video_id','label','split'])
        writer.writeheader()
        for sid, label, rows in selected:
            for r in rows:
                writer.writerow({
                    'unique_video_id': r.get('unique_video_id') or r.get('video_id'),
                    'label': label,
                    'split': split_name,
                })

def main():
    rows = load_metadata(METADATA_CSV)
    subjects = group_by_subject(rows)

    train_sel = sample_subjects(subjects, TRAIN_ASD, TRAIN_TD)
    remaining = [s for s in subjects if s not in train_sel]
    test_sel = sample_subjects(remaining, TEST_ASD, TEST_TD)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Write split CSVs
    write_split('train', train_sel, OUT_ROOT / 'train_split.csv')
    write_split('test', test_sel, OUT_ROOT / 'test_split.csv')

    # Write selected_subjects summary
    with SELECTED_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id','label','split','num_clips'])
        writer.writeheader()
        for split_name, sel in [('train', train_sel), ('test', test_sel)]:
            for sid, label, rows in sel:
                writer.writerow({
                    'subject_id': sid,
                    'label': label,
                    'split': split_name,
                    'num_clips': len(rows),
                })
    print('Subject selection complete. Files written to', OUT_ROOT)

if __name__ == '__main__':
    main()
