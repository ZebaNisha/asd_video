#!/usr/bin/env python3
"""Generate split.csv for the ASD dataset.

Scans the folder structure under `c:/asd_project/autism_data_anonymized/autism_data_anonymized`
which contains two top‑level folders `training_set` and `testing_set`. Inside each of those are
subfolders `ASD` and `TD` containing the video files.

The script creates a CSV file `c:/asd_project/reports/split.csv` with columns:
    video_id,split,label
where `split` is `train` or `test` and `label` is `ASD` or `TD`.
"""

import os
import csv

ROOT = r"c:/asd_project/autism_data_anonymized/autism_data_anonymized"
OUTPUT = r"c:/asd_project/reports/split.csv"

splits_map = {
    "training_set": "train",
    "testing_set": "test",
}

rows = []
for split_dir in os.listdir(ROOT):
    if split_dir not in splits_map:
        continue
    split_name = splits_map[split_dir]
    split_path = os.path.join(ROOT, split_dir)
    for label in ["ASD", "TD"]:
        label_path = os.path.join(split_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith('.mp4'):
                continue
            video_id = os.path.splitext(fname)[0]
            rows.append([video_id, split_name, label])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_id", "split", "label"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows in {OUTPUT}")
