#!/usr/bin/env python3
"""LOSO (Leave-One-Subject-Out) Evaluation placeholder.

This script reads the dataset manifest CSV generated earlier and computes:
- Overall training accuracy (using the 'split' column label values)
- Overall test accuracy
- A simple LOSO approximation: for each subject, treat its samples as test and predict the majority label of the training set. The overall LOSO accuracy is the average of these per‑subject accuracies.

The results are written to `c:/asd_project/reports/loso_report.txt`.
"""

import pandas as pd
import os
from collections import Counter

def main():
    manifest_path = os.path.abspath('c:/asd_project/reports/dataset_manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    # Ensure expected columns exist
    required = {'subject_id', 'split', 'label'}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing required columns: {required - set(df.columns)}")

    # Overall accuracies
    overall = {}
    for split_name in ['train', 'test']:
        split_df = df[df['split'] == split_name]
        if len(split_df) == 0:
            overall[split_name] = None
            continue
        acc = (split_df['label'] == split_df['label']).mean()  # placeholder (always 1)
        # Actually compute accuracy against itself? Use dummy prediction = most common label in split
        majority_label = split_df['label'].mode()[0]
        acc = (split_df['label'] == majority_label).mean()
        overall[split_name] = acc

    # LOSO approximation
    subjects = df['subject_id'].unique()
    loso_accuracies = []
    # Determine global majority label from training data
    train_df = df[df['split'] == 'train']
    global_majority = train_df['label'].mode()[0] if not train_df.empty else None
    for subj in subjects:
        test_subj = df[df['subject_id'] == subj]
        # Predict using global majority label
        if global_majority is None:
            loso_accuracies.append(0.0)
            continue
        acc = (test_subj['label'] == global_majority).mean()
        loso_accuracies.append(acc)
    loso_overall = sum(loso_accuracies) / len(loso_accuracies) if loso_accuracies else None

    # Write report
    out_path = os.path.abspath('c:/asd_project/reports/loso_report.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('LOSO Evaluation Report\n')
        f.write('----------------------\n')
        f.write(f"Train accuracy (majority baseline): {overall.get('train'):.4f}\n")
        f.write(f"Test accuracy (majority baseline): {overall.get('test'):.4f}\n")
        f.write(f"LOSO approximate accuracy (global majority prediction): {loso_overall:.4f}\n")
    print(f"Report written to {out_path}")

if __name__ == '__main__':
    main()
