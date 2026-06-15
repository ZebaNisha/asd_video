#!/usr/bin/env python3
"""Subject Leakage Audit

Generates a report listing:
- All unique subject IDs.
- Which subjects appear in train, test, or both.
- Count of videos per subject.
"""

import pandas as pd
import os
from collections import Counter

def main():
    manifest_path = os.path.abspath('c:/asd_project/reports/dataset_manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    # Expect columns: video_id, subject_id, split, label, feature_path
    subjects = df['subject_id'].unique()
    train_subjects = set(df.loc[df['split'] == 'train', 'subject_id'])
    test_subjects = set(df.loc[df['split'] == 'test', 'subject_id'])
    overlap = train_subjects.intersection(test_subjects)
    counts = Counter(df['subject_id'])
    report_lines = []
    report_lines.append('Subject Leakage Audit')
    report_lines.append('----------------------')
    report_lines.append(f'Total subjects: {len(subjects)}')
    report_lines.append(f'Train subjects: {len(train_subjects)}')
    report_lines.append(f'Test subjects: {len(test_subjects)}')
    report_lines.append(f'Overlap subjects: {len(overlap)}')
    report_lines.append('')
    report_lines.append('Overlap subject IDs:')
    report_lines.append(', '.join(sorted(overlap)))
    report_lines.append('')
    report_lines.append('Video count per subject (first 20):')
    for subj, cnt in counts.most_common(20):
        report_lines.append(f'{subj}: {cnt}')
    out_path = os.path.abspath('c:/asd_project/reports/subject_leakage_report.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'Report written to {out_path}')

if __name__ == '__main__':
    main()
