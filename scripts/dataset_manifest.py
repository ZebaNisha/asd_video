#!/usr/bin/env python3
"""Generate a manifest of the ASD dataset.

The script reads the feature CSV (default: `c:/asd_project/outputs/feature_dataset.csv`)
and the split definition CSV (default: `c:/asd_project/reports/split.csv`).
It creates a CSV file `c:/asd_project/reports/dataset_manifest.csv` with the
columns:
    video_id, subject_id, split (train/test), label, feature_path

The `subject_id` is extracted from the video identifier assuming the pattern
`Subj_<ID>_...`. If the pattern is not found, the whole video_id is used as the
subject.
"""

import argparse
import os
import pandas as pd
import re

def extract_subject(video_id: str) -> str:
    """Extract subject identifier from a video_id.
    Expected pattern: any substring matching ``Subj_\w+``.
    If none is found, return the original video_id.
    """
    m = re.search(r"Subj_[A-Za-z0-9]+", video_id)
    return m.group(0) if m else video_id

def main():
    parser = argparse.ArgumentParser(description="Create dataset manifest CSV")
    parser.add_argument("--feature_csv", default="c:/asd_project/outputs/features/labeled_features.csv",
                        help="Path to the per‑video feature CSV (must contain a 'video_id' column)")
    parser.add_argument("--split_csv", default="c:/asd_project/reports/split.csv",
                        help="Path to the split definition CSV (columns: video_id, split, label)")
    parser.add_argument("--output", default="c:/asd_project/reports/dataset_manifest.csv",
                        help="Where to write the manifest CSV")
    args = parser.parse_args()

    # Load data
    features = pd.read_csv(args.feature_csv)
    splits = pd.read_csv(args.split_csv)

    # Merge on video_id
    df = pd.merge(splits, features[['video_id']], on="video_id", how="inner")

    # Add subject column
    df["subject_id"] = df["video_id"].apply(extract_subject)

    # Add feature_path column (relative to feature CSV location)
    feature_dir = os.path.dirname(os.path.abspath(args.feature_csv))
    df["feature_path"] = df["video_id"].apply(lambda vid: os.path.join(feature_dir, f"{vid}.npy"))

    # Reorder columns for readability
    df = df[["video_id", "subject_id", "split", "label", "feature_path"]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Dataset manifest written to {args.output}")

if __name__ == "__main__":
    main()
