# build_feature_dataset.py
"""build_feature_dataset.py

Generate a feature dataset from accepted child‑sequence CSV files.

* Reads the list of accepted videos from ``accepted_videos.txt``.
* For each video loads ``<video_id>_child_sequence.csv``.
* Computes motion, bounding‑box and temporal activity features.
* Writes ``features.csv`` (one row per video) **including the unique identifier** and label information.
"""

import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from path_config import CHILD_SEQUENCES_DIR, FILTERING_DIR, FEATURES_DIR, REPORTS_ROOT

SELECTED_VIDEOS_FILE = REPORTS_ROOT / "selected_videos.csv"
ACCEPTED_VIDEOS_FILE = FILTERING_DIR / "accepted_videos.txt"
FEATURES_CSV = FEATURES_DIR / "features.csv"
REPORT_CSV = REPORTS_ROOT / "feature_extraction_report.csv"

# ---------------------------------------------------------------------------
# Helper: load accepted video IDs (basename without extension)
# ---------------------------------------------------------------------------
def load_accepted_videos() -> List[str]:
    """Return a list of video IDs (basename without extension)."""
    if not ACCEPTED_VIDEOS_FILE.is_file():
        raise FileNotFoundError(f"Accepted videos list not found: {ACCEPTED_VIDEOS_FILE}")
    videos: List[str] = []
    with open(ACCEPTED_VIDEOS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                videos.append(Path(line).stem)
    return videos

# ---------------------------------------------------------------------------
# Helper: load selected video metadata for mapping to unique IDs
# ---------------------------------------------------------------------------
def load_selected_metadata() -> pd.DataFrame:
    """Return a DataFrame of selected_videos.csv for quick lookup.
    Expected columns: unique_video_id, video_id, dataset_path, split, label
    """
    if not SELECTED_VIDEOS_FILE.is_file():
        raise FileNotFoundError(f"Selected videos metadata not found: {SELECTED_VIDEOS_FILE}")
    return pd.read_csv(SELECTED_VIDEOS_FILE)

# ---------------------------------------------------------------------------
# Feature extraction helpers (unchanged)
# ---------------------------------------------------------------------------

def compute_motion_features(df: pd.DataFrame) -> Tuple[float, float, float, float, np.ndarray]:
    """Calculate speed based statistics.
    Returns (mean_speed, max_speed, std_speed, total_distance, speed_array)
    """
    # Compute frame-to-frame centroid differences
    dx = df["centroid_x"].diff().iloc[1:].to_numpy()
    dy = df["centroid_y"].diff().iloc[1:].to_numpy()
    speeds = np.sqrt(dx ** 2 + dy ** 2)
    if speeds.size == 0:
        return 0.0, 0.0, 0.0, 0.0, speeds
    return (
        float(np.mean(speeds)),
        float(np.max(speeds)),
        float(np.std(speeds, ddof=0)),
        float(np.sum(speeds)),
        speeds,
    )

def compute_bbox_features(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return mean, std, min, max of bbox_area, mean/std of width, mean/std of height."""
    area = df["bbox_area"].to_numpy()
    width = df["bbox_width"].to_numpy()
    height = df["bbox_height"].to_numpy()
    return (
        float(np.mean(area)),
        float(np.std(area, ddof=0)),
        float(np.min(area)),
        float(np.max(area)),
        float(np.mean(width)),
        float(np.std(width, ddof=0)),
        float(np.mean(height)),
        float(np.std(height, ddof=0)),
    )

def compute_temporal_features(speeds: np.ndarray) -> Tuple[float, int]:
    """Compute activity ratio and motion burst count.
    * activity_ratio – proportion of frames where speed > 5 px.
    * motion_burst_count – frames where speed > mean + 2*std.
    """
    if speeds.size == 0:
        return 0.0, 0
    activity = np.mean(speeds > 5.0)
    mean = np.mean(speeds)
    std = np.std(speeds, ddof=0)
    burst = int(np.sum(speeds > (mean + 2 * std)))
    return float(activity), burst

# ---------------------------------------------------------------------------
# Main processing – now attaches unique_video_id, label, split, dataset_path
# ---------------------------------------------------------------------------

def process_video(video_id: str, unique_id: str, selected_meta: pd.DataFrame) -> Tuple[dict, dict, str]:
    """Process a single video's child‑sequence and return feature and report rows.
    Returns (feature_row, report_row, status).
    """
    seq_path = CHILD_SEQUENCES_DIR / f"{unique_id}_child_sequence.csv"
    if not seq_path.is_file():
        return {}, {}, "missing_sequence"
    try:
        df = pd.read_csv(seq_path)
    except Exception as e:
        return {}, {}, f"load_error:{e}"
    required_cols = {"frame_number", "centroid_x", "centroid_y", "bbox_width", "bbox_height", "bbox_area"}
    if not required_cols.issubset(df.columns):
        return {}, {}, "missing_columns"
    if len(df) < 30:
        return {}, {}, "too_few_frames"
    if df.empty:
        return {}, {}, "empty_file"
    # Motion / bbox / temporal features
    mean_speed, max_speed, std_speed, total_distance, speeds = compute_motion_features(df)
    (
        mean_area,
        std_area,
        min_area,
        max_area,
        mean_width,
        std_width,
        mean_height,
        std_height,
    ) = compute_bbox_features(df)
    activity_ratio, motion_burst_count = compute_temporal_features(speeds)
    # Base feature row (without label info yet)
    feature_row = {
        "video_id": video_id,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "std_speed": std_speed,
        "total_distance": total_distance,
        "mean_area": mean_area,
        "std_area": std_area,
        "min_area": min_area,
        "max_area": max_area,
        "mean_width": mean_width,
        "std_width": std_width,
        "mean_height": mean_height,
        "std_height": std_height,
        "activity_ratio": activity_ratio,
        "motion_burst_count": motion_burst_count,
    }
    # Attach metadata from selected_videos.csv
    meta_rows = selected_meta[selected_meta["unique_video_id"] == unique_id]
    if meta_rows.empty:
        return feature_row, {}, "no_metadata"
    chosen = meta_rows.iloc[0]
    feature_row["unique_video_id"] = chosen["unique_video_id"]
    feature_row["label"] = chosen["label"]
    feature_row["split"] = chosen["split"]
    feature_row["dataset_path"] = chosen["dataset_path"]
    report_row = {
        "video_id": video_id,
        "number_of_frames": int(len(df)),
        "total_distance": total_distance,
        "activity_ratio": activity_ratio,
        "processing_status": "ok",
    }
    return feature_row, report_row, "ok"

# ---------------------------------------------------------------------------
def main():
    # Load selected video metadata (includes unique_video_id, label, split, dataset_path)
    selected_meta = load_selected_metadata()
    # Optionally, load accepted video list to filter (if needed)
    accepted_videos = set(load_accepted_videos())

    found_count = 0
    missing_count = 0
    feature_dict: dict = {}
    report_rows: List[dict] = []
    skipped = 0

    for _, meta in selected_meta.iterrows():
        video_id = meta["video_id"].strip()
        unique_vid = meta["unique_video_id"].strip()
        
        # Compute sequence path and debug output
        seq_path = CHILD_SEQUENCES_DIR / f"{unique_vid}_child_sequence.csv"
        print(f"DEBUG: accepted set size={len(accepted_videos)}")
        print(f"DEBUG: unique_vid={unique_vid}, seq_path={seq_path}, exists={seq_path.is_file()}")
        if seq_path.is_file():
            found_count += 1
        else:
            missing_count += 1
            report_rows.append({"video_id": video_id, "number_of_frames": 0, "total_distance": 0.0, "activity_ratio": 0.0, "processing_status": "missing_sequence"})
            continue
        # Process the video as before
        feat, rep, status = process_video(video_id, unique_vid, selected_meta)
        if status != "ok":
            report_rows.append({"video_id": video_id, "number_of_frames": 0, "total_distance": 0.0, "activity_ratio": 0.0, "processing_status": status})
            print(f"Skipped {video_id}: {status}")
            continue
        # Attach metadata already added inside process_video, no need to duplicate
        # Store in dict
        if unique_vid in feature_dict:
            raise RuntimeError(f"Duplicate unique_video_id detected during feature extraction: {unique_vid}")
        feature_dict[unique_vid] = feat
        report_rows.append(rep)
        
    # Convert dict values to list for DataFrame creation
    feature_rows = list(feature_dict.values())

    # Save outputs – ensure unique_video_id is first column
    if feature_rows:
        df_feat = pd.DataFrame(feature_rows)
        cols = list(df_feat.columns)
        if "unique_video_id" in cols:
            cols.insert(0, cols.pop(cols.index("unique_video_id")))
            df_feat = df_feat[cols]
        df_feat.to_csv(FEATURES_CSV, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)

    print(f"Found sequence files: {found_count}")
    print(f"Missing sequence files: {missing_count}")
    print(f"Feature rows generated: {len(feature_rows)}")

if __name__ == "__main__":
    main()
