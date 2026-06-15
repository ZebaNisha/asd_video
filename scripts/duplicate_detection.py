# duplicate_detection.py
"""
Duplicate Detection Script
==========================
This script scans the dataset (train and test splits) for video duplicates.
Two types of duplicates are identified:
1. **Exact duplicates** – identical files (SHA‑256 hash match).
2. **Near duplicates** – visually similar videos. We extract the first frame
   of each video using ``ffmpeg`` and compute a perceptual hash (phash) with
   ``imagehash``. Two videos are considered near‑duplicates when the Hamming
   distance between their phashes is **≤ 8** (default, user‑approved).

For every duplicate group the script records:
- Duplicate Group ID
- Number of videos in the group
- Whether the videos share the same label (ASD/TD)
- Whether they belong to the same split (train/test)
- Whether they come from the same subject

The results are written to:
- ``reports/duplicate_analysis.md`` (human‑readable summary)
- ``reports/duplicate_groups.csv`` (machine‑readable table)

Usage::
    python scripts/duplicate_detection.py
"""

import os
import hashlib
import csv
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import imagehash
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (user‑approved values)
# ---------------------------------------------------------------------------
PHASH_HAMMING_THRESHOLD = 8  # ≤ 8 bits difference -> near duplicate
DATA_ROOT = Path("c:/asd_project")
SPLIT_CSV = DATA_ROOT / "reports" / "split.csv"
FEATURES_CSV = DATA_ROOT / "reports" / "dataset_manifest.csv"
REPORT_MD = DATA_ROOT / "reports" / "duplicate_analysis.md"
REPORT_CSV = DATA_ROOT / "reports" / "duplicate_groups.csv"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return the SHA‑256 hash of a file (read in binary mode)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_first_frame(video_path: Path, temp_dir: Path) -> Path:
    """Extract the first frame of *video_path* using ffmpeg.
    The frame is saved as a temporary PNG inside *temp_dir* and the path
    to the PNG is returned.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{video_path.stem}_frame.png"
    # Suppress ffmpeg output; -v quiet eliminates console spam.
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if it exists
        "-loglevel",
        "quiet",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path

def phash_file(video_path: Path, temp_dir: Path) -> imagehash.ImageHash:
    """Compute a perceptual hash for *video_path*.
    The hash is based on the first extracted frame.
    """
    frame_path = extract_first_frame(video_path, temp_dir)
    with Image.open(frame_path) as img:
        return imagehash.phash(img)

# ---------------------------------------------------------------------------
# Load split metadata (assumes CSV with columns: video_path,label,subject)
# ---------------------------------------------------------------------------
def load_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda c: c.strip().lower())
    return df

metadata_df = load_split(SPLIT_CSV)

# ---------------------------------------------------------------------------
# Compute hashes for every video
# ---------------------------------------------------------------------------
hash_records = []
phash_records = []
temp_hash_dir = DATA_ROOT / "scratch" / "phash_frames"
for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Hashing videos"):
    video_path = DATA_ROOT / row["video_id"]
    if not video_path.is_file():
        continue
    sha = sha256_file(video_path)
    hash_records.append({
        "video": str(video_path),
        "sha256": sha,
        "label": row["label"],
        "split": row["split"],
        "subject": row.get("subject_id", None),
    })
    # Perceptual hash (may be slower – keep inside try/catch)
    try:
        ph = phash_file(video_path, temp_hash_dir)
        phash_records.append({
            "video": str(video_path),
            "phash": str(ph),
            "label": row["label"],
            "split": row["split"],
            "subject": row.get("subject_id", None),
        })
    except Exception as e:
        # If extraction fails, skip phash for this video
        print(f"[WARN] Failed phash for {video_path}: {e}")

hash_df = pd.DataFrame(hash_records)
phash_df = pd.DataFrame(phash_records)

# ---------------------------------------------------------------------------
# Exact duplicate groups (SHA‑256)
# ---------------------------------------------------------------------------
if not hash_df.empty:
    exact_groups = (
        hash_df.groupby("sha256")
        .filter(lambda g: len(g) > 1)
        .groupby("sha256")
    )
else:
    exact_groups = []

# ---------------------------------------------------------------------------
# Near‑duplicate groups (phash Hamming distance ≤ threshold)
# ---------------------------------------------------------------------------
if not phash_df.empty:
    # Build a list of (index, phash) for pairwise comparison
    phash_list = list(phash_df["phash"].apply(imagehash.hex_to_hash).items())
    near_groups = defaultdict(list)
    group_id = 0
    for i in range(len(phash_list)):
        idx_i, hash_i = phash_list[i]
        for j in range(i + 1, len(phash_list)):
            idx_j, hash_j = phash_list[j]
            if hash_i - hash_j <= PHASH_HAMMING_THRESHOLD:
                # Assign both videos to the same group
                # Use a simple union‑find via dict of group ids
                gid_i = near_groups.get(idx_i)
                gid_j = near_groups.get(idx_j)
                if gid_i is None and gid_j is None:
                    group_id += 1
                    near_groups[idx_i] = group_id
                    near_groups[idx_j] = group_id
                elif gid_i is not None and gid_j is None:
                    near_groups[idx_j] = gid_i
                elif gid_i is None and gid_j is not None:
                    near_groups[idx_i] = gid_j
                else:
                    # Merge groups (assign all members of gid_j to gid_i)
                    for k, v in list(near_groups.items()):
                        if v == gid_j:
                            near_groups[k] = gid_i
    # Convert near_groups mapping to a DataFrame
    near_df = pd.DataFrame([
        {
            "group_id": gid,
            "video": phash_df.iloc[idx]["video"],
            "label": phash_df.iloc[idx]["label"],
            "subject": phash_df.iloc[idx]["subject"],
            "split": phash_df.iloc[idx]["split"],
        }
        for idx, gid in near_groups.items()
    ])
else:
    near_df = pd.DataFrame()




# ---------------------------------------------------------------------------
# Helper to summarise a duplicate group
# ---------------------------------------------------------------------------
def summarise_group(df: pd.DataFrame):
    n = len(df)
    same_label = df["label"].nunique() == 1
    same_split = df["split"].nunique() == 1
    same_subject = df["subject"].nunique() == 1
    return {
        "group_id": df.iloc[0]["group_id"] if "group_id" in df.columns else df.iloc[0]["sha256"],
        "num_videos": n,
        "same_label": same_label,
        "same_split": same_split,
        "same_subject": same_subject,
    }

summary_rows = []
# Exact groups
for sha, grp in exact_groups:
    grp_df = hash_df[hash_df["sha256"] == sha]
    summary = summarise_group(grp_df)
    summary["type"] = "exact"
    summary_rows.append(summary)
# Near groups
if not near_df.empty:
    for gid, grp in near_df.groupby("group_id"):
        summary = summarise_group(grp)
        summary["type"] = "near"
        summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)

# ---------------------------------------------------------------------------
# Write CSV and Markdown report
# ---------------------------------------------------------------------------
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(REPORT_CSV, index=False)

with REPORT_MD.open("w", encoding="utf-8") as f_md:
    f_md.write("# Duplicate Detection Report\n\n")
    f_md.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f_md.write("## Summary Table\n\n")
    # Write summary table with fallback if tabulate is missing
    try:
        table_md = summary_df.to_markdown(index=False)
    except Exception:
        # Fallback to plain string representation
        table_md = summary_df.to_string(index=False)
    f_md.write(table_md)
    f_md.write("\n\n")
    f_md.write("## Detailed Groups\n\n")
    for _, row in summary_df.iterrows():
        f_md.write(f"### Group `{row['group_id']}` ({row['type']} duplicate)\n")
        f_md.write(f"- Number of videos: {row['num_videos']}\n")
        f_md.write(f"- Same label? {'Yes' if row['same_label'] else 'No'}\n")
        f_md.write(f"- Same split? {'Yes' if row['same_split'] else 'No'}\n")
        f_md.write(f"- Same subject? {'Yes' if row['same_subject'] else 'No'}\n\n")

print("Duplicate detection completed. Reports written to:")
print(REPORT_MD)
print(REPORT_CSV)
