# collect_dataset_stats.py
"""
Script to collect statistics from the autism video dataset and generate a markdown report.

It walks the dataset directory, counts ASD and TD videos, gathers FPS, resolution,
frame count, duration, and flags for corrupted files. The results are written to
`c:/asd_project/dataset_report.md`.
"""

import os
import glob
import random
from pathlib import Path
import cv2
import json
from collections import Counter, defaultdict

# Root of the dataset (as per project structure)
DATASET_ROOT = Path(r"c:/asd_project/autism_data_anonymized/autism_data_anonymized")
REPORT_PATH = Path(r"c:/asd_project/dataset_report.md")

def get_video_stats(video_path: Path):
    """Extract basic statistics from a video file.

    Returns a dict with keys: fps, width, height, frame_count, duration,
    corrupted (bool). If the video cannot be opened, all numeric fields are None
    and corrupted is True.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": None, "width": None, "height": None, "frame_count": None, "duration": None, "corrupted": True}
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else None
    cap.release()
    return {"fps": fps, "width": width, "height": height, "frame_count": frame_count, "duration": duration, "corrupted": False}

def load_metadata(json_path: Path):
    """Load optional per‑video metadata.
    Expected keys: "child", "examiner", "motion" (bool flags).
    If the file does not exist or parsing fails, return defaults (None).
    """
    if not json_path.is_file():
        return {"child": None, "examiner": None, "motion": None}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "child": data.get("child"),
            "examiner": data.get("examiner"),
            "motion": data.get("motion")
        }
    except Exception:
        return {"child": None, "examiner": None, "motion": None}

def main():
    # Containers for aggregated statistics
    stats = []
    category_counts = Counter()
    fps_vals = []
    resolution_vals = []
    duration_vals = []
    corrupted_videos = []
    without_child = []
    only_examiner = []
    missing_motion = []

    # Walk through ASD and TD subfolders
    for category in ["ASD", "TD"]:
        category_path = DATASET_ROOT / "training_set" / category
        if not category_path.is_dir():
            continue
        video_files = list(category_path.rglob("*.mp4"))
        category_counts[category] = len(video_files)
        for video_file in video_files:
            vstat = get_video_stats(video_file)
            vstat["path"] = video_file
            vstat["category"] = category
            stats.append(vstat)
            if vstat["corrupted"]:
                corrupted_videos.append(str(video_file))
                continue
            # Collect distributions
            if vstat["fps"] is not None:
                fps_vals.append(vstat["fps"])
            if vstat["width"] is not None and vstat["height"] is not None:
                resolution_vals.append(f"{vstat['width']}x{vstat['height']}")
            if vstat["duration"] is not None:
                duration_vals.append(vstat["duration"])

            # Look for accompanying JSON metadata (same stem)
            meta_path = video_file.with_suffix('.json')
            meta = load_metadata(meta_path)
            if meta["child"] is False:
                without_child.append(str(video_file))
            if meta["examiner"] is True and meta["child"] is None:
                only_examiner.append(str(video_file))
            if meta["motion"] is False:
                missing_motion.append(str(video_file))

    # Write markdown report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Dataset Report\n\n")
        f.write(f"**Total videos**: {sum(category_counts.values())}\n\n")
        f.write("## Video counts by category\n")
        for cat, cnt in category_counts.items():
            f.write(f"- {cat}: {cnt}\n")
        f.write("\n")
        # Helper to produce simple histogram text
        def histogram(values, bins=10):
            if not values:
                return "No data"
            min_v, max_v = min(values), max(values)
            bin_width = (max_v - min_v) / bins if max_v > min_v else 1
            bins_edges = [min_v + i * bin_width for i in range(bins + 1)]
            counts = [0] * bins
            for v in values:
                idx = min(int((v - min_v) / bin_width), bins - 1)
                counts[idx] += 1
            lines = []
            for i, c in enumerate(counts):
                low = bins_edges[i]
                high = bins_edges[i+1]
                lines.append(f"{low:.2f} – {high:.2f}: {c}")
            return "\n".join(lines)

        f.write("## FPS distribution\n```")
        f.write(histogram(fps_vals))
        f.write("```\n\n")
        f.write("## Resolution distribution\n```\n")
        # Write resolution distribution
        resolution_counts = Counter(resolution_vals)
        for res, cnt in resolution_counts.most_common():
            f.write(f"{res}: {cnt}\n")
        f.write("```\n\n")
        f.write("## Duration distribution (seconds)\n```")
        f.write(histogram(duration_vals))
        f.write("```\n\n")
        f.write("## Corrupted videos\n")
        for v in corrupted_videos:
            f.write(f"- {v}\n")
        f.write("\n## Videos without child flag\n")
        for v in without_child:
            f.write(f"- {v}\n")
        f.write("\n## Videos with only examiner flag\n")
        for v in only_examiner:
            f.write(f"- {v}\n")
        f.write("\n## Videos missing motion flag\n")
        for v in missing_motion:
            f.write(f"- {v}\n")

if __name__ == "__main__":
    main()
