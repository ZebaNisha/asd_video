# analyze_tracks.py
"""
Analysis script for tracked skeleton data.

This script reads the CSV output from `centroid_tracker.py` (columns: frame,object_id,x,y,w,h),
computes per‑track summary statistics and writes a new CSV with the results.

Metrics per `track_id`:
  - average width
  - average height
  - average area (width * height)
  - average x position (top‑left x of the bounding box)
  - duration in frames (number of detections for that track)
  - rank by average height (smallest height = rank 1)

Usage:
    python analyze_tracks.py --input tracked.csv --output track_analysis.csv
"""

import argparse
import csv
from pathlib import Path
from path_config import TRACK_ANALYSIS_DIR
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze tracking CSV and produce per‑track statistics.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the tracked CSV produced by centroid_tracker.py",
    )
    parser.add_argument(
        "--output",
        default=str(TRACK_ANALYSIS_DIR / "track_analysis.csv"),
        help="Path to write the analysis CSV (default: track_analysis.csv)",
    )
    return parser.parse_args()


def load_tracks(csv_path: Path):
    """Load tracking data.

    Returns a dict mapping track_id -> list of (x, y, w, h) tuples.
    """
    tracks = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            # row: frame, object_id, x, y, w, h
            try:
                track_id = int(row[1])
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
            except ValueError:
                continue
            tracks[track_id].append((x, y, w, h))
    return tracks


def compute_stats(tracks: dict):
    """Calculate required statistics for each track.

    Returns a list of dicts, each containing:
        track_id, avg_width, avg_height, avg_area, avg_x, duration_frames
    """
    results = []
    for track_id, boxes in tracks.items():
        if not boxes:
            continue
        total_w = sum(b[2] for b in boxes)
        total_h = sum(b[3] for b in boxes)
        total_area = sum(b[2] * b[3] for b in boxes)
        total_x = sum(b[0] for b in boxes)
        count = len(boxes)
        results.append(
            {
                "track_id": track_id,
                "avg_width": total_w / count,
                "avg_height": total_h / count,
                "avg_area": total_area / count,
                "avg_x": total_x / count,
                "duration_frames": count,
            }
        )
    # Rank by average height (smallest -> largest)
    results.sort(key=lambda r: r["avg_height"])
    for rank, r in enumerate(results, start=1):
        r["height_rank"] = rank
    return results


def write_analysis(csv_path: Path, stats: list):
    fieldnames = [
        "track_id",
        "avg_width",
        "avg_height",
        "avg_area",
        "avg_x",
        "duration_frames",
        "height_rank",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


def main():
    args = parse_args()
    input_path = Path(args.input)
    # If the provided path does not exist, try looking in the 'reports' subdirectory
    if not input_path.is_file():
        fallback_path = Path('reports') / input_path.name
        if fallback_path.is_file():
            print(f"Input CSV not found at {input_path}, using fallback {fallback_path}")
            input_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Detection CSV not found: {input_path.resolve()}\n"
                "Run detect_skeletons.py first, e.g.:\n"
                '  python scripts/detect_skeletons.py --input "path/to/video.mp4" '
                '--csv "Subj_2_part_75_detections.csv"'
            )
    output_path = Path(args.output)
    tracks = load_tracks(input_path)
    stats = compute_stats(tracks)
    write_analysis(output_path, stats)
    print(f"Analysis written to {output_path}")


if __name__ == "__main__":
    main()
