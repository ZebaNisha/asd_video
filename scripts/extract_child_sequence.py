# extract_child_sequence.py
"""extract_child_sequence.py

Create a clean, child‑only temporal sequence from the tracking results.

**Why we need this**
* The downstream ST‑GCN (spatial‑temporal graph convolutional network) expects a
  *single* ordered list of skeletal keypoints per video.  By extracting only the
  child's detections we remove noise from other participants and ensure the
  model sees a consistent subject.
* Temporal ordering is crucial: the graph is built over time, connecting a node
  (the child’s skeleton) at frame *t* to the same node at frame *t+1*.  If the
  frames are shuffled the temporal edges become meaningless and the model learns
  incorrect motion patterns.

**Inputs**
1. `tracked.csv` – produced by `centroid_tracker.py` with columns:
   `frame,object_id,x,y,w,h`
2. `child_track_report.csv` – created by `extract_child_track.py`.  It contains a
   single row with the column `child_track_id` (the `object_id` of the child).

**Outputs**
`<video_id>_child_sequence.csv` – one row per frame for the child, columns:
```
frame_number,centroid_x,centroid_y,bbox_width,bbox_height,bbox_area
```
All rows are sorted by `frame_number` (ascending).

**How it works**
1. Parse command‑line arguments (`--input-tracked`, `--child-report`, `--output-dir`).
2. Load the child track id from the report (CSV with a header; we read the first
   data row and cast the value to `int`).
3. Stream the tracked CSV, keep only rows where `object_id == child_track_id`.
   For each kept row we compute:
   * `centroid_x = x + w/2`
   * `centroid_y = y + h/2`
   * `bbox_area = w * h`
   and store a tuple `(frame, centroid_x, centroid_y, w, h, area)`.
4. After reading the whole file we sort the list by `frame` to guarantee strict
   temporal order (the tracker already writes frames in order, but sorting makes the
   script robust to any out‑of‑order rows).  This step guarantees that the
   resulting sequence can be directly fed to a temporal model.
5. Write the sorted sequence to `<output_dir>/<video_id>_child_sequence.csv`.
   `video_id` is derived from the tracked‑file name (e.g. `Subj_2_part_62`
   from `Subj_2_part_62_tracked.csv`).

**Usage example**
```powershell
python scripts/extract_child_sequence.py \
    --tracked "C:/asd_project/reports/Subj_2_part_62_tracked.csv" \
    --report "C:/asd_project/reports/Subj_2_part_62_child_report.csv" \
    --output-dir "C:/asd_project/reports"
```
The script will create `C:/asd_project/reports/Subj_2_part_62_child_sequence.csv`.

**Notes**
* The script only uses the Python standard library (csv, pathlib, argparse).
* If the input files are not found at the exact location it will also try the
  `reports/` sub‑directory as a fallback, mirroring the behaviour of the previous
  stage.
* No ST‑GCN logic is included – this script merely prepares the clean, ordered
  dataset required for later graph‑based modeling.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the child's temporal sequence from tracked CSV."
    )
    parser.add_argument(
        "--tracked",
        required=True,
        help="Path to the tracked CSV produced by centroid_tracker.py",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to child_track_report.csv containing the selected child_track_id",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the child sequence CSV will be written",
    )
    return parser.parse_args()


def load_child_id(report_path: Path) -> int:
    """Read the child_track_id from the report CSV.

    The report has a header line and a single data row. We locate the column
    `child_track_id` (or fall back to the second column if the header is missing).
    """
    with open(report_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Prefer the explicit column name, otherwise use the first value
            if "child_track_id" in row:
                return int(row["child_track_id"])
            # Fallback: the report may have only numeric columns without headers
            # (unlikely given our previous script, but we guard against it).
            for value in row.values():
                try:
                    return int(value)
                except ValueError:
                    continue
    raise RuntimeError(f"Could not read child_track_id from {report_path}")


def extract_sequence(
    tracked_path: Path, child_id: int
) -> List[Tuple[int, float, float, float, float, float]]:
    """Return a list of per‑frame data for the specified child track.

    Each element is a tuple:
    (frame, centroid_x, centroid_y, width, height, area)
    """
    seq: List[Tuple[int, float, float, float, float, float]] = []
    with open(tracked_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                frame = int(row[0])
                track_id = int(row[1])
                if track_id != child_id:
                    continue
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
            except ValueError:
                continue
            centroid_x = x + w / 2.0
            centroid_y = y + h / 2.0
            area = w * h
            seq.append((frame, centroid_x, centroid_y, w, h, area))
    return seq


def write_sequence(
    output_path: Path,
    video_id: str,
    sequence: List[Tuple[int, float, float, float, float, float]],
) -> None:
    """Write the child sequence CSV sorted by frame number.

    The header follows the specification required for downstream ST‑GCN.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_number",
                "centroid_x",
                "centroid_y",
                "bbox_width",
                "bbox_height",
                "bbox_area",
            ]
        )
        for row in sequence:
            writer.writerow([
                row[0],                # frame_number
                f"{row[1]:.2f}",      # centroid_x
                f"{row[2]:.2f}",      # centroid_y
                f"{row[3]:.2f}",      # bbox_width
                f"{row[4]:.2f}",      # bbox_height
                f"{row[5]:.2f}",      # bbox_area
            ])
    print(f"Child sequence written to {output_path}")


def main():
    args = parse_args()
    tracked_path = Path(args.tracked)
    report_path = Path(args.report)
    # Fallback logic (same as previous stage)
    if not tracked_path.is_file():
        fallback = Path("reports") / tracked_path.name
        if fallback.is_file():
            print(f"Tracked CSV not found at {tracked_path}, using fallback {fallback}")
            tracked_path = fallback
        else:
            sys.exit(f"Error: tracked CSV not found at {tracked_path} or {fallback}")
    if not report_path.is_file():
        fallback = Path("reports") / report_path.name
        if fallback.is_file():
            print(f"Report CSV not found at {report_path}, using fallback {fallback}")
            report_path = fallback
        else:
            sys.exit(f"Error: child report not found at {report_path} or {fallback}")
    child_id = load_child_id(report_path)
    sequence = extract_sequence(tracked_path, child_id)
    if not sequence:
        sys.exit("No frames found for the selected child track.")
    # Ensure temporal order – sorting protects against any accidental out‑of‑order rows
    sequence.sort(key=lambda r: r[0])
    # Derive video_id from the tracked filename (e.g., Subj_2_part_62_tracked.csv -> Subj_2_part_62)
    video_id = tracked_path.stem.replace("_tracked", "")
    output_file = Path(args.output_dir) / f"{video_id}_child_sequence.csv"
    write_sequence(output_file, video_id, sequence)


if __name__ == "__main__":
    main()
