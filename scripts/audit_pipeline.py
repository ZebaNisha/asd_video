#!/usr/bin/env python
"""Read‑only audit of the ASD pipeline.
Generates:
  - outputs/reports/audit_summary.csv
  - outputs/reports/audit_recommendations.md
"""

import csv
import json
import pathlib
import sys
import numpy as np
from typing import List, Tuple, Dict

# ---------- Configuration ----------
BASE_DIR = pathlib.Path('c:/asd_project')
REPORTS_DIR = BASE_DIR / 'outputs' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Define stages with expected directories, file patterns, and validation rules
STAGES = [
    {
        "name": "Detection CSVs",
        "dir": BASE_DIR / 'outputs' / 'detections',
        "pattern": "*_detections.csv",
        "required_columns": ["video_id", "frame_number", "skeleton_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_area", "centroid_x", "centroid_y"],
        "type": "csv",
    },
    {
        "name": "Tracked CSVs",
        "dir": BASE_DIR / 'outputs' / 'tracked',
        "pattern": "*_tracked.csv",
        "required_columns": None,
        "type": "csv",
    },
    {
        "name": "Track Analysis CSVs",
        "dir": BASE_DIR / 'outputs' / 'analysis',
        "pattern": "*_analysis.csv",
        "required_columns": None,
        "type": "csv",
    },
    {
        "name": "Child Reports",
        "dir": BASE_DIR / 'outputs' / 'child_sequences',
        "pattern": "*_child_report.csv",
        "required_columns": None,
        "type": "csv",
    },
    {
        "name": "Child Sequences",
        "dir": BASE_DIR / 'outputs' / 'child_sequences',
        "pattern": "*_child_sequence.csv",
        "required_columns": None,
        "type": "csv",
    },
    {
        "name": "Filtering Outputs",
        "dir": BASE_DIR / 'outputs' / 'filtering',
        "pattern": "accepted_videos.txt",
        "required_columns": None,
        "type": "txt",
    },
    {
        "name": "Feature CSVs",
        "dir": BASE_DIR / 'outputs' / 'features',
        "pattern": "*.csv",
        "required_columns": None,
        "type": "csv",
    },
    {
        "name": "VGG16 NPZ Files",
        "dir": BASE_DIR / 'outputs' / 'features',
        "pattern": "*_vgg16.npz",
        "required_keys": ["X", "y", "lengths", "video_ids"],
        "type": "npz",
    },
    {
        "name": "JSON Reports",
        "dir": BASE_DIR / 'outputs' / 'reports',
        "pattern": "*.json",
        "type": "json",
    },
]

# ---------- Helper Functions ----------
def count_files(stage: Dict) -> Tuple[int, List[pathlib.Path]]:
    """Return expected count (based on pattern glob) and list of existing files."""
    files = list(stage["dir"].glob(stage["pattern"]))
    return len(files), files

def is_zero_byte(p: pathlib.Path) -> bool:
    return p.stat().st_size == 0

def validate_csv(p: pathlib.Path, required_columns: List[str] = None, allow_headerless: bool = False) -> bool:
    if is_zero_byte(p):
        return False
    try:
        with p.open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return False
        # Determine if header is present based on required_columns
        header_present = bool(required_columns)
        data_rows = len(rows) - (1 if header_present else 0)
        # Must have more than 1 data row
        if data_rows < 2:
            return False
        # Column count consistency
        expected_len = len(rows[0])
        for row in rows:
            if len(row) != expected_len:
                return False
        # Header handling
        if required_columns:
            header = rows[0]
            # Ensure required columns are present in header
            for col in required_columns:
                if col not in header:
                    return False
            data_rows = len(rows) - 1  # exclude header
        elif allow_headerless:
            data_rows = len(rows)  # all rows are data rows
        else:
            # Assume there is a header but no required columns specified
            data_rows = len(rows) - 1
        # Need more than one data row
        if data_rows < 2:
            return False
    except Exception:
        return False
    return True

def validate_npz(p: pathlib.Path, required_keys: List[str] = None) -> bool:
    if is_zero_byte(p):
        return False
    try:
        data = np.load(p, allow_pickle=True)
        if required_keys:
            for key in required_keys:
                if key not in data:
                    return False
                arr = data[key]
                if arr.shape[0] == 0:
                    return False
    except Exception:
        return False
    return True

def validate_json(p: pathlib.Path) -> bool:
    if is_zero_byte(p):
        return False
    try:
        with p.open() as f:
            json.load(f)
    except Exception:
        return False
    return True

def validate_txt(p: pathlib.Path) -> bool:
    if is_zero_byte(p):
        return False
    return True

# ---------- Main Audit Logic ----------
def audit_stage(stage: Dict) -> Dict:
    expected, files = count_files(stage)
    existing = len(files)
    valid = 0
    corrupted = 0
    for f in files:
        if stage["type"] == "csv":
            if stage["name"] == "Tracked CSVs":
                ok = validate_csv(f, stage.get("required_columns"), allow_headerless=True)
            else:
                ok = validate_csv(f, stage.get("required_columns"))
        elif stage["type"] == "npz":
            ok = validate_npz(f, stage.get("required_keys"))
        elif stage["type"] == "json":
            ok = validate_json(f)
        elif stage["type"] == "txt":
            ok = validate_txt(f)
        else:
            ok = True
        if ok:
            valid += 1
        else:
            corrupted += 1
    completion_pct = (existing / expected * 100) if expected > 0 else 100
    status = "COMPLETE" if completion_pct == 100 and corrupted == 0 else "INCOMPLETE"
    return {
        "Stage": stage["name"],
        "Expected Files": expected,
        "Existing Files": existing,
        "Valid Files": valid,
        "Corrupted Files": corrupted,
        "Completion %": round(completion_pct, 2),
        "Status": status,
    }

def compute_readiness_score(overall_pct: float) -> Tuple[int, str]:
    if overall_pct == 100:
        return overall_pct, "Ready for BiLSTM training immediately"
    elif 80 <= overall_pct < 100:
        return overall_pct, "Minor fixes required"
    elif 50 <= overall_pct < 80:
        return overall_pct, "Significant missing artifacts"
    else:
        return overall_pct, "Rebuild recommended"

def generate_recommendations(results: List[Dict]) -> Tuple[str, str, str]:
    # Determine high, medium, low priority actions
    high = []
    medium = []
    low = []
    for r in results:
        if r["Status"] == "INCOMPLETE":
            # Identify which stage blocks training
            if r["Stage"] in {"Child Sequences", "Filtering Outputs", "VGG16 NPZ Files"}:
                high.append(r["Stage"])
            elif r["Stage"] in {"Feature CSVs", "JSON Reports"}:
                medium.append(r["Stage"])
            else:
                low.append(r["Stage"])
    # Build recommendation text
    rec_lines = []
    if not high and not medium and not low:
        rec_lines.append("All stages complete and valid. Proceed directly to BiLSTM training.")
    else:
        if high:
            rec_lines.append("**HIGH PRIORITY** – stages blocking training:")
            rec_lines.extend([f"- {s}" for s in high])
        if medium:
            rec_lines.append("**MEDIUM PRIORITY** – stages affecting evaluation quality:")
            rec_lines.extend([f"- {s}" for s in medium])
        if low:
            rec_lines.append("**LOW PRIORITY** – missing reports, visualizations, diagnostics:")
            rec_lines.extend([f"- {s}" for s in low])
    recommendation_md = "\n".join(rec_lines)
    # Determine READY_FOR_TRAINING flag
    ready = "YES" if not high else "NO"
    # Summary block
    overall_pct = sum(r["Completion %"] for r in results) / len(results)
    score, description = compute_readiness_score(overall_pct)
    summary = f"Pipeline Readiness Score: {score}% ({description})\nReady for Training: {ready}\n"
    return recommendation_md, ready, summary

def write_csv_report(results: List[Dict]):
    csv_path = REPORTS_DIR / "audit_summary.csv"
    fieldnames = ["Stage", "Expected Files", "Existing Files", "Valid Files", "Corrupted Files", "Completion %", "Status"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

def write_md_recommendations(recommendation_md: str, summary: str, ready_flag: str):
    md_path = REPORTS_DIR / "audit_recommendations.md"
    with md_path.open("w") as f:
        f.write("# Audit Recommendations\n\n")
        f.write(summary + "\n\n")
        f.write(f"READY_FOR_TRAINING = {ready_flag}\n\n")
        f.write(recommendation_md)

def main():
    all_results = []
    for stage in STAGES:
        res = audit_stage(stage)
        all_results.append(res)
    # Write reports
    write_csv_report(all_results)
    recommendation_md, ready_flag, summary = generate_recommendations(all_results)
    write_md_recommendations(recommendation_md, summary, ready_flag)
    # Console output (concise matrix)
    print("Audit completed. Summary:\n")
    for r in all_results:
        print(f"{r['Stage']}: {r['Status']} ({r['Completion %']}% complete, {r['Corrupted Files']} corrupted)")
    print("\nRead the detailed reports at:")
    print(f"- {REPORTS_DIR / 'audit_summary.csv'}")
    print(f"- {REPORTS_DIR / 'audit_recommendations.md'}")

if __name__ == "__main__":
    main()
