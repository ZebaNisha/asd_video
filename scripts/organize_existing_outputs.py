import shutil
from pathlib import Path
import csv

# Define source directories (old locations) – assume project root
PROJECT_ROOT = Path(r"c:/asd_project")

# Define new directories (must match path_config definitions)
from path_config import (
    BBOX_VIDEOS_DIR,
    DETECTIONS_DIR,
    TRACK_ANALYSIS_DIR,
    CHILD_SEQUENCES_DIR,
    FILTERING_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    REPORTS_ROOT,
)

# Mapping of glob patterns to target directories
MAPPINGS = {
    "*_bbox.mp4": BBOX_VIDEOS_DIR,
    "*_detections.csv": DETECTIONS_DIR,
    "*_track_analysis.csv": TRACK_ANALYSIS_DIR,
    "*_child_sequence.csv": CHILD_SEQUENCES_DIR,
    "accepted_videos.txt": FILTERING_DIR,
    "rejected_videos.csv": FILTERING_DIR,
    "features.csv": FEATURES_DIR,
    "labeled_features.csv": FEATURES_DIR,
    "*_model.pkl": MODELS_DIR,
    "*_report.csv": REPORTS_ROOT,
}

# Report file for conflicts / actions
REPORT_CSV = PROJECT_ROOT / "migration_report.csv"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def migrate_file(src: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    dest = dest_dir / src.name
    if dest.exists():
        return False, f"SKIPPED: {dest} already exists"
    try:
        shutil.move(str(src), str(dest))
        return True, f"MOVED: {src} -> {dest}"
    except Exception as e:
        return False, f"ERROR moving {src} to {dest}: {e}"

def main():
    rows = []
    for pattern, target_dir in MAPPINGS.items():
        for src in PROJECT_ROOT.rglob(pattern):
            # Skip files already in the target directory
            if src.parent == target_dir:
                continue
            success, msg = migrate_file(src, target_dir)
            rows.append({"source": str(src), "destination": str(target_dir / src.name), "status": msg})
            print(msg)
    # Write report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "destination", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Migration completed. Report saved to {REPORT_CSV}")

if __name__ == "__main__":
    main()
