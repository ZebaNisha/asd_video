# path_config.py
"""Centralized configuration of project directories.
This module defines Path objects for all major output locations
and ensures that the required directories exist at import time.
"""

from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(r"c:/asd_project")

# Top‑level directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS_ROOT = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Sub‑folders within outputs
BBOX_VIDEOS_DIR = OUTPUTS_ROOT / "bbox_videos"
DETECTIONS_DIR = OUTPUTS_ROOT / "detections"
TRACKED_DIR = OUTPUTS_ROOT / "tracked"
TRACK_ANALYSIS_DIR = OUTPUTS_ROOT / "track_analysis"
CHILD_SEQUENCES_DIR = OUTPUTS_ROOT / "child_sequences"
FILTERING_DIR = OUTPUTS_ROOT / "filtering"
FEATURES_DIR = OUTPUTS_ROOT / "features"
MODELS_DIR = OUTPUTS_ROOT / "models"

# Ensure all directories exist (no‑op if already present)
for _dir in [
    SCRIPTS_DIR,
    OUTPUTS_ROOT,
    BBOX_VIDEOS_DIR,
    DETECTIONS_DIR,
    TRACKED_DIR,
    TRACK_ANALYSIS_DIR,
    CHILD_SEQUENCES_DIR,
    FILTERING_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    REPORTS_ROOT,
    LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# Export a convenient dict if needed elsewhere
DIRS = {
    "project_root": PROJECT_ROOT,
    "scripts": SCRIPTS_DIR,
    "outputs": OUTPUTS_ROOT,
    "bbox_videos": BBOX_VIDEOS_DIR,
    "detections": DETECTIONS_DIR,
    "tracked": TRACKED_DIR,
    "track_analysis": TRACK_ANALYSIS_DIR,
    "child_sequences": CHILD_SEQUENCES_DIR,
    "filtering": FILTERING_DIR,
    "features": FEATURES_DIR,
    "models": MODELS_DIR,
    "reports": REPORTS_ROOT,
    "logs": LOGS_DIR,
}
