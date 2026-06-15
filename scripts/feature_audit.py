import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"c:/asd_project")
REPORTS_ROOT = PROJECT_ROOT / "reports"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "features" / "features.csv"
SELECTED_CSV = REPORTS_ROOT / "selected_videos.csv"
FILTER_AUDIT = PROJECT_ROOT / "outputs" / "filtering" / "filter_audit.csv"

def main():
    # Load data
    selected = pd.read_csv(SELECTED_CSV)
    filter_audit = pd.read_csv(FILTER_AUDIT)
    features = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.is_file() else pd.DataFrame()

    # Counts
    total_selected = len(selected)
    total_accepted = filter_audit[filter_audit["accepted"] == True].shape[0]
    total_features = len(features)

    # Unique counts
    unique_video_ids = selected["video_id"].nunique()
    unique_unique_ids = selected["unique_video_id"].nunique()
    duplicate_video_ids = selected["video_id"].value_counts()[lambda x: x > 1]
    duplicate_unique_ids = selected["unique_video_id"].value_counts()[lambda x: x > 1]

    # Missing in features
    missing_unique = set(selected["unique_video_id"]).difference(set(features["unique_video_id"]))
    missing_video = set(selected["video_id"]).difference(set(features["video_id"]))

    # Report
    report_path = REPORTS_ROOT / "feature_audit.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Feature Extraction Audit\n")
        f.write("========================\n\n")
        f.write(f"Total rows in selected_videos.csv: {total_selected}\n")
        f.write(f"Total accepted rows (filter_audit): {total_accepted}\n")
        f.write(f"Rows in features.csv: {total_features}\n\n")
        f.write(f"Unique video_id count: {unique_video_ids}\n")
        f.write(f"Unique unique_video_id count: {unique_unique_ids}\n\n")
        f.write("Duplicate video_id values (expected):\n")
        f.write(duplicate_video_ids.to_string())
        f.write("\n\nDuplicate unique_video_id values (unexpected):\n")
        f.write(duplicate_unique_ids.to_string())
        f.write("\n\nMissing unique_video_id in features.csv:\n")
        f.write("\n".join(sorted(missing_unique)))
        f.write("\n\nMissing video_id in features.csv (should be none if using unique):\n")
        f.write("\n".join(sorted(missing_video)))
    print(f"Audit written to {report_path}")

if __name__ == "__main__":
    main()
