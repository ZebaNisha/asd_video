#!/usr/bin/env python
"""Orchestrate pipeline execution on Dataset-2 Svideos in parallel."""

import os
import sys
import csv
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
PROJECT_ROOT = Path("c:/asd_project").resolve()
DATASET2_ROOT = PROJECT_ROOT / "Dataset-2" / "Dataset"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "dataset2"
MODEL_TASK = PROJECT_ROOT / "pose_landmarker_lite.task"

DETECTIONS_DIR = OUTPUTS_ROOT / "detections"
TRACKED_DIR = OUTPUTS_ROOT / "tracked"
CHILD_SEQUENCES_DIR = OUTPUTS_ROOT / "child_sequences"
FEATURES_DIR = OUTPUTS_ROOT / "features"

# Create directories
for d in [DETECTIONS_DIR, TRACKED_DIR, CHILD_SEQUENCES_DIR, FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def find_svideos():
    """Discover all Svideo.avi files and assign splits and labels."""
    svideos = []
    
    # Autism
    autism_dir = DATASET2_ROOT / "Autism" / "children with ASD"
    if autism_dir.is_dir():
        for item in autism_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                child_id = int(item.name)
                svideo_path = item / "video" / "Svideo.avi"
                if svideo_path.is_file():
                    split = "train" if child_id <= 40 else "test"
                    svideos.append({
                        "path": svideo_path,
                        "label": "asd",
                        "split": split,
                        "child_id": child_id,
                        "unique_id": f"{split}_asd_Subj_D2_{child_id}_Svideo"
                    })
                    
    # Typical
    typical_dir = DATASET2_ROOT / "Typical"
    if typical_dir.is_dir():
        for item in typical_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                child_id = int(item.name)
                svideo_path = item / "video" / "Svideo.avi"
                if svideo_path.is_file():
                    split = "train" if child_id <= 40 else "test"
                    svideos.append({
                        "path": svideo_path,
                        "label": "td",
                        "split": split,
                        "child_id": child_id,
                        "unique_id": f"{split}_td_Subj_D2_{child_id}_Svideo"
                    })
                    
    svideos.sort(key=lambda x: (x["label"], x["child_id"]))
    return svideos

def run_command(cmd, log_prefix=""):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{log_prefix}Failed with return code {result.returncode}")
        print(f"Stderr:\n{result.stderr}")
        return False
    return True

def process_single_video(item, idx, total_videos):
    unique_id = item["unique_id"]
    video_path = item["path"]
    
    detection_csv = DETECTIONS_DIR / f"{unique_id}_detections.csv"
    tracked_csv = TRACKED_DIR / f"{unique_id}_tracked.csv"
    child_report_csv = CHILD_SEQUENCES_DIR / f"{unique_id}_child_report.csv"
    child_sequence_csv = CHILD_SEQUENCES_DIR / f"{unique_id}_child_sequence.csv"
    
    # Check if already processed
    if child_sequence_csv.is_file() and child_report_csv.is_file() and tracked_csv.is_file() and detection_csv.is_file():
        return unique_id, True
        
    # a. Pose detection
    cmd_detect = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "detect_skeletons.py"),
        "--input", str(video_path),
        "--unique-id", unique_id,
        "--csv", str(detection_csv),
        "--model", str(MODEL_TASK)
    ]
    bbox_video = DETECTIONS_DIR / f"{unique_id}_bbox.mp4"
    cmd_detect += ["--output", str(bbox_video)]
    
    if not run_command(cmd_detect, f"[{unique_id} Pose Detect] "):
        return unique_id, False
        
    # b. Centroid tracking
    cmd_track = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "centroid_tracker.py"),
        "--input", str(detection_csv),
        "--output", str(tracked_csv),
        "--max-distance", "150",
        "--max-disappeared", "10"
    ]
    if not run_command(cmd_track, f"[{unique_id} Tracking] "):
        return unique_id, False
        
    # c. Extract child track
    cmd_child = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "extract_child_track.py"),
        "--input", str(tracked_csv),
        "--output", str(child_report_csv)
    ]
    if not run_command(cmd_child, f"[{unique_id} Child Track] "):
        return unique_id, False
        
    # d. Extract child sequence
    cmd_seq = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "extract_child_sequence.py"),
        "--tracked", str(tracked_csv),
        "--report", str(child_report_csv),
        "--output-dir", str(CHILD_SEQUENCES_DIR)
    ]
    if not run_command(cmd_seq, f"[{unique_id} Sequence Ext] "):
        return unique_id, False
        
    return unique_id, True

def main():
    print("=" * 60)
    print("Starting Dataset-2 Svideo Pipeline Orchestration (Parallel)")
    print("=" * 60)
    
    svideos = find_svideos()
    total_videos = len(svideos)
    print(f"Found {total_videos} Svideo.avi files to process.")
    
    # 1. Run Preprocessing in parallel using ThreadPoolExecutor
    print(f"\nRunning video preprocessing in parallel using 8 workers...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_single_video, item, idx, total_videos): item["unique_id"]
            for idx, item in enumerate(svideos, 1)
        }
        for future in as_completed(futures):
            unique_id = futures[future]
            try:
                unique_id, ok = future.result()
                if ok:
                    success_count += 1
                    print(f"Finished processing {unique_id} successfully. ({success_count}/{total_videos})")
                else:
                    print(f"Failed processing {unique_id}!")
            except Exception as e:
                print(f"Exception while processing {unique_id}: {e}")

    print(f"\nPreprocessed {success_count}/{total_videos} videos successfully.")
    
    # 2. Generate Metadata CSV
    metadata_csv = OUTPUTS_ROOT / "metadata_svideo.csv"
    print(f"\nGenerating metadata file: {metadata_csv}")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_video_id", "video_id", "label", "split", "dataset_path"])
        for item in svideos:
            writer.writerow([
                item["unique_id"],
                item["unique_id"],
                item["label"],
                item["split"],
                str(item["path"].as_posix())
            ])
            
    # 3. VGG16 Feature Extraction
    print("\nRunning VGG16 Feature Extraction...")
    cmd_features = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "extract_child_vgg16_features.py"),
        "--metadata", str(metadata_csv),
        "--child-seq-dir", str(CHILD_SEQUENCES_DIR),
        "--out-dir", str(FEATURES_DIR),
        "--force"
    ]
    # We want features output logging in real-time, so we run synchronously with a subprocess log
    res_feat = subprocess.run(cmd_features, capture_output=True, text=True)
    if res_feat.returncode != 0:
        print("Feature extraction failed.")
        print(f"Stdout:\n{res_feat.stdout}")
        print(f"Stderr:\n{res_feat.stderr}")
        sys.exit(1)
    else:
        print("Feature extraction completed successfully.")
        
    # 4. Bi-LSTM Training
    print("\nRunning Bi-LSTM Classifier Training...")
    report_json = OUTPUTS_ROOT / "child_vgg16_lstm_report.json"
    cmd_train = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_child_vgg16_lstm.py"),
        "--data-dir", str(FEATURES_DIR),
        "--report", str(report_json),
        "--epochs", "30"
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    if res_train.returncode != 0:
        print("LSTM Training failed.")
        print(f"Stdout:\n{res_train.stdout}")
        print(f"Stderr:\n{res_train.stderr}")
        sys.exit(1)
    else:
        print("LSTM Training completed successfully.")
        
    # 5. Read and report results
    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 60)
    if report_json.is_file():
        with open(report_json, encoding="utf-8") as f:
            report_data = json.load(f)
            
        clip_test = report_data.get("clip_metrics", {}).get("test", {})
        subject_test = report_data.get("subject_metrics", {}).get("test", {})
        
        print("\nClip-level Test Metrics:")
        print(f"  Accuracy:  {clip_test.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {clip_test.get('precision', 0.0):.4f}")
        print(f"  Recall:    {clip_test.get('recall', 0.0):.4f}")
        print(f"  F1-Score:  {clip_test.get('f1', 0.0):.4f}")
        print(f"  ROC-AUC:   {clip_test.get('roc_auc', 'N/A')}")
        print(f"  Confusion Matrix: {clip_test.get('confusion_matrix')}")
        
        print("\nSubject-level (Majority Vote) Test Metrics:")
        print(f"  Accuracy:  {subject_test.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {subject_test.get('precision', 0.0):.4f}")
        print(f"  Recall:    {subject_test.get('recall', 0.0):.4f}")
        print(f"  F1-Score:  {subject_test.get('f1', 0.0):.4f}")
        print(f"  ROC-AUC:   {subject_test.get('roc_auc', 'N/A')}")
        print(f"  Confusion Matrix: {subject_test.get('confusion_matrix')}")
    else:
        print("Error: JSON report file not generated!")

if __name__ == "__main__":
    main()
