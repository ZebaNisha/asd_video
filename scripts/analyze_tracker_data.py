"""
Tracker Data Analyzer & CSV Report Generator
============================================
This script runs the custom Centroid Tracker over 100 randomly sampled videos from the dataset.
For each video, it tracks all skeleton figures and calculates their average bounding-box dimensions
(height, width, area) and horizontal X-position. It writes these statistics to a CSV report
and computes aggregate statistics to determine if the smallest tracked person is usually the child.
"""

import csv
import random
from collections import OrderedDict
import numpy as np
from pathlib import Path
import cv2

# Set random seed for reproducibility
random.seed(42)

ROOT_DIR = Path(r"c:\asd_project\autism_data_anonymized\autism_data_anonymized")
CSV_PATH = Path(r"c:\asd_project\tracker_analysis_report.csv")


class CentroidTracker:
    def __init__(self, maxDisappeared=10, maxDistance=75.0):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            N = len(objectCentroids)
            M = len(inputCentroids)
            D = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(N)).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            unusedCols = set(range(M)).difference(usedCols)
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects


def analyze_video(video_path: Path) -> dict:
    """
    Processes a single video, tracks individuals, and returns their average bounding-box metrics.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    tracker = CentroidTracker(maxDisappeared=10, maxDistance=75.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # To store stats for each tracked ID: id -> list of (width, height, area, cX)
    id_history = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

        rects = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= 150:
                rects.append((x, y, w, h))

        tracked_objects = tracker.update(rects)

        # Match new bounding boxes in this frame to the tracker's IDs
        for (x, y, w, h) in rects:
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))

            matched_id = -1
            min_dist = 9999.0
            for (objectID, centroid) in tracked_objects.items():
                dist = np.linalg.norm(np.array([cX, cY]) - np.array(centroid))
                if dist < min_dist and dist < 20.0:
                    min_dist = dist
                    matched_id = objectID

            if matched_id != -1:
                if matched_id not in id_history:
                    id_history[matched_id] = []
                # Append metrics: (width, height, area, centroid_X)
                id_history[matched_id].append((w, h, w * h, cX))

    cap.release()

    # Calculate average metrics for each tracked ID
    id_stats = {}
    for objectID, history in id_history.items():
        if len(history) < 15: # Filter out short-lived tracking fragments (less than 15 frames)
            continue
        
        widths = [h[0] for h in history]
        heights = [h[1] for h in history]
        areas = [h[2] for h in history]
        cXs = [h[3] for h in history]

        id_stats[objectID] = {
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "avg_area": float(np.mean(areas)),
            "avg_x": float(np.mean(cXs)),
            "frames_tracked": len(history)
        }

    return id_stats


def main():
    print("Finding all videos...")
    all_videos = sorted(list(ROOT_DIR.glob("**/*.mp4")))
    print(f"Total videos found: {len(all_videos)}")

    # Sample 100 random videos
    sampled_videos = random.sample(all_videos, min(100, len(all_videos)))
    print(f"Sampled {len(sampled_videos)} videos for analysis.")

    # Prepare CSV headers
    csv_headers = [
        "video_path", "group", "track_id", "frames_tracked",
        "avg_width", "avg_height", "avg_area", "avg_x",
        "relative_size_rank", "is_smallest_in_video", "is_left_side"
    ]

    print(f"Starting tracking and data collection. Writing to {CSV_PATH}...")

    # Statistics to help answer the user's question
    total_valid_videos = 0
    smallest_is_left_count = 0
    smallest_is_right_count = 0
    largest_is_left_count = 0
    largest_is_right_count = 0

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for idx, video_path in enumerate(sampled_videos):
            rel_path = video_path.relative_to(ROOT_DIR).as_posix()
            group = "ASD" if "/ASD/" in video_path.as_posix() or "\\ASD\\" in video_path.as_posix() else "TD"
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/100 videos...")

            # Run tracking
            id_stats = analyze_video(video_path)
            if not id_stats:
                continue

            # Sort tracked IDs by average area to assign relative size ranks
            # Smallest tracked person has rank 1, second smallest has rank 2, etc.
            sorted_ids = sorted(id_stats.items(), key=lambda item: item[1]["avg_area"])
            
            # Record sizes for analysis if there are at least 2 tracked people in the video
            if len(sorted_ids) >= 2:
                total_valid_videos += 1
                smallest_id, smallest_data = sorted_ids[0]
                largest_id, largest_data = sorted_ids[-1]

                # Check spatial side of the smallest person (relative to center x=160 in 320x240 frame)
                if smallest_data["avg_x"] < 160:
                    smallest_is_left_count += 1
                else:
                    smallest_is_right_count += 1

                # Check spatial side of the largest person
                if largest_data["avg_x"] < 160:
                    largest_is_left_count += 1
                else:
                    largest_is_right_count += 1

            for rank, (objectID, stats) in enumerate(sorted_ids, start=1):
                is_smallest = 1 if rank == 1 else 0
                is_left = 1 if stats["avg_x"] < 160 else 0
                
                writer.writerow({
                    "video_path": rel_path,
                    "group": group,
                    "track_id": objectID,
                    "frames_tracked": stats["frames_tracked"],
                    "avg_width": round(stats["avg_width"], 2),
                    "avg_height": round(stats["avg_height"], 2),
                    "avg_area": round(stats["avg_area"], 2),
                    "avg_x": round(stats["avg_x"], 2),
                    "relative_size_rank": rank,
                    "is_smallest_in_video": is_smallest,
                    "is_left_side": is_left
                })

    print("\n" + "=" * 50)
    print("DATA ANALYSIS COMPLETE!")
    print(f"CSV report written to: {CSV_PATH}")
    print("=" * 50)
    
    if total_valid_videos > 0:
        print(f"\nStatistical insights from {total_valid_videos} videos with multi-person tracking:")
        print(f"  - Smallest person's average X-position:")
        print(f"      * Seated on the Left (<160px):  {smallest_is_left_count} times ({100.0 * smallest_is_left_count / total_valid_videos:.1f}%)")
        print(f"      * Seated on the Right (>=160px): {smallest_is_right_count} times ({100.0 * smallest_is_right_count / total_valid_videos:.1f}%)")
        print(f"  - Largest person's average X-position:")
        print(f"      * Seated on the Left (<160px):  {largest_is_left_count} times ({100.0 * largest_is_left_count / total_valid_videos:.1f}%)")
        print(f"      * Seated on the Right (>=160px): {largest_is_right_count} times ({100.0 * largest_is_right_count / total_valid_videos:.1f}%)")
        
        print("\nInterpretation for child seating determination:")
        print("  - If the child is usually seated on the LEFT, and the smallest tracked person is usually on the left,")
        print("    this confirms size correlates directly with age/role in this ADOS dataset.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
