"""
Centroid-Based Multi-Object Tracker from Scratch
================================================
This script implements a custom CentroidTracker class from scratch and integrates
it with the skeleton detection pipeline. It reads a video, detects skeletons,
tracks them frame-by-frame with persistent IDs, and writes the annotated video.

It contains no external tracking libraries, providing a pure mathematical and
logical implementation of centroid tracking.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np

# Default sample video from the dataset
DEFAULT_VIDEO = r"c:\asd_project\autism_data_anonymized\autism_data_anonymized\training_set\ASD\Subj_10_part_1.mp4"


class CentroidTracker:
    def __init__(self, maxDisappeared: int = 10, maxDistance: float = 75.0):
        """
        Initializes the tracker state.
        
        Parameters:
        - maxDisappeared: The number of consecutive frames an object can go unmatched
                          before it is deregistered (removed).
        - maxDistance: The maximum Euclidean distance allowed between a tracked object
                       and a new detection to consider them the same object.
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()       # Dict mapping object ID -> centroid coordinates (cX, cY)
        self.disappeared = OrderedDict()   # Dict mapping object ID -> consecutive frames unseen
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid: np.ndarray) -> None:
        """Registers a new object with the next available ID."""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID: int) -> None:
        """Deregisters (removes) an object that has been lost."""
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects: list[tuple[int, int, int, int]]) -> OrderedDict:
        """
        Updates the tracker with the bounding boxes detected in the current frame.
        
        Parameters:
        - rects: List of bounding boxes in format [(x, y, w, h), ...]
        
        Returns:
        - OrderedDict of active tracked objects {objectID: (cX, cY), ...}
        """
        # Step 1: Handle the case where no bounding boxes were detected
        if len(rects) == 0:
            # Increment the disappeared count for all active tracked objects
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # If an object has been missing for too many frames, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            return self.objects

        # Step 2: Calculate the centroid of each incoming bounding box
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            # Centroid formula: cX = x + w/2, cY = y + h/2
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            inputCentroids[i] = (cX, cY)

        # Step 3: If we are not currently tracking any objects, register all input centroids
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
                
        # Step 4: Otherwise, match existing tracked objects with new input detections
        else:
            # Retrieve active tracked IDs and their current centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the pairwise Euclidean distance matrix between tracked objects (N) and input detections (M)
            N = len(objectCentroids)
            M = len(inputCentroids)
            D = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    # Euclidean Distance formula: d = sqrt((x1 - x2)^2 + (y1 - y2)^2)
                    D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))

            # To match objects, we perform a greedy assignment:
            # 1. Find the minimum distance in each row (existing object to closest detection)
            # 2. Sort the row indices based on their minimum distance to order matches by confidence
            rows = D.min(axis=1).argsort()
            
            # 3. Find the minimum column index for each sorted row to find the closest detection
            cols = D.argmin(axis=1)[rows]

            # Track which rows (tracked objects) and columns (detections) have been matched
            usedRows = set()
            usedCols = set()

            # Loop over the sorted (row, col) match indices
            for (row, col) in zip(rows, cols):
                # Ignore if we have already used this row or column
                if row in usedRows or col in usedCols:
                    continue

                # If the distance is greater than the threshold, do not match them
                if D[row, col] > self.maxDistance:
                    continue

                # Retrieve the objectID and update its centroid & reset disappeared count
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # Mark this row and column as used
                usedRows.add(row)
                usedCols.add(col)

            # Step 5: Handle unmatched tracked objects (some tracked objects went unseen)
            unusedRows = set(range(N)).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                
                # If they have been missing too long, remove them
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Step 6: Handle unmatched detections (these are newly appeared objects)
            unusedCols = set(range(M)).difference(usedCols)
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects


def track_video(video_path: str, output_path: str, min_area: int = 150) -> None:
    print(f"Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames.")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize our Custom Tracker
    # maxDisappeared=10 frames buffer, maxDistance=75px threshold
    tracker = CentroidTracker(maxDisappeared=10, maxDistance=75.0)
    
    # Kernel for skeleton dilation (connecting disjoint joints)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        frame_num += 1
        
        # 1. Image preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 2. Blob detection using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
        
        rects = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out tiny noise elements
            if area >= min_area:
                rects.append((x, y, w, h))

        # 3. Update the Centroid Tracker
        tracked_objects = tracker.update(rects)
        
        # 4. Draw bounding boxes, centroids, and IDs
        # We want to match the centroids returned by the tracker back to our bounding boxes
        # to draw the ID above each correct bounding box.
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            
            # Find the ID of the tracked object closest to this bounding box's centroid
            matched_id = -1
            min_dist = 9999.0
            
            for (objectID, centroid) in tracked_objects.items():
                dist = np.linalg.norm(np.array([cX, cY]) - np.array(centroid))
                if dist < min_dist and dist < 20.0: # Close threshold to prevent mismatch
                    min_dist = dist
                    matched_id = objectID
            
            # Draw Bounding Box (Blue)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw persistent ID label above the bounding box
            if matched_id != -1:
                label_text = f"ID: {matched_id}"
                cv2.putText(
                    frame, 
                    label_text, 
                    (x, max(15, y - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0),  # Green text
                    2, 
                    cv2.LINE_AA
                )
                
        # Draw all tracked centroids to visualize trajectories
        for (objectID, centroid) in tracked_objects.items():
            cX, cY = centroid
            # Draw centroid point (Red circle)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # Label the ID next to the centroid
            cv2.putText(
                frame, 
                f"ID {objectID}", 
                (cX + 8, cY - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (0, 0, 255), 
                1, 
                cv2.LINE_AA
            )

        # Draw frame information overlay
        info_text = f"Frame: {frame_num}/{total_frames} | Active: {len(tracked_objects)}"
        cv2.putText(
            frame, 
            info_text, 
            (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        out.write(frame)

    cap.release()
    out.release()
    print("--------------------------------------------------")
    print("Tracking Complete!")
    print(f"Tracked video saved successfully to: {output_path}")
    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Multi-object centroid tracker for skeleton figures.")
    parser.add_argument(
        "--video", 
        type=str, 
        default=DEFAULT_VIDEO, 
        help="Path to input skeleton video."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="tracked_output.mp4", 
        help="Path to save tracked output video."
    )
    parser.add_argument(
        "--min-area", 
        type=int, 
        default=150, 
        help="Minimum area of skeleton blob."
    )
    
    args = parser.parse_args()
    
    video_path = str(Path(args.video).resolve())
    output_path = str(Path(args.output).resolve())
    
    if not os.path.exists(video_path):
        print(f"Error: Input video does not exist: {video_path}")
        sys.exit(1)
        
    track_video(video_path, output_path, args.min_area)


if __name__ == "__main__":
    main()
