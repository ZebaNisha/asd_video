# detect_skeletons.py
"""
Skeleton detection and bounding‑box annotation script.

Usage:
    python detect_skeletons.py --input path/to/video.mp4 [--output path/to/output.mp4]

The script reads an OpenPose-style stickman video with OpenCV, detects skeleton
blobs by thresholding the rendered lines, computes a tight bounding box around
each detected skeleton, draws the boxes on each frame, and saves an annotated
video.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Detect skeletons in a video and draw bounding boxes.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to save annotated video (optional). If omitted, a file with '_bbox.mp4' suffix is created next to input.")
    parser.add_argument("--csv", help="Path to save detection CSV (optional). If omitted, a file with '_detections.csv' suffix is created next to input.")
    parser.add_argument("--display", action="store_true", help="Show video while processing (default: True if output not specified)")
    parser.add_argument("--unique-id", help="Unique identifier for video (e.g., ASD_Subj_10_part_1). If not provided, defaults to input video stem.")
    return parser.parse_args()


def main():
    args = get_args()
    video_path = Path(args.input)
    video_id = args.unique_id if args.unique_id else video_path.stem
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Determine output paths using centralized config
    from path_config import BBOX_VIDEOS_DIR, DETECTIONS_DIR
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = BBOX_VIDEOS_DIR / f"{video_id}_bbox.mp4"
    # CSV output path (optional)
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = DETECTIONS_DIR / f"{video_id}_detections.csv"
    # Open CSV for writing
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["video_id", "frame_number", "skeleton_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_area", "centroid_x", "centroid_y"])

    # OpenCV: VideoCapture reads frames from the video file.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Retrieve video properties – required for writing output with same codec/frame size.
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for MP4 output
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Simple binary threshold (assumes white skeleton on black background)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        # Morphological closing to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Find external contours (each skeleton blob)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skeleton_id = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # filter tiny noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            centroid_x = x + w / 2.0
            centroid_y = y + h / 2.0
            # Write detection row to CSV
            csv_writer.writerow([video_id, frame_idx, skeleton_id, x, y, w, h, area, centroid_x, centroid_y])
            # Draw bounding box and ID on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Skeleton {skeleton_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            skeleton_id += 1

        # Write annotated frame to output video.
        writer.write(frame)

        # Optional live display using OpenCV imshow (creates a window).
        if args.display:
            cv2.imshow("Skeleton Detection", frame)
            # WaitKey(1) shows the frame for ~1 ms; press 'q' to quit early.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    # Release resources: VideoCapture, VideoWriter, and close any OpenCV windows.
    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_path}")
    print(f"Detections CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
