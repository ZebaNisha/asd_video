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

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
import os

# duplicate import removed
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="Detect skeletons in a video and draw bounding boxes.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to save annotated video (optional). If omitted, a file with '_bbox.mp4' suffix is created next to input.")
    parser.add_argument("--csv", help="Path to save detection CSV (optional). If omitted, a file with '_detections.csv' suffix is created next to input.")
    parser.add_argument("--display", action="store_true", help="Show video while processing (default: True if output not specified)")
    parser.add_argument("--unique-id", help="Unique identifier for video (e.g., ASD_Subj_10_part_1). If not provided, defaults to input video stem.")
    parser.add_argument("--model", required=True, help="Path to MediaPipe pose landmarker .task model file")
    return parser.parse_args()


def main():
    args = get_args()
    video_path = Path(args.input)
    video_id = args.unique_id if args.unique_id else video_path.stem
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")


    # Determine output paths
    csv_path = Path(args.csv) if args.csv else video_path.with_name(f"{video_path.stem}_detections.csv")
    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_bbox.mp4")

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
    # Initialize MediaPipe Pose solution
    # Initialize MediaPipe Tasks PoseLandmarker
    base_options = BaseOptions(model_asset_path=args.model)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for MP4 output
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run MediaPipe Tasks pose detection on the current frame (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        if result.pose_landmarks:
            # Assuming single person (first detected pose)
            landmarks = result.pose_landmarks[0]
            xs = [lm.x * width for lm in landmarks]
            ys = [lm.y * height for lm in landmarks]
            # Compute bounding box from landmarks
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            x = int(min_x)
            y = int(min_y)
            w = int(max_x - min_x)
            h = int(max_y - min_y)
            area = w * h
            centroid_x = x + w / 2.0
            centroid_y = y + h / 2.0
            csv_writer.writerow([video_id, frame_idx, 0, x, y, w, h, area, centroid_x, centroid_y])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Write annotated frame to output video
            writer.write(frame)
        else:
            csv_writer.writerow([video_id, frame_idx, 0, 0, 0, 0, 0, 0, 0, 0])
            # Write original frame when no detection
            writer.write(frame)

        frame_idx += 1

    # Release resources: VideoCapture, VideoWriter, and close any OpenCV windows.
    cap.release()
    writer.release()
    csv_file.close()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Annotated video saved to: {output_path}")
    print(f"Detections CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
