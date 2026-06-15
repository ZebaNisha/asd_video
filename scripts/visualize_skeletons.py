"""
Skeleton Video Analyzer & Visualizer
=====================================
This script analyzes a skeleton keypoint video (anonymized OpenPose stick figures on a black background),
detects and counts each distinct skeleton in each frame, draws bounding boxes around them,
and saves the annotated output to a new video file.

It uses standard OpenCV morphological operations and connected component analysis.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Default sample video from the dataset
DEFAULT_VIDEO = r"c:\asd_project\autism_data_anonymized\autism_data_anonymized\training_set\ASD\Subj_10_part_1.mp4"


def analyze_and_visualize(video_path: str, output_path: str, min_area: int = 150) -> None:
    print(f"Reading video: {video_path}")
    
    # 1. OpenCV Operation: cv2.VideoCapture
    # Explanation: Opens the video file for reading. It connects to the video stream and provides
    # access to frames and properties (like width, height, FPS, and frame count).
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties: Resolution = {width}x{height}, FPS = {fps}, Total Frames = {total_frames}")

    # 2. OpenCV Operation: cv2.VideoWriter_fourcc
    # Explanation: Defines the codec used to compress the frames. 'mp4v' is a standard MPEG-4 codec.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 3. OpenCV Operation: cv2.VideoWriter
    # Explanation: Initializes a video writer object to save our annotated frames to a new video file.
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 4. OpenCV Operation: cv2.getStructuringElement
    # Explanation: Creates a structuring element (kernel) of a specified shape and size.
    # We use a rectangular kernel of size 21x21. This will be used in the dilation step to
    # merge disjoint skeleton bones (e.g. gaps between limbs, head, and torso) into a single blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    frame_num = 0
    while True:
        # 5. OpenCV Operation: cap.read
        # Explanation: Decodes and returns the next frame from the video stream.
        # 'ret' is a boolean indicating success, and 'frame' is the frame image (numpy array).
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        frame_num += 1
        
        # 6. OpenCV Operation: cv2.cvtColor
        # Explanation: Converts an image from one color space to another.
        # Here we convert the frame from BGR (color) to Grayscale. Since the skeletons are colored
        # lines on a black background, converting to grayscale preserves the shape structure while
        # reducing the data to a single channel (essential for thresholding).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 7. OpenCV Operation: cv2.threshold
        # Explanation: Applies a fixed-level threshold to each pixel.
        # Any pixel with grayscale value > 10 (skeleton lines) is set to 255 (white),
        # and all other pixels (black background) are set to 0. This yields a clean binary mask.
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 8. OpenCV Operation: cv2.dilate
        # Explanation: Dilates the white regions in the binary image by applying the structuring element.
        # Dilation acts as a 'grow' operation. By expanding the white lines, it connects the separated
        # skeleton keypoint components (head, arms, spine, legs) into a single contiguous blob.
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 9. OpenCV Operation: cv2.connectedComponentsWithStats
        # Explanation: Computes the connected components labeled image of a binary image.
        # It clusters contiguous groups of white pixels into labeled blobs. It returns:
        # - num_labels: Total number of blobs found (including the black background which is label 0).
        # - labels: A labeled grid where each pixel value is the ID of its cluster.
        # - stats: A matrix containing statistics for each label (x, y, width, height, and area in pixels).
        # - centroids: The (x, y) center coordinates of each label.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
        
        # Filter and track valid skeleton blobs
        skeleton_count = 0
        
        # Loop starts at 1 to skip the background component (label 0)
        for i in range(1, num_labels):
            # Extract statistics for component i
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out tiny components (which are just noise or stray keypoint dots)
            if area < min_area:
                continue
                
            skeleton_count += 1
            
            # Bounding box color (Red in BGR)
            bbox_color = (0, 0, 255) 
            
            # 10. OpenCV Operation: cv2.rectangle
            # Explanation: Draws a rectangle on the image frame given the top-left coordinate (x,y),
            # bottom-right coordinate (x+w, y+h), color, and line thickness.
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
            
            # Draw skeleton ID text
            label_text = f"Skel {skeleton_count}"
            # 11. OpenCV Operation: cv2.putText
            # Explanation: Renders a text string on the frame. We specify the coordinates,
            # font face, scale, color, thickness, and line type.
            cv2.putText(
                frame, 
                label_text, 
                (x, max(15, y - 5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (0, 255, 0),  # Green text
                1, 
                cv2.LINE_AA
            )
            
        # Draw frame information overlay
        info_text = f"Frame: {frame_num}/{total_frames} | Count: {skeleton_count}"
        cv2.putText(
            frame, 
            info_text, 
            (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255),  # White text
            1, 
            cv2.LINE_AA
        )
        
        # 12. OpenCV Operation: out.write
        # Explanation: Writes the annotated frame to the output video stream.
        out.write(frame)

    # Release resources
    # 13. OpenCV Operation: cap.release & out.release
    # Explanation: Deallocates the video capture and writer objects and closes the files.
    cap.release()
    out.release()
    
    print("--------------------------------------------------")
    print(f"Processing Complete!")
    print(f"Annotated video saved successfully to: {output_path}")
    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Analyze a skeleton video and draw bounding boxes around figures.")
    parser.add_argument(
        "--video", 
        type=str, 
        default=DEFAULT_VIDEO, 
        help="Path to the input skeleton video file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="annotated_output.mp4", 
        help="Path to save the output annotated video."
    )
    parser.add_argument(
        "--min-area", 
        type=int, 
        default=150, 
        help="Minimum pixel area to consider a blob as a valid skeleton figure."
    )
    
    args = parser.parse_args()
    
    # Resolve relative paths or make sure target folder exists
    video_path = str(Path(args.video).resolve())
    output_path = str(Path(args.output).resolve())
    
    if not os.path.exists(video_path):
        print(f"Error: Input video does not exist: {video_path}")
        print(f"Please specify a valid path with --video")
        sys.exit(1)
        
    analyze_and_visualize(video_path, output_path, args.min_area)


if __name__ == "__main__":
    main()
