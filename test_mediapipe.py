import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os

video_path = r"C:\asd_project\autism_data_anonymized\autism_data_anonymized\testing_set\ASD\Subj_100_part_1.mp4"
model_path = r"C:\asd_project\pose_landmarker_heavy.task"

print(f"Testing MediaPipe on: {video_path}")
if not os.path.exists(video_path):
    print("Video file does not exist.")
    sys.exit(1)

if not os.path.exists(model_path):
    print("Model file does not exist.")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video")
    sys.exit(1)

frame_count = 0
detected_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened() and frame_count < 150: # test 150 frames
    success, image = cap.read()
    if not success:
        break
        
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp_ms = int(frame_count * 1000 / fps) if fps > 0 else frame_count * 33
    
    # Process the image and find poses
    pose_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if pose_result.pose_landmarks:
        detected_count += 1
        if detected_count == 1:
            # Print sample data for the first detected frame
            print(f"Success on frame {frame_count}!")
            print(f"Nose landmark (x,y,z,visibility):")
            # Nose is landmark 0
            nose = pose_result.pose_landmarks[0][0]
            print(f"  x: {nose.x:.4f}, y: {nose.y:.4f}, z: {nose.z:.4f}, v: {nose.visibility:.4f}")
            print(f"Total landmarks detected: {len(pose_result.pose_landmarks[0])}")

    frame_count += 1

print(f"Tested {frame_count} frames. Landmarks detected in {detected_count} frames.")

cap.release()
detector.close()
