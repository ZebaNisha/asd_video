import argparse
import cv2
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(description='Generate stickman video from raw video')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output stickman video')
    parser.add_argument('--model', default=r'c:/asd_project/pose_landmarker_lite.task', help='Path to MediaPipe pose model')
    args = parser.parse_args()

    INPUT_VIDEO = args.input
    OUTPUT_VIDEO = args.output
    MODEL_PATH = args.model

    # Mediapipe tasks API imports
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    # Create PoseLandmarker options
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Initialize PoseLandmarker
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (11, 12), (12, 13), (13, 14),
        (15, 16), (16, 17), (17, 18), (18, 19),
        (0, 9), (0, 10), (0, 11), (0, 12),
        (23, 24), (23, 25), (24, 26), (25, 27),
        (27, 29), (28, 30), (29, 31), (30, 32),
    ]

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((frame_idx / fps) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for start_idx, end_idx in pose_connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * width), int(start.y * height)
                x2, y2 = int(end.x * width), int(end.y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    pose_landmarker.close()
    print(f"Stickman video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
