import cv2
import sys

video_path = r"C:\asd_project\autism_data_anonymized\autism_data_anonymized\testing_set\ASD\Subj_100_part_1.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video")
    sys.exit(1)

success, image = cap.read()
if success:
    cv2.imwrite(r"C:\Users\moham\.gemini\antigravity\brain\e2d7b2f6-fb8f-40ba-bf18-ad93781be4d9\scratch\frame1.jpg", image)
    print("Saved frame1.jpg")
else:
    print("Failed to read frame")

cap.release()
