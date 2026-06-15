import os
import sys
import cv2
import pandas as pd
import numpy as np
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=None, help="Process only a specific subject (e.g. Subj_100)")
    parser.add_argument('--subjects', type=str, default=None, help="Comma‑separated list of subjects to process (overrides --subject)")
    parser.add_argument('--csv', type=str, default=r"C:\asd_project\outputs\features\features.csv", help="Metadata CSV")
    parser.add_argument('--out_dir', type=str, default=r"C:\asd_project\outputs\pose_sequences", help="Output directory")
    parser.add_argument('--model', type=str, default=r"C:\asd_project\pose_landmarker_heavy.task", help="MediaPipe model task file")
    parser.add_argument('--base_dir', type=str, default=r"C:\asd_project", help="Base directory for relative paths")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(args.csv)
    
    if args.subjects:
        subjects_list = [s.strip() for s in args.subjects.split(',')]
        # Filter rows where video_id starts with any of the selected subject prefixes
        mask = df['video_id'].apply(lambda vid: any(vid.startswith(f"{sub}_") for sub in subjects_list))
        df = df[mask]
        print(f"Filtered to {len(df)} videos for subjects {', '.join(subjects_list)}")
        if len(df) == 0:
            print("No videos found for the specified subjects.")
            sys.exit(1)
    elif args.subject:
        # Filter rows by subject
        # video_id is like Subj_100_part_1
        df = df[df['video_id'].str.startswith(f"{args.subject}_")]
        print(f"Filtered to {len(df)} videos for subject {args.subject}")
        if len(df) == 0:
            print("No videos found for this subject.")
            sys.exit(1)
    
    # Setup MediaPipe
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
        
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO)
    
    # Define columns for the output CSV
    columns = ['frame_idx']
    for i in range(33):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])
        
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Poses"):
        unique_id = row['video_id']
        video_rel_path = row['dataset_path']
        
        # Construct absolute path
        # Some paths might already be absolute, or relative to base_dir
        if os.path.isabs(video_rel_path):
            video_abs_path = video_rel_path
        else:
            video_abs_path = os.path.join(args.base_dir, video_rel_path)
            
        out_file = os.path.join(args.out_dir, f"{unique_id}_pose.csv")
        out_npy = os.path.join(args.out_dir, f"{unique_id}_pose.npy")
        
        # Skip if already exists
        if os.path.exists(out_file):
            continue
            
        if not os.path.exists(video_abs_path):
            print(f"Warning: Video not found at {video_abs_path}")
            continue
            
        cap = cv2.VideoCapture(video_abs_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_abs_path}")
            continue
            
        with vision.PoseLandmarker.create_from_options(options) as detector:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            
            rows = []
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = frame_count * 33
                
                pose_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    landmarks = pose_result.pose_landmarks[0]
                    row_data = [frame_count]
                    for lm in landmarks:
                        row_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                    rows.append(row_data)
                
                frame_count += 1
            
        cap.release()
        
        # Save to CSV and NPY
        if rows:
            data_array = np.array(rows)
            out_df = pd.DataFrame(data_array, columns=columns)
            out_df.to_csv(out_file, index=False)
            np.save(out_npy, data_array)
        else:
            # Create empty file so we don't retry failed ones endlessly
            pd.DataFrame(columns=columns).to_csv(out_file, index=False)
            np.save(out_npy, np.array([]))

if __name__ == '__main__':
    main()
