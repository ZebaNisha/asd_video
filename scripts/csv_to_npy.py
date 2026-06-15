import os
import glob
import pandas as pd
import numpy as np

pose_dir = r"C:\\asd_project\\outputs\\pose_sequences"
csv_files = glob.glob(os.path.join(pose_dir, "*.csv"))
for csv_path in csv_files:
    base, _ = os.path.splitext(csv_path)
    npy_path = base + ".npy"
    if os.path.exists(npy_path):
        continue
    df = pd.read_csv(csv_path)
    # If CSV is empty (only header), save empty array
    if df.empty:
        np.save(npy_path, np.array([]))
    else:
        np.save(npy_path, df.values)
    print(f"Saved {npy_path}")
