import pandas as pd
import numpy as np

BASELINE_CSV = r'c:/asd_project/outputs/features/features.csv'
PHASE1_CSV = r'c:/asd_project/outputs/features/features_v2_phase1.csv'
REPORT_MD = r'c:/asd_project/FFT_ACF_DEBUG_REPORT.md'

# Load dataframes
baseline_df = pd.read_csv(BASELINE_CSV)
phase1_df = pd.read_csv(PHASE1_CSV)

# Count total videos (rows) processed
total_videos = len(baseline_df)

# Identify new feature columns
fft_cols = ['fft_peak_freq', 'fft_power_ratio', 'fft_entropy']
acf_cols = ['acf_lag1', 'acf_decay', 'acf_entropy']
new_cols = fft_cols + acf_cols

# Count valid (non-NaN) per new feature
valid_fft = phase1_df[fft_cols].notna().all(axis=1).sum()
valid_acf = phase1_df[acf_cols].notna().all(axis=1).sum()

# Rows with any NaN in new features
nan_rows = phase1_df[new_cols].isnull().any(axis=1).sum()

# Reasons for NaNs per column
nan_counts = phase1_df[new_cols].isnull().sum()

# Rows removed during cleaning in training script (train_mask based on split)
train_mask = baseline_df['split'] == 'train'
X_train_base = baseline_df[train_mask].drop(columns=['unique_video_id','video_id','label','split','dataset_path'])
# Align with phase1 features
train_phase1 = phase1_df.loc[train_mask]
# Combine base and new features
X_train = pd.concat([X_train_base, train_phase1[fft_cols + acf_cols].reset_index(drop=True)], axis=1)
# Rows with any NaN after merge
train_nan_rows = X_train.isnull().any(axis=1).sum()

# Generate report
lines = []
lines.append('# FFT/ACF Debug Report')
lines.append('')
lines.append(f'Raw videos (baseline rows): {total_videos}')
lines.append(f'Videos with valid FFT features (all three non‑NaN): {valid_fft}')
lines.append(f'Videos with valid ACF features (all three non‑NaN): {valid_acf}')
lines.append(f'Rows with any NaN in new features: {nan_rows}')
lines.append('')
lines.append('## NaN counts per new feature')
for col, cnt in nan_counts.items():
    lines.append(f'- {col}: {cnt} NaN values')
lines.append('')
lines.append(f'Rows removed during training cleaning (after merge): {train_nan_rows}')
lines.append('')
lines.append('## Example failing rows')
example_fail = phase1_df[new_cols].isnull().any(axis=1)
if example_fail.any():
    sample = phase1_df[example_fail].head(3)
    lines.append('```')
    lines.append(sample.to_csv(index=False))
    lines.append('```')
else:
    lines.append('No failing rows found.')

with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Debug report written to', REPORT_MD)
