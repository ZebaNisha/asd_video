import pandas as pd

csv_path = r'C:/asd_project/outputs/features/features.csv'

df = pd.read_csv(csv_path)

# Count NaNs in label column
nan_labels = df['label'].isna()
print('Total rows:', len(df))
print('NaN label rows:', nan_labels.sum())
if nan_labels.any():
    print('Rows with NaN labels:')
    print(df[nan_labels][['unique_video_id', 'video_id', 'label']].head().to_string(index=False))
else:
    print('No NaN labels found.')
