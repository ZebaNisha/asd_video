import pandas as pd
import os

csv_path = os.path.join('outputs', 'features', 'labeled_features.csv')

df = pd.read_csv(csv_path)

print('Rows:', len(df))
print('unique_video_id conflicts:', (df.groupby('unique_video_id')['label'].nunique() > 1).sum())
print('unique unique_video_ids:', df['unique_video_id'].nunique())
print('duplicate unique_video_ids:', len(df) - df['unique_video_id'].nunique())
print('\nSample rows:')
print('label distribution:', df['label'].value_counts())
