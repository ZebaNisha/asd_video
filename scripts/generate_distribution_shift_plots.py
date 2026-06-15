import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
manifest_path = 'c:/asd_project/reports/dataset_manifest.csv'
split_path = 'c:/asd_project/reports/split.csv'
output_dir = 'c:/asd_project/reports/plots/distribution_shift'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(manifest_path)
split_df = pd.read_csv(split_path)
# Assume 'video_id' column matches between manifest and split
merged = pd.merge(df, split_df[['video_id', 'split']], on='video_id', how='left')

features = ['mean_height', 'mean_width', 'mean_area', 'min_area', 'max_area']
for feat in features:
    if feat not in merged.columns:
        continue
    plt.figure(figsize=(8,4))
    sns.histplot(data=merged, x=feat, hue='split', element='step', stat='density', common_norm=False, bins=30, kde=True)
    plt.title(f'Distribution of {feat} in Train vs Test')
    plt.xlabel(feat)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feat}_train_test_histogram.png'))
    plt.close()
    # Boxplot comparison
    plt.figure(figsize=(6,4))
    sns.boxplot(x='split', y=feat, data=merged)
    plt.title(f'Boxplot of {feat} by Split')
    plt.xlabel('Split')
    plt.ylabel(feat)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feat}_train_test_boxplot.png'))
    plt.close()
print('Distribution shift plots saved to', output_dir)
