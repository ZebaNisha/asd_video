import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
manifest_path = 'c:/asd_project/reports/dataset_manifest.csv'
output_dir = 'c:/asd_project/reports/plots/bbox_features'
os.makedirs(output_dir, exist_ok=True)

# Load manifest
df = pd.read_csv(manifest_path)
# The bbox features are likely stored in separate feature files per video; however, summary statistics are in feature_importance file.
# For illustration, we will create synthetic data: assume the features are present in the manifest as columns (if not, this script will skip).
features = ['mean_height', 'mean_width', 'mean_area', 'min_area', 'max_area']
# Check which features exist
existing = [f for f in features if f in df.columns]
if not existing:
    print('No bbox feature columns found in manifest.')
else:
    for feat in existing:
        plt.figure(figsize=(8,4))
        sns.boxplot(y=df[feat].dropna())
        plt.title(f'Boxplot of {feat}')
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feat}_boxplot.png'))
        plt.close()
        plt.figure(figsize=(8,4))
        sns.histplot(df[feat].dropna(), kde=True, bins=30)
        plt.title(f'Histogram of {feat}')
        plt.xlabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feat}_histogram.png'))
        plt.close()
    print('Plots saved to', output_dir)
