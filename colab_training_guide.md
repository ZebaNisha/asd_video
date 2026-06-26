# Google Colab Training Guide: VGG16 + LSTM

This guide outlines how to offload the VGG16 feature extraction and Bidirectional LSTM training to Google Colab to take advantage of free/low-cost GPU accelerators (T4, L4, or A100), reducing processing time from hours to minutes.

---

## Step 1: Package Your Local Workspace
Run the following PowerShell command in your local project root (`c:\asd_project`) to zip the raw stickman videos, child tracking sequences, scripts, and label files:

```powershell
Compress-Archive -Path "autism_data_anonymized", "outputs\child_sequences", "outputs\features\labeled_features.csv", "scripts" -DestinationPath "asd_colab_package.zip" -Force
```
*Note: This creates a single archive `asd_colab_package.zip`. Since the stickman videos are small (320x240 resolution, 5-second duration), the file size is highly manageable.*

---

## Step 2: Upload to Google Drive
Upload `asd_colab_package.zip` directly to the root of your Google Drive.

---

## Step 3: Run on Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and set the runtime to a GPU:
   - Go to **Runtime** > **Change runtime type**.
   - Select **T4 GPU** (or L4/A100 if available) under *Hardware accelerator*.
3. Copy-paste and run the following cells in your notebook:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Extract the Package
```python
import os
import shutil

# Copy and unzip workspace package from Drive
zip_path = '/content/drive/MyDrive/asd_colab_package.zip'
extract_path = '/content/asd_project'

if os.path.exists(zip_path):
    print("Extracting workspace...")
    shutil.unpack_archive(zip_path, extract_path)
    os.chdir(extract_path)
    print(f"Extraction completed. Current directory: {os.getcwd()}")
else:
    print("Error: asd_colab_package.zip not found in Google Drive root!")
```

### Cell 3: Setup Dependencies
```python
!pip install opencv-python-headless tensorflow pandas numpy scikit-learn
```

### Cell 4: Generate Full Metadata (All Subjects & Videos)
```python
# Create a metadata file that includes all subjects and all of their videos
!python scripts/create_balanced_subset_metadata.py --subjects-per-group 200 --output outputs/vgg16_lstm/full_dataset_metadata.csv
```

### Cell 5: Extract Features on GPU
```python
# Extract VGG16 features utilizing the Colab GPU
!python scripts/extract_child_vgg16_features.py --metadata outputs/vgg16_lstm/full_dataset_metadata.csv --child-seq-dir outputs/child_sequences --out-dir outputs/vgg16_lstm/full_dataset --max-frames 30 --batch-size 128 --force
```

### Cell 6: Train Bidirectional LSTM on Full Dataset
```python
# Train the sequence classifier on the full dataset features
!python scripts/train_child_vgg16_lstm.py --data-dir outputs/vgg16_lstm/full_dataset --epochs 30 --batch-size 32 --hidden-dim 64 --dropout 0.5
```

### Cell 7: Compress and Save Results Back to Google Drive
```python
import shutil

# Compress the outputs and model weights to save back to Drive
shutil.make_archive('/content/drive/MyDrive/asd_colab_results', 'zip', 'outputs/vgg16_lstm/full_dataset')
print("Results successfully saved to Google Drive as asd_colab_results.zip!")
```

---

## Step 4: Download Results Locally
1. Download `asd_colab_results.zip` from your Google Drive.
2. Extract the files into your local directory at `c:\asd_project\outputs\vgg16_lstm\full_dataset\`.
3. The folder will now contain:
   - `child_vgg16_lstm.keras` (Trained model weights)
   - `child_vgg16_lstm_report.json` (Detailed model evaluation metrics)
   - `vgg16_child_train.npz` & `vgg16_child_test.npz` (Extracted sequence features)
