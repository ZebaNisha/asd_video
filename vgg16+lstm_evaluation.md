# VGG16 + LSTM Model Evaluation Report

This report documents the performance evaluation of the **VGG16 + Bidirectional LSTM** deep sequence models. These models are trained on sequence features extracted by a pre-trained VGG16 model from bounding box image sequences containing the tracked child.

---

## 1. Full Dataset Experiment (`subset_allsubjects_20videos`)
This experiment consists of 120 subjects (40 train, 80 test) with 20 videos per subject. 

### Dataset Configuration
- **Train Shape**: `[1600, 30, 512]` (1,600 clips, 30 frames, 512 VGG16 features)
- **Test Shape**: `[1600, 30, 512]` (1,600 clips, 30 frames, 512 VGG16 features)
- **Class Balance (Train & Test)**: Balanced (800 ASD clips, 800 TD clips)
- **Subject Balance**: 40 unique train subjects (all ASD), 80 unique test subjects (40 ASD, 40 TD) to prevent subject leakage.

### Model Hyperparameters
- **Architecture**: Bidirectional LSTM
- **Hidden Dimension**: 64
- **Dropout / Recurrent Dropout**: 0.5 / 0.2
- **Batch Size**: 16
- **Learning Rate / L2**: 0.0001 / 0.0001
- **Patience**: 6

### Clip-level Evaluation
Clip-level metrics represent predictions made individually for each 30-frame clip.

| Split | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Train** | 0.6013 | 0.5711 | 0.8138 | 0.6711 | 0.6731 |
| **Test** | 0.5394 | 0.5267 | 0.7763 | 0.6276 | 0.5246 |

#### Test Confusion Matrix
```
[[242  558]   (True TD, False ASD)
 [179  621]]  (False TD, True ASD)
```

### Subject-level Evaluation (Official Majority Voting)
Subject-level predictions are aggregated by majority voting of clip predictions for each subject.

| Split | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Train** | 0.9750 | 1.0000 | 0.9750 | 0.9873 | NaN |
| **Test** | 0.5250 | 0.5132 | 0.9750 | 0.6724 | 0.5931 |

#### Test Subject Confusion Matrix
```
[[ 3  37]   (True TD, False ASD)
 [ 1  39]]   (False TD, True ASD)
```

### Detailed Feature & Complexity Analysis
- **Trainable Parameters**: `303,745` (Bidirectional LSTM)
- **Train Feature Variance** (across all VGG16 features and frames):
  - Mean variance: `2.797197`
  - Std variance: `9.161541`
  - Min variance: `0.000160`
  - Max variance: `119.990410`
- **Test Feature Variance** (across all VGG16 features and frames):
  - Mean variance: `2.930807`
  - Std variance: `9.712844`
  - Min variance: `0.000079`
  - Max variance: `123.015137`

---

## 2. Pilot Subset Experiment (`subset_2subjects_30videos`)
An initial run evaluated on a smaller subset containing 30 videos from 2 subjects.

### Dataset Configuration
- **Train Shape**: `[120, 30, 512]`
- **Test Shape**: `[120, 30, 512]`
- **Class Balance (Train & Test)**: Balanced (60 ASD clips, 60 TD clips)

### Performance Metrics

| Split | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Train** | 0.9917 | 0.9836 | 1.0000 | 0.9917 |
| **Test (Final)** | 0.6583 | 0.7209 | 0.5167 | 0.6019 |

#### Train Confusion Matrix
```
[[59  1]
 [ 0 60]]
```

#### Test Confusion Matrix
```
[[48  12]
 [29  31]]
```

---

## 3. Medium Dataset Experiment (`subset_5subjects`)
This experiment consists of 5 subjects per group and aggregates a medium-sized subset. Three architectural variants were evaluated.

### Dataset Configuration
- **Train Shape**: `[1208, 30, 512]` (603 TD, 605 ASD)
- **Test Shape**: `[1127, 30, 512]` (524 TD, 603 ASD)

### Architectural Variants & Results

#### A. Baseline Configuration (`child_vgg16_lstm_report.json`)
- **Accuracy**: `0.5430` (54.30%)
- **Precision**: `0.5952`
- **Recall**: `0.4561`
- **F1-Score**: `0.5164`
- **Test Confusion Matrix**:
  ```
  [[337 187]
   [328 275]]
  ```

#### B. Early Stopping Configuration (`child_vgg16_lstm_earlystop_report.json`)
- **Hyperparameters**: hidden_dim: 128, dropout: 0.3, recurrent_dropout: 0.0, L2: 0.0, batch_size: 16
- **Accuracy**: `0.5395` (53.95%)
- **Precision**: `0.5695`
- **Recall**: `0.5705`
- **F1-Score**: `0.5700`
- **Test Confusion Matrix**:
  ```
  [[264 260]
   [259 344]]
  ```

#### C. Regularized Configuration (`child_vgg16_lstm_regularized_report.json`)
- **Hyperparameters**: hidden_dim: 64, dropout: 0.5, recurrent_dropout: 0.2, L2: 0.0001, batch_size: 32
- **Accuracy**: `0.5359` (53.59%)
- **Precision**: `0.6053`
- **Recall**: `0.3814`
- **F1-Score**: `0.4680`
- **Test Confusion Matrix**:
  ```
  [[374 150]
   [373 230]]
  ```

---

## 4. Small Dataset Experiment (`subset_2subjects_10videos`)
A small configuration consisting of 2 subjects per group and 10 videos per subject.

### Dataset Configuration
- **Train Shape**: `[40, 30, 512]` (20 TD, 20 ASD)
- **Test Shape**: `[40, 30, 512]` (20 TD, 20 ASD)

### Performance Metrics (`child_vgg16_lstm_report.json`)
- **Accuracy**: `0.6750` (67.50%)
- **Precision**: `0.7333`
- **Recall**: `0.5500`
- **F1-Score**: `0.6286`
- **Test Confusion Matrix**:
  ```
  [[16  4]
   [ 9 11]]
  ```

---

> [!IMPORTANT]
> **Comparison with Baseline Comparison Report:**
> The file [baseline_comparison_report.md](file:///c:/asd_project/reports/baseline_comparison_report.md) contains a table listing an accuracy of `0.8900` (clip) / `0.9000` (subject) for a Bidirectional LSTM. Those values are placeholders or target/idealized metrics and do not reflect the actual trained model outputs (`0.5394` clip / `0.5250` subject accuracy) recorded in the experiment output JSON.
