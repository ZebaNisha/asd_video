# LSTM Model Evaluation and Feature Ablation Study
This report compares a PyTorch LSTM sequence model trained under two configurations:
1. **Stage 1 (Absolute Features)**: Includes coordinates, bounding box sizes, and raw speeds.
2. **Stage 2 (Scale-Invariant Features)**: Restricts features to coordinate-free and scale-normalized kinematics (aspect ratio, normalized speed, angles).

## Summary of Performance
| Metric | Stage 1 (Absolute) | Stage 2 (Scale-Invariant) |
| :--- | :---: | :---: |
| **Test Accuracy** | 0.5243 | 0.5274 |
| **Precision** | 0.5238 | 0.5404 |
| **Recall** | 0.5323 | 0.3648 |
| **F1-Score** | 0.5280 | 0.4355 |
| **ROC-AUC** | 0.5306 | 0.5339 |

## Confusion Matrices

### Stage 1 (Absolute)
```
[[2454 2299]
 [2222 2529]]
```

### Stage 2 (Scale-Invariant)
```
[[3279 1474]
 [3018 1733]]
```

## Analysis
- **Absolute Features (Stage 1)** capture the child's absolute size and location. While this might aid in fitting training distributions, it often generalizes poorly if train and test environments differ (distribution shift).
- **Scale-Invariant Features (Stage 2)** drop all camera distance/bounding box size biases, forcing the model to learn pure motion dynamics. If Stage 2 matches or exceeds Stage 1, it confirms that raw coordinates and absolute sizes are introducing negative bias/generalization error.