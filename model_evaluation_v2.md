# Model Evaluation V2

## Dynamics Feature Set
### Logistic Regression
- Accuracy: 0.4936
- Precision (weighted): 0.4932
- Recall (weighted): 0.4936
- F1-score (weighted): 0.4903
- ROC-AUC: 0.5161791652340573
- Confusion Matrix:
```
[[497 704]
 [515 691]]
```

### Random Forest
- Accuracy: 0.5206
- Precision (weighted): 0.5207
- Recall (weighted): 0.5206
- F1-score (weighted): 0.5201
- ROC-AUC: 0.5185711050630832
- Confusion Matrix:
```
[[663 538]
 [616 590]]
```

### XGBoost
- Accuracy: 0.5193
- Precision (weighted): 0.5193
- Recall (weighted): 0.5193
- F1-score (weighted): 0.5193
- ROC-AUC: 0.5206934381658181
- Confusion Matrix:
```
[[620 581]
 [576 630]]
```

## Periodicity Feature Set
### Logistic Regression
- Accuracy: 0.5064
- Precision (weighted): 0.5064
- Recall (weighted): 0.5064
- F1-score (weighted): 0.5064
- ROC-AUC: 0.5216921222364448
- Confusion Matrix:
```
[[593 608]
 [580 626]]
```

### Random Forest
- Accuracy: 0.4923
- Precision (weighted): 0.4923
- Recall (weighted): 0.4923
- F1-score (weighted): 0.4919
- ROC-AUC: 0.49219141594276744
- Confusion Matrix:
```
[[626 575]
 [647 559]]
```

### XGBoost
- Accuracy: 0.5206
- Precision (weighted): 0.5207
- Recall (weighted): 0.5206
- F1-score (weighted): 0.5203
- ROC-AUC: 0.5160172631154524
- Confusion Matrix:
```
[[654 547]
 [607 599]]
```

## Spatial Feature Set
### Logistic Regression
- Accuracy: 0.4973
- Precision (weighted): 0.4971
- Recall (weighted): 0.4973
- F1-score (weighted): 0.4954
- ROC-AUC: 0.5036688608028412
- Confusion Matrix:
```
[[524 677]
 [533 673]]
```

### Random Forest
- Accuracy: 0.5044
- Precision (weighted): 0.5044
- Recall (weighted): 0.5044
- F1-score (weighted): 0.5042
- ROC-AUC: 0.513619454766136
- Confusion Matrix:
```
[[624 577]
 [616 590]]
```

### XGBoost
- Accuracy: 0.5177
- Precision (weighted): 0.5177
- Recall (weighted): 0.5177
- F1-score (weighted): 0.5176
- ROC-AUC: 0.5173532144992494
- Confusion Matrix:
```
[[634 567]
 [594 612]]
```

## Burst Feature Set
### Logistic Regression
- Accuracy: 0.4907
- Precision (weighted): 0.4801
- Recall (weighted): 0.4907
- F1-score (weighted): 0.4063
- ROC-AUC: 0.498750350385182
- Confusion Matrix:
```
[[1043  158]
 [1068  138]]
```

### Random Forest
- Accuracy: 0.5226
- Precision (weighted): 0.5226
- Recall (weighted): 0.5226
- F1-score (weighted): 0.5226
- ROC-AUC: 0.5169462153567439
- Confusion Matrix:
```
[[611 590]
 [559 647]]
```

### XGBoost
- Accuracy: 0.5035
- Precision (weighted): 0.5036
- Recall (weighted): 0.5035
- F1-score (weighted): 0.5034
- ROC-AUC: 0.49881732055790984
- Confusion Matrix:
```
[[627 574]
 [621 585]]
```

## All Feature Set
### Logistic Regression
- Accuracy: 0.4744
- Precision (weighted): 0.4738
- Recall (weighted): 0.4744
- F1-score (weighted): 0.4718
- ROC-AUC: 0.4807685138006884
- Confusion Matrix:
```
[[485 716]
 [549 657]]
```

### Random Forest
- Accuracy: 0.5177
- Precision (weighted): 0.5179
- Recall (weighted): 0.5177
- F1-score (weighted): 0.5164
- ROC-AUC: 0.5278885892491469
- Confusion Matrix:
```
[[682 519]
 [642 564]]
```

### XGBoost
- Accuracy: 0.5160
- Precision (weighted): 0.5160
- Recall (weighted): 0.5160
- F1-score (weighted): 0.5160
- ROC-AUC: 0.5226386800386079
- Confusion Matrix:
```
[[628 573]
 [592 614]]
```

## LSTM Model Evaluation and Feature Ablation Study
This report compares a PyTorch LSTM sequence model trained under two configurations:
1. **Stage 1 (Absolute Features)**: Includes coordinates, bounding box sizes, and raw speeds.
2. **Stage 2 (Scale-Invariant Features)**: Restricts features to coordinate-free and scale-normalized kinematics (aspect ratio, normalized speed, angles).

## Summary of Performance
| Metric | Stage 1 (Absolute) | Stage 2 (Scale-Invariant) |
| :--- | :---: | :---: |
| **Test Accuracy** | 0.5466 | 0.5428 |
| **Precision** | 0.5598 | 0.5982 |
| **Recall** | 0.4357 | 0.2604 |
| **F1-Score** | 0.4900 | 0.3628 |
| **ROC-AUC** | 0.5607 | 0.5301 |

## Confusion Matrices
### Stage 1 (Absolute)
```
[[3125 1628]
 [2681 2070]]
```

### Stage 2 (Scale-Invariant)
```
[[3922  831]
 [3514 1237]]
```

## Analysis
- **Absolute Features (Stage 1)** capture the child's absolute size and location. While this might aid in fitting training distributions, it often generalizes poorly if train and test environments differ (distribution shift).
- **Scale-Invariant Features (Stage 2)** drop all camera distance/bounding box size biases, forcing the model to learn pure motion dynamics. If Stage 2 matches or exceeds Stage 1, it confirms that raw coordinates and absolute sizes are introducing negative bias/generalization error.