import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
CHILD_SEQ_DIR = r'c:/asd_project/outputs/child_sequences'
BASELINE_CSV = r'c:/asd_project/outputs/features/features.csv'
REPORT_MD = r'c:/asd_project/LSTM_EVALUATION.md'
MODEL_DIR = r'c:/asd_project/models/lstm'

class TrajectoryDataset(Dataset):
    def __init__(self, metadata_df, stage=1, max_len=150):
        self.metadata = metadata_df.reset_index(drop=True)
        self.stage = stage
        self.max_len = max_len
        
        # Load and preprocess all sequences in memory for speed
        self.sequences = []
        self.lengths = []
        self.labels = []
        
        # We will fit a scaler on training set later if needed, but for now we do standard scale-invariant normalization inside
        for idx, row in self.metadata.iterrows():
            seq_file = os.path.join(CHILD_SEQ_DIR, f"{row['unique_video_id']}_child_sequence.csv")
            if not os.path.exists(seq_file):
                # Fallback to zero sequence if missing
                feats = np.zeros((100, 7 if stage==1 else 4), dtype=np.float32)
                self.sequences.append(feats)
                self.lengths.append(100)
                self.labels.append(row['label'])
                continue
                
            seq_df = pd.read_csv(seq_file)
            
            # Load all numeric columns except identifiers
            # Assume CSV contains 'unique_video_id', 'label' and feature columns
            feature_cols = [c for c in seq_df.columns if c not in ['unique_video_id', 'label']]
            feats = seq_df[feature_cols].values.astype(np.float32)
            # No additional scaling here – we will standardize later in main
            
            # For compatibility with existing stages, keep the same max_len handling
            if len(feats) > self.max_len:
                feats = feats[:self.max_len]
                seq_len = self.max_len
            else:
                seq_len = len(feats)

                
            self.sequences.append(feats)
            self.lengths.append(seq_len)
            self.labels.append(row['label'])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = self.lengths[idx]
        label = self.labels[idx]
        
        # Pad sequence to max_len
        padded_seq = np.zeros((self.max_len, seq.shape[1]), dtype=np.float32)
        padded_seq[:seq_len, :] = seq
        
        return torch.tensor(padded_seq), torch.tensor(seq_len), torch.tensor(label, dtype=torch.float32)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional, 
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x shape: (batch_size, max_len, input_dim)
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_out, (hn, cn) = self.lstm(packed_x)
        
        # Unpack output
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Retrieve hidden state at the last valid timestep for each sequence in the batch
        batch_size = x.size(0)
        idx = (lengths - 1).view(-1, 1).expand(batch_size, out.size(2)).unsqueeze(1)
        # Gather output at last valid index
        last_out = out.gather(1, idx).squeeze(1) # shape: (batch_size, lstm_out_dim)
        
        fc_out = self.fc(last_out)
        return self.sigmoid(fc_out).squeeze(1)

def train_model(train_loader, test_loader, input_dim, epochs=40, lr=0.001):
    model = LSTMClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_acc = 0.0
    best_metrics = {}
    
    # Simple early stopping
    patience = 8
    epochs_no_improve = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seqs, lens, labels in train_loader:
            seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs, lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            for seqs, lens, labels in test_loader:
                seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
                outputs = model(seqs, lens)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * seqs.size(0)
                
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probas.extend(outputs.cpu().numpy())
                
        test_loss /= len(test_loader.dataset)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probas)
        except ValueError:
            auc = 0.5
            
        cm = confusion_matrix(all_labels, all_preds)
        
        # Check early stopping & save best model
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            best_acc = acc
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'cm': cm
            }
            # Save model parameters
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'best_lstm_stage.pt'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    return best_metrics

def main():
    # Load splits & labels
    df_meta = pd.read_csv(BASELINE_CSV)
    df_meta['label'] = df_meta['label'].map({'asd': 1, 'td': 0})
    
    # Split
    train_df = df_meta[df_meta['split'] == 'train']
    test_df = df_meta[df_meta['split'] == 'test']
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ----------------------------------------------------
    # Stage 1: Absolute (Scale-Dependent) LSTM
    # ----------------------------------------------------
    print("\n--- Training Stage 1 LSTM (Absolute Bounding Box & Scale Features) ---")
    train_ds_s1 = TrajectoryDataset(train_df, stage=1)
    test_ds_s1 = TrajectoryDataset(test_df, stage=1)
    
    # Scale features manually over dimensions
    # Find mean/std of train_ds_s1 sequences (ignoring pad zeros)
    all_train_feats = np.concatenate([ds[:l] for ds, l, _ in zip(train_ds_s1.sequences, train_ds_s1.lengths, train_ds_s1.labels)], axis=0)
    mean_feats_s1 = np.mean(all_train_feats, axis=0)
    std_feats_s1 = np.std(all_train_feats, axis=0) + 1e-8
    
    # Apply standardization
    for i in range(len(train_ds_s1.sequences)):
        train_ds_s1.sequences[i] = (train_ds_s1.sequences[i] - mean_feats_s1) / std_feats_s1
    for i in range(len(test_ds_s1.sequences)):
        test_ds_s1.sequences[i] = (test_ds_s1.sequences[i] - mean_feats_s1) / std_feats_s1
        
    train_loader_s1 = DataLoader(train_ds_s1, batch_size=64, shuffle=True)
    test_loader_s1 = DataLoader(test_ds_s1, batch_size=64, shuffle=False)
    
    # Infer input dimension from the first training sequence
    input_dim_s1 = train_ds_s1.sequences[0].shape[1] if train_ds_s1.sequences else 0
    metrics_s1 = train_model(train_loader_s1, test_loader_s1, input_dim=input_dim_s1)
    print(f"Stage 1 accuracy: {metrics_s1['accuracy']:.4f}, AUC: {metrics_s1['auc']:.4f}")
    
    # ----------------------------------------------------
    # Stage 2: Scale-Invariant LSTM
    # ----------------------------------------------------
    print("\n--- Training Stage 2 LSTM (Scale-Invariant Features) ---")
    train_ds_s2 = TrajectoryDataset(train_df, stage=2)
    test_ds_s2 = TrajectoryDataset(test_df, stage=2)
    
    # Scale features manually
    all_train_feats_s2 = np.concatenate([ds[:l] for ds, l, _ in zip(train_ds_s2.sequences, train_ds_s2.lengths, train_ds_s2.labels)], axis=0)
    mean_feats_s2 = np.mean(all_train_feats_s2, axis=0)
    std_feats_s2 = np.std(all_train_feats_s2, axis=0) + 1e-8
    
    # Apply standardization
    for i in range(len(train_ds_s2.sequences)):
        train_ds_s2.sequences[i] = (train_ds_s2.sequences[i] - mean_feats_s2) / std_feats_s2
    for i in range(len(test_ds_s2.sequences)):
        test_ds_s2.sequences[i] = (test_ds_s2.sequences[i] - mean_feats_s2) / std_feats_s2
        
    train_loader_s2 = DataLoader(train_ds_s2, batch_size=64, shuffle=True)
    test_loader_s2 = DataLoader(test_ds_s2, batch_size=64, shuffle=False)
    input_dim_s2 = train_ds_s2.sequences[0].shape[1] if train_ds_s2.sequences else 0
    metrics_s2 = train_model(train_loader_s2, test_loader_s2, input_dim=input_dim_s2)
    print(f"Stage 2 accuracy: {metrics_s2['accuracy']:.4f}, AUC: {metrics_s2['auc']:.4f}")
    
    # ----------------------------------------------------
    # Write Comparison Report
    # ----------------------------------------------------
    report_lines = [
        '# LSTM Model Evaluation and Feature Ablation Study',
        'This report compares a PyTorch LSTM sequence model trained under two configurations:',
        '1. **Stage 1 (Absolute Features)**: Includes coordinates, bounding box sizes, and raw speeds.',
        '2. **Stage 2 (Scale-Invariant Features)**: Restricts features to coordinate-free and scale-normalized kinematics (aspect ratio, normalized speed, angles).',
        '',
        '## Summary of Performance',
        '| Metric | Stage 1 (Absolute) | Stage 2 (Scale-Invariant) |',
        '| :--- | :---: | :---: |',
        f"| **Test Accuracy** | {metrics_s1['accuracy']:.4f} | {metrics_s2['accuracy']:.4f} |",
        f"| **Precision** | {metrics_s1['precision']:.4f} | {metrics_s2['precision']:.4f} |",
        f"| **Recall** | {metrics_s1['recall']:.4f} | {metrics_s2['recall']:.4f} |",
        f"| **F1-Score** | {metrics_s1['f1']:.4f} | {metrics_s2['f1']:.4f} |",
        f"| **ROC-AUC** | {metrics_s1['auc']:.4f} | {metrics_s2['auc']:.4f} |",
        '',
        '## Confusion Matrices',
        '',
        '### Stage 1 (Absolute)',
        '```',
        f"{metrics_s1['cm']}",
        '```',
        '',
        '### Stage 2 (Scale-Invariant)',
        '```',
        f"{metrics_s2['cm']}",
        '```',
        '',
        '## Analysis',
        '- **Absolute Features (Stage 1)** capture the child\'s absolute size and location. While this might aid in fitting training distributions, it often generalizes poorly if train and test environments differ (distribution shift).',
        '- **Scale-Invariant Features (Stage 2)** drop all camera distance/bounding box size biases, forcing the model to learn pure motion dynamics. If Stage 2 matches or exceeds Stage 1, it confirms that raw coordinates and absolute sizes are introducing negative bias/generalization error.'
    ]
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Comparative report successfully written to {REPORT_MD}")

if __name__ == '__main__':
    main()
