import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

CHILD_SEQ_DIR = r'c:/asd_project/outputs/child_sequences'
BASELINE_CSV = r'c:/asd_project/outputs/features/features.csv'
MODEL_DIR = r'c:/asd_project/models/lstm'

class TrajectoryDataset(Dataset):
    def __init__(self, metadata_df, stage=2, max_len=150):
        self.metadata = metadata_df.reset_index(drop=True)
        self.stage = stage
        self.max_len = max_len
        self.sequences = []
        self.lengths = []
        self.labels = []
        for _, row in self.metadata.iterrows():
            seq_file = os.path.join(CHILD_SEQ_DIR, f"{row['unique_video_id']}_child_sequence.csv")
            if not os.path.exists(seq_file):
                feats = np.zeros((100, 7 if stage == 1 else 4), dtype=np.float32)
                self.sequences.append(feats)
                self.lengths.append(100)
                self.labels.append(row['label'])
                continue
            seq_df = pd.read_csv(seq_file)
            feature_cols = [c for c in seq_df.columns if c not in ['unique_video_id', 'label']]
            feats = seq_df[feature_cols].values.astype(np.float32)
            seq_len = min(len(feats), self.max_len)
            if len(feats) > self.max_len:
                feats = feats[:self.max_len]
            self.sequences.append(feats)
            self.lengths.append(seq_len)
            self.labels.append(row['label'])
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = self.lengths[idx]
        label = self.labels[idx]
        padded = np.zeros((self.max_len, seq.shape[1]), dtype=np.float32)
        padded[:seq_len] = seq
        return torch.tensor(padded), torch.tensor(seq_len), torch.tensor(label, dtype=torch.float32)

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=0.3 if num_layers > 1 else 0.0)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = torch.nn.Linear(lstm_out_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        batch = x.size(0)
        idx = (lengths - 1).view(-1, 1).expand(batch, out.size(2)).unsqueeze(1)
        last = out.gather(1, idx).squeeze(1)
        return self.sigmoid(self.fc(last)).squeeze(1)

# Load metadata
df = pd.read_csv(BASELINE_CSV)
# Map labels to binary
label_map = {'asd': 1, 'td': 0}
df['label'] = df['label'].map(label_map)
train_df = df[df['split'] == 'train']

# Create dataset
train_ds = TrajectoryDataset(train_df, stage=2)
# Compute scaling stats across all training sequences
all_feats = np.concatenate([seq[:l] for seq, l, _ in zip(train_ds.sequences, train_ds.lengths, train_ds.labels)], axis=0)
mean = np.mean(all_feats, axis=0)
std = np.std(all_feats, axis=0) + 1e-8
# Apply scaling
for i in range(len(train_ds.sequences)):
    train_ds.sequences[i] = (train_ds.sequences[i] - mean) / std

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

# Load model
model_path = os.path.join(MODEL_DIR, 'best_lstm_stage.pt')
input_dim = train_ds.sequences[0].shape[1]
model = LSTMClassifier(input_dim)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for seqs, lens, labels in train_loader:
        outputs = model(seqs, lens)
        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
acc = accuracy_score(all_labels, all_preds)
print('Train Accuracy (Stage2 model):', acc)
