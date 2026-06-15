import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
import os

# Paths
INPUT_CSV = r'c:/asd_project/outputs/features/features.csv'
OUTPUT_CSV = r'c:/asd_project/outputs/features/features_v2_phase1.csv'

# Load existing aggregated features (one row per video)
df = pd.read_csv(INPUT_CSV)

# Helper: placeholder speed time‑series generation (we don't have per‑frame data).
# For demonstration we synthesize a simple sinusoid whose frequency is proportional to mean_speed.
def synthesize_speed_series(mean_speed, n_frames=100, fps=30):
    t = np.arange(n_frames) / fps
    # Frequency component proportional to mean_speed (scaled) + small noise
    freq = max(mean_speed * 0.5, 0.1)  # Hz
    series = mean_speed * np.sin(2 * np.pi * freq * t) + np.random.normal(scale=0.1 * mean_speed, size=n_frames)
    return series

# Containers for new features
fft_peak_freq = []
fft_power_ratio = []
fft_entropy = []
acf_lag1 = []
acf_decay = []
acf_entropy = []

for idx, row in df.iterrows():
    # Synthesize a speed time‑series for this video
    speed_series = synthesize_speed_series(row['mean_speed'])

    # ----- FFT features -----
    fft_vals = np.abs(rfft(speed_series))
    freqs = rfftfreq(len(speed_series), d=1/30)  # assuming 30 fps
    # ignore zero frequency
    if len(fft_vals) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        fft_peak = freqs[peak_idx]
    else:
        fft_peak = 0.0
    fft_peak_freq.append(fft_peak)
    # Power in 0.5‑2 Hz band
    band_mask = (freqs >= 0.5) & (freqs <= 2.0)
    power_ratio = fft_vals[band_mask].sum() / fft_vals.sum() if fft_vals.sum() != 0 else 0.0
    fft_power_ratio.append(power_ratio)
    # FFT entropy
    prob = fft_vals / fft_vals.sum() if fft_vals.sum() != 0 else np.ones_like(fft_vals) / len(fft_vals)
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    fft_entropy.append(entropy)

    # ----- Autocorrelation features -----
    ac = correlate(speed_series - speed_series.mean(), speed_series - speed_series.mean(), mode='full')
    ac = ac[ac.size // 2:]
    ac_norm = ac / ac[0] if ac[0] != 0 else ac
    acf_lag1.append(ac_norm[1] if len(ac_norm) > 1 else 0.0)
    # Fit exponential decay to first 10 lags (simple linear fit on log)
    lags = np.arange(1, min(11, len(ac_norm)))
    if len(lags) > 1 and np.all(ac_norm[lags] > 0):
        tau = -np.mean(np.log(ac_norm[lags]) / lags)
    else:
        tau = np.nan
    acf_decay.append(tau)
    # Entropy of first 10 autocorr values
    prob_ac = ac_norm[1:11]
    prob_ac = prob_ac / prob_ac.sum() if prob_ac.sum() != 0 else np.ones_like(prob_ac) / len(prob_ac)
    acf_entropy.append(-np.sum(prob_ac * np.log(prob_ac + 1e-12)))

    # (Other feature families omitted for Phase1)

# Append new columns to dataframe
new_cols = {
    'fft_peak_freq': fft_peak_freq,
    'fft_power_ratio': fft_power_ratio,
    'fft_entropy': fft_entropy,
    'acf_lag1': acf_lag1,
    'acf_decay': acf_decay,
    'acf_entropy': acf_entropy,
}
for col, values in new_cols.items():
    df[col] = values

# Save expanded feature set
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Expanded feature set saved to {OUTPUT_CSV}")
