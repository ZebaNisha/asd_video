import pathlib
import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.spatial import ConvexHull

PROJECT_ROOT = pathlib.Path('c:/asd_project')
CHILD_SEQ_DIR = PROJECT_ROOT / 'outputs' / 'child_sequences'
FEATURES_PATH = PROJECT_ROOT / 'features.csv'

def compute_features(seq_path):
    df = pd.read_csv(seq_path)
    # Ensure required columns exist
    required = ['centroid_x', 'centroid_y']
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {seq_path}")

    # Basic trajectory metrics
    xs = df['centroid_x'].values
    ys = df['centroid_y'].values
    dt = 1  # assume constant frame interval = 1

    # Velocity
    vx = np.gradient(xs, dt)
    vy = np.gradient(ys, dt)
    speed = np.sqrt(vx**2 + vy**2)

    # Acceleration (second derivative)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    acceleration = np.sqrt(ax**2 + ay**2)

    # Jerk (third derivative)
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jerk = np.sqrt(jx**2 + jy**2)

    # Direction change rate (angular change between velocity vectors)
    dot = vx[:-1]*vx[1:] + vy[:-1]*vy[1:]
    norm_prod = np.sqrt((vx[:-1]**2 + vy[:-1]**2)*(vx[1:]**2 + vy[1:]**2))
    cos_angle = np.clip(dot / (norm_prod + 1e-8), -1, 1)
    angle = np.arccos(cos_angle)
    direction_change_rate = np.mean(angle)

    # Periodicity from autocorrelation of speed
    acf = np.correlate(speed - speed.mean(), speed - speed.mean(), mode='full')
    acf = acf[acf.size // 2:]
    # Find first peak after lag 0
    peaks = np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
    Periodicity = float(peaks[0]) if peaks.size > 0 else np.nan

    # FFT based features on speed
    speed_fft = np.abs(fft(speed))
    freqs = np.fft.fftfreq(len(speed), d=dt)
    positive = freqs > 0
    fft_mag = speed_fft[positive]
    fft_freq = freqs[positive]
    if fft_mag.size == 0:
        fft_peak_freq = np.nan
        fft_entropy = np.nan
    else:
        fft_peak_freq = float(fft_freq[np.argmax(fft_mag)])
        prob = fft_mag / fft_mag.sum()
        fft_entropy = -float(np.sum(prob * np.log2(prob + 1e-12)))

    # ACF decay (fit exponential) and entropy
    # Simple approximation: decay constant as slope of log(acf)
    log_acf = np.log(acf + 1e-12)
    acf_decay = float(-np.mean(np.diff(log_acf))) if len(log_acf) > 1 else np.nan
    acf_prob = acf / (acf.sum() + 1e-12)
    acf_entropy = -float(np.sum(acf_prob * np.log2(acf_prob + 1e-12)))

    # Burst features (speed > mean + 1 std)
    threshold = speed.mean() + speed.std()
    above = speed > threshold
    # Find contiguous bursts
    bursts = []
    current_len = 0
    for val in above:
        if val:
            current_len += 1
        elif current_len > 0:
            bursts.append(current_len)
            current_len = 0
    if current_len > 0:
        bursts.append(current_len)
    burst_duration = float(np.mean(bursts)) if bursts else 0.0
    burst_frequency = float(len(bursts)) / (len(speed) / 1000.0) if len(speed) > 0 else 0.0
    inter_burst_interval = float(np.mean(np.diff(np.where(above)[0]))) if len(bursts) > 1 else np.nan

    # Spatial features
    # Occupancy (grid of 1-pixel resolution)
    visited = set(zip(xs.astype(int), ys.astype(int)))
    total_grid = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    spatial_occupancy = len(visited) / total_grid if total_grid > 0 else 0.0

    # Path efficiency
    straight = np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]])
    path_len = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    path_efficiency = straight / path_len if path_len > 0 else 0.0

    # Convex hull area
    if len(xs) >= 3:
        points = np.column_stack((xs, ys))
        hull = ConvexHull(points)
        convex_hull_area = float(hull.area)
    else:
        convex_hull_area = 0.0

    # Assemble results
    video_id = seq_path.stem.replace('_child_sequence', '')
    return {
        'video_id': video_id,
        'acceleration': float(np.mean(acceleration)),
        'jerk': float(np.mean(jerk)),
        'direction_change_rate': direction_change_rate,
        'Periodicity': Periodicity,
        'fft_peak_freq': fft_peak_freq,
        'fft_entropy': fft_entropy,
        'acf_decay': acf_decay,
        'acf_entropy': acf_entropy,
        'burst_duration': burst_duration,
        'burst_frequency': burst_frequency,
        'inter_burst_interval': inter_burst_interval,
        'spatial_occupancy': spatial_occupancy,
        'path_efficiency': path_efficiency,
        'convex_hull_area': convex_hull_area
    }

def main():
    seq_files = list(CHILD_SEQ_DIR.glob('*_child_sequence.csv'))
    # Optional: limit to a pilot set using a manifest file
    # manifest = pathlib.Path('pilot_manifest.txt')
    # if manifest.exists():
    #     wanted = {line.strip() for line in manifest.read_text().splitlines()}
    #     seq_files = [p for p in seq_files if p.stem.replace('_child_sequence', '') in wanted]

    results = []
    for seq_path in seq_files:
        try:
            feats = compute_features(seq_path)
            results.append(feats)
        except Exception as e:
            print(f"Failed on {seq_path}: {e}")
    df_out = pd.DataFrame(results)
    df_out.to_csv(FEATURES_PATH, index=False)
    print(f"Feature file written to {FEATURES_PATH}, rows={len(df_out)}")

if __name__ == "__main__":
    main()
