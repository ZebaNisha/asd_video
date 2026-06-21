import numpy as np, pathlib, json

def load_info(npz_path: pathlib.Path):
    data = np.load(npz_path, allow_pickle=True)
    info = {}
    info['keys'] = list(data.files)
    info['shapes'] = {k: tuple(data[k].shape) for k in data.files}
    # Expected keys
    X = data['X']
    y = data['y']
    lengths = data['lengths']
    video_ids = data['video_ids']
    info['num_samples'] = X.shape[0]
    # feature dimension (assuming shape (samples, seq_len, feat_dim))
    info['feature_dim'] = X.shape[2] if X.ndim == 3 else None
    info['seq_len_min'] = int(lengths.min())
    info['seq_len_max'] = int(lengths.max())
    info['seq_len_mean'] = float(lengths.mean())
    # class distribution
    uniq, cnt = np.unique(y, return_counts=True)
    info['class_distribution'] = {int(u): int(c) for u, c in zip(uniq, cnt)}
    # unique subjects from video_ids (format like "Subj_XX_part_YY")
    subjects = set()
    for vid in video_ids:
        s = str(vid)
        if 'Subj_' in s:
            part = s.split('Subj_')[1]
            subj = part.split('_')[0]
            subjects.add(subj)
    info['unique_subjects'] = len(subjects)
    return info

def main():
    base = pathlib.Path('c:/asd_project/outputs')
    train_path = base / 'vgg16_child_train.npz'
    test_path = base / 'vgg16_child_test.npz'
    train_info = load_info(train_path)
    test_info = load_info(test_path)
    report_path = base / 'reports' / 'vgg16_npz_inspection_report.md'
    with report_path.open('w', encoding='utf-8') as f:
        f.write('# VGG16 NPZ Inspection Report\n\n')
        for name, info in [('Train', train_info), ('Test', test_info)]:
            f.write(f'## {name} Set\n')
            f.write(f'- **File**: {train_path if name=="Train" else test_path}\n')
            f.write(f'- **Keys**: {", ".join(info["keys"]) }\n')
            f.write('- **Shapes**:\n')
            for k, sh in info['shapes'].items():
                f.write(f'  - {k}: {sh}\n')
            f.write(f'- **Number of samples**: {info["num_samples"]}\n')
            f.write(f'- **Feature dimension**: {info["feature_dim"]}\n')
            f.write(f'- **Sequence lengths**: min={info["seq_len_min"]}, max={info["seq_len_max"]}, mean={info["seq_len_mean"]:.2f}\n')
            f.write('- **Class distribution**:\n')
            for cls, cnt in info['class_distribution'].items():
                f.write(f'  - Class {cls}: {cnt}\n')
            f.write(f'- **Unique subjects represented**: {info["unique_subjects"]}\n\n')
    print('Report written to', report_path)

if __name__ == '__main__':
    main()
