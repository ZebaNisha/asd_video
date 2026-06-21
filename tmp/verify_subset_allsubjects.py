import pathlib, numpy as np, datetime, json

BASE = pathlib.Path(r'c:/asd_project')
SUBSET = BASE / 'outputs' / 'vgg16_lstm' / 'subset_allsubjects_20videos'
TRAIN_PATH = SUBSET / 'vgg16_child_train.npz'
TEST_PATH = SUBSET / 'vgg16_child_test.npz'
REPORT = BASE / 'outputs' / 'reports' / 'subset_allsubjects_20videos_verification.md'

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        'X': data['X'],
        'y': data['y'],
        'lengths': data['lengths'],
        'video_ids': data['video_ids']
    }

def subject_from_id(vid):
    s = str(vid)
    if 'Subj_' in s:
        part = s.split('Subj_')[1]
        return part.split('_')[0]
    return None

def analyze(split_name, npz):
    vids = npz['video_ids']
    labels = npz['y']
    # assume binary labels 0/1 where 1=ASD, 0=TD
    subject_counts = {}
    subject_label = {}
    for vid, lab in zip(vids, labels):
        subj = subject_from_id(vid)
        if subj is None:
            continue
        subject_counts.setdefault(subj, 0)
        subject_counts[subj] += 1
        # store label (assuming all same per subject)
        if subj not in subject_label:
            subject_label[subj] = int(lab)
    total_samples = len(labels)
    asd_samples = int((labels == 1).sum())
    td_samples = total_samples - asd_samples
    asd_subjects = sum(1 for s in subject_label.values() if s == 1)
    td_subjects = sum(1 for s in subject_label.values() if s == 0)
    clips = list(subject_counts.values())
    return {
        'subjects': set(subject_counts.keys()),
        'subject_counts': subject_counts,
        'subject_label': subject_label,
        'total_samples': total_samples,
        'asd_samples': asd_samples,
        'td_samples': td_samples,
        'asd_subjects': asd_subjects,
        'td_subjects': td_subjects,
        'clips': clips,
        'split': split_name
    }

def main():
    train_npz = load_npz(TRAIN_PATH)
    test_npz = load_npz(TEST_PATH)
    train_info = analyze('train', train_npz)
    test_info = analyze('test', test_npz)

    # leakage check
    leakage = train_info['subjects'] & test_info['subjects']
    leakage_exists = len(leakage) > 0

    # class balance percentages
    def pct(num, tot):
        return (num / tot * 100) if tot else 0

    # subject balance percentages
    train_total_subj = train_info['asd_subjects'] + train_info['td_subjects']
    test_total_subj = test_info['asd_subjects'] + test_info['td_subjects']

    # clip stats
    def clip_stats(clips):
        arr = np.array(clips)
        return arr.min(), arr.max(), arr.mean()
    train_min, train_max, train_mean = clip_stats(train_info['clips'])
    test_min, test_max, test_mean = clip_stats(test_info['clips'])

    ready = not leakage_exists

    with REPORT.open('w', encoding='utf-8') as f:
        f.write('# Subset "allsubjects_20videos" Verification Report\n\n')
        f.write('## Subject Counts\n')
        f.write(f"- Unique train subjects: {len(train_info['subjects'])}\n")
        f.write(f"- Unique test subjects: {len(test_info['subjects'])}\n")
        f.write(f"- ASD train subjects: {train_info['asd_subjects']}\n")
        f.write(f"- TD train subjects: {train_info['td_subjects']}\n")
        f.write(f"- ASD test subjects: {test_info['asd_subjects']}\n")
        f.write(f"- TD test subjects: {test_info['td_subjects']}\n\n")
        f.write('## Leakage Check\n')
        f.write(f"- Overlap between train and test subjects: {len(leakage)}\n")
        if leakage_exists:
            f.write(f"  Subjects: {', '.join(sorted(leakage))}\n")
        else:
            f.write('  **No subject leakage detected.**\n')
        f.write('\n')
        f.write('## Class Balance (samples)\n')
        f.write(f"- Train ASD samples: {train_info['asd_samples']} ({pct(train_info['asd_samples'], train_info['total_samples']):.1f}%)\n")
        f.write(f"- Train TD samples: {train_info['td_samples']} ({pct(train_info['td_samples'], train_info['total_samples']):.1f}%)\n")
        f.write(f"- Test ASD samples: {test_info['asd_samples']} ({pct(test_info['asd_samples'], test_info['total_samples']):.1f}%)\n")
        f.write(f"- Test TD samples: {test_info['td_samples']} ({pct(test_info['td_samples'], test_info['total_samples']):.1f}%)\n\n")
        f.write('## Subject Balance (by label)\n')
        f.write(f"- Train ASD subjects: {train_info['asd_subjects']} ({pct(train_info['asd_subjects'], train_total_subj):.1f}%)\n")
        f.write(f"- Train TD subjects: {train_info['td_subjects']} ({pct(train_info['td_subjects'], train_total_subj):.1f}%)\n")
        f.write(f"- Test ASD subjects: {test_info['asd_subjects']} ({pct(test_info['asd_subjects'], test_total_subj):.1f}%)\n")
        f.write(f"- Test TD subjects: {test_info['td_subjects']} ({pct(test_info['td_subjects'], test_total_subj):.1f}%)\n\n")
        f.write('## Clip Statistics per Subject\n')
        f.write(f"- Train clips per subject: min={train_min}, max={train_max}, mean={train_mean:.2f}\n")
        f.write(f"- Test clips per subject:  min={test_min}, max={test_max}, mean={test_mean:.2f}\n\n")
        f.write('## Readiness\n')
        f.write(f"READY_FOR_BILSTM_TRAINING = {'YES' if ready else 'NO'}\n")
        f.write('\n')
    print('Report written to', REPORT)

if __name__ == '__main__':
    main()
