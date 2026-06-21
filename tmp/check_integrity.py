import csv, pathlib, json, sys, numpy as np
BASE_DIR = pathlib.Path(r'c:/asd_project')
outputs = BASE_DIR / 'outputs'

# Validation helpers
def is_zero_byte(p: pathlib.Path) -> bool:
    return p.stat().st_size == 0

def validate_csv(p: pathlib.Path, required: list=None, allow_headerless=False) -> (bool,str):
    if is_zero_byte(p):
        return False, 'zero-byte file'
    try:
        with p.open(newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return False, 'empty file'
        # column count consistency
        expected_len = len(rows[0])
        for r in rows:
            if len(r) != expected_len:
                return False, f'row length mismatch (expected {expected_len}, got {len(r)})'
        if required:
            header = rows[0]
            for col in required:
                if col not in header:
                    return False, f'missing required column {col}'
            data_rows = len(rows) - 1
        else:
            # decide header presence
            if allow_headerless:
                data_rows = len(rows)
            else:
                # assume first row is header
                data_rows = len(rows) - 1
        if data_rows < 2:
            return False, f'not enough data rows ({data_rows})'
    except Exception as e:
        return False, f'exception {e}'
    return True, ''

def validate_npz(p: pathlib.Path) -> (bool,str):
    if is_zero_byte(p):
        return False, 'zero-byte file'
    try:
        data = np.load(p, allow_pickle=True)
        # expect keys X, y, lengths, video_ids
        for key in ['X','y','lengths','video_ids']:
            if key not in data:
                return False, f'missing key {key}'
            if data[key].shape[0] == 0:
                return False, f'empty array for {key}'
    except Exception as e:
        return False, f'exception {e}'
    return True, ''

stages = {
    'Detection CSVs': {'dir': outputs / 'detections', 'pattern': '*_detections.csv', 'required': ["video_id","frame_number","skeleton_id","bbox_x","bbox_y","bbox_width","bbox_height","bbox_area","centroid_x","centroid_y"], 'allow_headerless': False},
    'Tracked CSVs': {'dir': outputs / 'tracked', 'pattern': '*_tracked.csv', 'required': None, 'allow_headerless': True},
    'Child Sequences': {'dir': outputs / 'child_sequences', 'pattern': '*_child_sequence.csv', 'required': None, 'allow_headerless': False},
    'VGG16 NPZ Files': {'dir': outputs / 'features', 'pattern': '*_vgg16.npz', 'required': None, 'allow_headerless': False},
}

report = []
for stage, cfg in stages.items():
    files = list(cfg['dir'].glob(cfg['pattern']))
    total = len(files)
    unreadable = 0
    details = []
    for f in files:
        if stage.endswith('NPZ Files'):
            ok, reason = validate_npz(f)
        else:
            ok, reason = validate_csv(f, cfg['required'], cfg['allow_headerless'])
        if not ok:
            unreadable += 1
            details.append(f"- {f.relative_to(BASE_DIR)}: {reason}")
    perc = (unreadable/total*100) if total>0 else 0
    report.append(f"## {stage}\n- Total files: {total}\n- Unreadable files: {unreadable}\n- Percentage unreadable: {perc:.2f}%\n")
    if details:
        report.append('\n'.join(details))
    report.append('\n')

# Write markdown report
out_path = outputs / 'reports' / 'genuine_corruption_report.md'
with out_path.open('w', encoding='utf-8') as out:
    out.write('# Genuine Corruption & Integrity Report\n\n')
    out.writelines('\n'.join(report))
print('Report written to', out_path)
