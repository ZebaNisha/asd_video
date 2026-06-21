import csv, pathlib, json, sys
BASE_DIR = pathlib.Path(r'c:/asd_project')
outputs = BASE_DIR / 'outputs'
report_lines = []

def validate_detection(p):
    required = ["video_id","frame_number","skeleton_id","bbox_x","bbox_y","bbox_width","bbox_height","bbox_area","centroid_x","centroid_y"]
    try:
        with p.open(newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return False, 'empty file'
        header = rows[0]
        for col in required:
            if col not in header:
                return False, f'missing column {col}'
        if len(rows) < 3:
            return False, 'not enough data rows'
        col_len = len(header)
        for i, r in enumerate(rows[1:], start=2):
            if len(r) != col_len:
                return False, f'row {i} column count mismatch'
        return True, ''
    except Exception as e:
        return False, f'exception {e}'

def validate_tracked(p):
    try:
        with p.open(newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return False, 'empty file'
        # Determine if first row is header by checking if all values are numeric (header likely non-numeric)
        first = rows[0]
        is_header = any(not val.replace('.','',1).isdigit() for val in first)
        data = rows[1:] if is_header else rows
        if len(data) < 2:
            return False, 'not enough data rows'
        col_len = len(rows[0])
        for i, r in enumerate(data, start=(2 if is_header else 1)):
            if len(r) != col_len:
                return False, f'row {i} column count mismatch'
        return True, ''
    except Exception as e:
        return False, f'exception {e}'

def scan_stage(name, pattern, validator):
    stage_dir = outputs / name.lower().replace(' ', '_')
    files = list(stage_dir.glob(pattern))
    for f in files:
        ok, reason = validator(f)
        if not ok:
            report_lines.append(f"- {name}: {f.relative_to(BASE_DIR)} – {reason}")

scan_stage('detections', '*_detections.csv', validate_detection)
scan_stage('tracked', '*_tracked.csv', validate_tracked)

report_path = BASE_DIR / 'outputs' / 'reports' / 'genuine_corruption_report.md'
with report_path.open('w', encoding='utf-8') as out:
    out.write('# Genuine Corruption Report\n\n')
    if report_lines:
        out.write('\n'.join(report_lines))
    else:
        out.write('No genuinely corrupted files detected.')
print('Report written to', report_path)
