import pathlib, random, csv, sys, json

random.seed(42)
base_det = pathlib.Path(r'c:/asd_project/outputs/detections')
base_trk = pathlib.Path(r'c:/asd_project/outputs/tracked')

files_det = list(base_det.glob('*_detections.csv'))
files_trk = list(base_trk.glob('*_tracked.csv'))

sample_det = random.sample(files_det, min(5, len(files_det)))
sample_trk = random.sample(files_trk, min(5, len(files_trk)))

def analyze_csv(path):
    info = {
        'file_path': str(path),
        'size_bytes': path.stat().st_size,
        'header': None,
        'data_rows': 0,
        'corruption_reason': None,
    }
    try:
        with path.open('r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        info['corruption_reason'] = f'Failed to read CSV: {e}'
        return info
    if info['size_bytes'] == 0:
        info['corruption_reason'] = 'Zero-byte file'
    elif len(rows) < 2:
        info['corruption_reason'] = 'Header only, no data rows'
    else:
        info['corruption_reason'] = 'None'
    if rows:
        info['header'] = rows[0]
        info['data_rows'] = len(rows) - 1
    return info

results = []
for p in sample_det + sample_trk:
    results.append(analyze_csv(p))

# Output as JSON for easy parsing
print(json.dumps(results, indent=2))
