import os, pathlib, json, datetime, sys
import numpy as np

BASE = pathlib.Path(r'c:/asd_project')
REPORT_PATH = BASE / 'outputs' / 'reports' / 'npz_investigation_report.md'

keywords = ['vgg16', 'lstm', 'feature', 'child']
search_dirs = ['outputs', 'reports', 'logs']

def format_ts(ts):
    return datetime.datetime.fromtimestamp(ts).isoformat()

npz_info = []
found_keywords = set()

# Walk the entire project for .npz files
for root, dirs, files in os.walk(BASE):
    for f in files:
        if f.lower().endswith('.npz'):
            p = pathlib.Path(root) / f
            try:
                stat = p.stat()
                size = stat.st_size
                ctime = format_ts(stat.st_ctime)
                mtime = format_ts(stat.st_mtime)
            except Exception as e:
                size = ctime = mtime = 'N/A'
            # Try loading
            try:
                data = np.load(p, allow_pickle=True)
                load_success = True
                keys = list(data.files)
                shapes = {k: tuple(data[k].shape) for k in keys}
            except Exception as e:
                load_success = False
                keys = []
                shapes = {}
            npz_info.append({
                'path': str(p),
                'size': size,
                'ctime': ctime,
                'mtime': mtime,
                'load_success': load_success,
                'keys': keys,
                'shapes': shapes,
                'error': str(e) if not load_success else ''
            })

# Search for keywords in selected directories
for sub in search_dirs:
    dir_path = BASE / sub
    if not dir_path.is_dir():
        continue
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            fp = pathlib.Path(root) / f
            # only text-like files
            if fp.suffix.lower() in {'.txt', '.log', '.md', '.json', '.py', '.csv'}:
                try:
                    text = fp.read_text(errors='ignore').lower()
                except Exception:
                    continue
                for kw in keywords:
                    if kw in text:
                        found_keywords.add(str(fp))

# Write report
with REPORT_PATH.open('w', encoding='utf-8') as out:
    out.write('# NPZ Investigation Report\n\n')
    if not npz_info:
        out.write('**VGG16 feature extraction has never been completed.**\n\n')
    else:
        out.write('## Discovered NPZ Files\n\n')
        for info in npz_info:
            out.write(f"- **Path**: {info['path']}\n")
            out.write(f"  - Size: {info['size']} bytes\n")
            out.write(f"  - Creation time: {info['ctime']}\n")
            out.write(f"  - Modification time: {info['mtime']}\n")
            out.write(f"  - Load success: {info['load_success']}\n")
            if info['load_success']:
                out.write(f"  - Keys: {', '.join(info['keys'])}\n")
                out.write("  - Shapes:\n")
                for k, sh in info['shapes'].items():
                    out.write(f"    - {k}: {sh}\n")
            else:
                out.write(f"  - Load error: {info['error']}\n")
            out.write('\n')
    out.write('## Keyword Search Results (vgg16, lstm, feature, child)\n\n')
    if found_keywords:
        for p in sorted(found_keywords):
            out.write(f"- {p}\n")
    else:
        out.write('No occurrences found in the searched directories.\n')
    out.write('\n')
print('Report written to', REPORT_PATH)
