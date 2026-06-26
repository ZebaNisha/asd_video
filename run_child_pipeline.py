
# Build set of detection base names (without _detections)
detected = {p.stem.replace('_detections', '') for p in detections_dir.glob('*_detections.csv')}

for tracked_file in tracked_dir.glob('*_tracked.csv'):
    base = tracked_file.stem.replace('_tracked', '')
    if base not in detected:
        continue
    # Run extract_child_track.py
    try:
        subprocess.run([
            sys.executable,
            str(project_root / 'scripts' / 'extract_child_track.py'),
            '--input', str(tracked_file)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"extract_child_track failed for {tracked_file}: {e}")
        continue

    report_path = out_dir / f"{base}_child_report.csv"
    sequence_path = out_dir / f"{base}_child_sequence.csv"
    # If both report and sequence already exist, skip this video
    if report_path.is_file() and sequence_path.is_file():
        continue
    # Run extract_child_sequence.py

    try:
        subprocess.run([
            sys.executable,
            str(project_root / 'scripts' / 'extract_child_sequence.py'),
            '--tracked', str(tracked_file),
            '--report', str(report_path),
            '--output-dir', str(out_dir)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"extract_child_sequence failed for {tracked_file}: {e}")
        continue

print('Filtered batch processing completed.')
