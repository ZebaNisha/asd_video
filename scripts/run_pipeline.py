import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from path_config import (
    CHILD_SEQUENCES_DIR,
    DETECTIONS_DIR,
    FEATURES_DIR,
    FILTERING_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    REPORTS_ROOT,
    SCRIPTS_DIR,
    TRACKED_DIR,
    TRACK_ANALYSIS_DIR,
)

log_file = LOGS_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


def discover_videos(data_root: Path) -> list[Path]:
    """Return all video paths under training_set/testing_set asd/td folders."""
    video_paths: list[Path] = []
    for split in ("training_set", "testing_set"):
        for label in ("asd", "td"):
            folder = data_root / split / label
            if folder.is_dir():
                video_paths.extend(folder.rglob("*.mp4"))
    video_paths.sort()
    return video_paths


def select_videos(
    data_root: Path,
    train_asd: int | None,
    train_td: int | None,
    test_asd: int | None,
    test_td: int | None,
    max_videos: int | None,
    train_asd_subjects: int | None = None,
    train_td_subjects: int | None = None,
    test_asd_subjects: int | None = None,
    test_td_subjects: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Build a balanced video list from subject-level selection, explicit clip counts, or max-videos fallback."""
    use_subjects = any(n is not None for n in (train_asd_subjects, train_td_subjects, test_asd_subjects, test_td_subjects))
    
    if use_subjects:
        import random
        buckets = [
            (data_root / "training_set" / "asd", train_asd_subjects, "Training ASD"),
            (data_root / "training_set" / "td", train_td_subjects, "Training TD"),
            (data_root / "testing_set" / "asd", test_asd_subjects, "Testing ASD"),
            (data_root / "testing_set" / "td", test_td_subjects, "Testing TD"),
        ]
        selected: list[Path] = []
        results_summary = []
        
        for folder, subj_count, display_name in buckets:
            if subj_count is None or subj_count <= 0:
                results_summary.append((display_name, 0, 0))
                continue
            if not folder.is_dir():
                logging.warning("Missing folder: %s", folder)
                results_summary.append((display_name, 0, 0))
                continue
            pool = sorted(folder.rglob("*.mp4"))
            
            # Group by subject ID using splitting by _part_
            subject_to_videos = {}
            for path in pool:
                subj_id = path.stem.split("_part_")[0]
                if subj_id not in subject_to_videos:
                    subject_to_videos[subj_id] = []
                subject_to_videos[subj_id].append(path)
            
            unique_subjects = sorted(list(subject_to_videos.keys()))
            if len(unique_subjects) < subj_count:
                logging.warning(
                    "Only %s subjects in %s (requested %s)", len(unique_subjects), folder, subj_count
                )
                subj_count = len(unique_subjects)
                
            # Random sampling using reproducible random seed
            selected_subjects = sorted(random.Random(seed).sample(unique_subjects, subj_count))
            
            print(f"Selected {display_name} Subjects:")
            for subj in selected_subjects:
                print(subj)
            print()
            
            num_videos = 0
            for subj in selected_subjects:
                subj_videos = subject_to_videos[subj]
                selected.extend(subj_videos)
                num_videos += len(subj_videos)
            
            results_summary.append((display_name, len(selected_subjects), num_videos))
            
        name_map = {
            "Training ASD": "Train ASD",
            "Training TD": "Train TD",
            "Testing ASD": "Test ASD",
            "Testing TD": "Test TD"
        }
        
        for display_name, sub_cnt, vid_cnt in results_summary:
            mapped_name = name_map.get(display_name, display_name)
            subj_label = f"{mapped_name} subjects".ljust(18)
            vid_label = f"{mapped_name} videos".ljust(18)
            print(f"{subj_label} : {sub_cnt}")
            print(f"{vid_label} : {vid_cnt}")
            print()
            
        selected.sort()
        return selected

    if any(n is not None for n in (train_asd, train_td, test_asd, test_td)):
        buckets = [
            (data_root / "training_set" / "asd", train_asd),
            (data_root / "training_set" / "td", train_td),
            (data_root / "testing_set" / "asd", test_asd),
            (data_root / "testing_set" / "td", test_td),
        ]
        selected = []
        for folder, count in buckets:
            if count is None or count <= 0:
                continue
            if not folder.is_dir():
                logging.warning("Missing folder: %s", folder)
                continue
            pool = sorted(folder.rglob("*.mp4"))
            if len(pool) < count:
                logging.warning(
                    "Only %s videos in %s (requested %s)", len(pool), folder, count
                )
            selected.extend(pool[:count])
        selected.sort()
        return selected

    videos = discover_videos(data_root)
    if max_videos is not None:
        videos = videos[:max_videos]
    return videos


def load_state(state_path: Path) -> dict:
    if state_path.is_file():
        try:
            with open(state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning("Failed to read state file %s: %s", state_path, e)
    return {"processed": []}


def save_state(state_path: Path, state: dict) -> None:
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.error("Unable to write state file %s: %s", state_path, e)


def backfill_child_sequences() -> int:
    """Create missing *_child_sequence.csv files from existing tracked + report pairs."""
    created = 0
    for report_path in sorted(CHILD_SEQUENCES_DIR.glob("*_child_report.csv")):
        stem = report_path.stem.replace("_child_report", "")
        sequence_path = CHILD_SEQUENCES_DIR / f"{stem}_child_sequence.csv"
        if sequence_path.is_file():
            continue
        tracked_path = TRACKED_DIR / f"{stem}_tracked.csv"
        if not tracked_path.is_file():
            logging.warning("Cannot backfill %s: missing %s", stem, tracked_path.name)
            continue
        logging.info("Backfilling child sequence for %s", stem)
        run_subscript(
            "extract_child_sequence.py",
            [
                "--tracked",
                str(tracked_path),
                "--report",
                str(report_path),
                "--output-dir",
                str(CHILD_SEQUENCES_DIR),
            ],
        )
        created += 1
    if created:
        logging.info("Backfilled %s child sequence file(s)", created)
    return created


def run_subscript(script_name: str, args: list) -> subprocess.CompletedProcess:
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(
            "%s failed (exit %s)\nStdout: %s\nStderr: %s",
            script_name,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(f"{script_name} failed")
    logging.info("%s succeeded", script_name)
    return result


def get_unique_video_id(video_path: Path, data_root: Path) -> str:
    rel = video_path.relative_to(data_root)
    split_raw = rel.parts[0]
    split = "train" if split_raw.startswith("training") else "test"
    label = rel.parts[1].lower()
    video_id = video_path.stem
    return f"{split}_{label}_{video_id}"


def save_selected_videos(selected_videos: list[Path]) -> None:
    """Write selected video list to reports/selected_videos.csv.

    Columns (ordered):
        unique_video_id, video_id, dataset_path, split, label
    """
    data_root = PROJECT_ROOT / "autism_data_anonymized" / "autism_data_anonymized"
    csv_path = REPORTS_ROOT / "selected_videos.csv"
    # Build rows with unique identifier
    rows = []
    for vid in selected_videos:
        rel = vid.relative_to(data_root)
        split_raw = rel.parts[0]
        split = "train" if split_raw.startswith("training") else "test"
        label = rel.parts[1].lower()
        video_id = vid.stem
        unique_video_id = get_unique_video_id(vid, data_root)
        rows.append([unique_video_id, video_id, str(vid), split, label])
    # Write CSV with ordered columns
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_video_id", "video_id", "dataset_path", "split", "label"])
        writer.writerows(rows)
    # Validation: ensure each group has at least one video
    counts = {"train_asd": 0, "train_td": 0, "test_asd": 0, "test_td": 0}
    for uid, vid, path, split, label in rows:
        key = f"{split}_{label}"
        if key in counts:
            counts[key] += 1
    missing = [k for k, v in counts.items() if v == 0]
    if missing:
        raise RuntimeError(f"No videos selected for groups: {', '.join(missing)}")
    logging.info("Selected videos written to %s", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full ASD video preprocessing pipeline."
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process at most N videos (used if per-split counts are not set).",
    )
    parser.add_argument("--train-asd", type=int, default=None, help="Training ASD videos to sample.")
    parser.add_argument("--train-td", type=int, default=None, help="Training TD videos to sample.")
    parser.add_argument("--test-asd", type=int, default=None, help="Testing ASD videos to sample.")
    parser.add_argument("--test-td", type=int, default=None, help="Testing TD videos to sample.")
    parser.add_argument("--train-asd-subjects", type=int, default=None, help="Training ASD subjects to sample.")
    parser.add_argument("--train-td-subjects", type=int, default=None, help="Training TD subjects to sample.")
    parser.add_argument("--test-asd-subjects", type=int, default=None, help="Testing ASD subjects to sample.")
    parser.add_argument("--test-td-subjects", type=int, default=None, help="Testing TD subjects to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject sampling.")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist.")
    parser.add_argument("--resume", action="store_true", help="Resume from pipeline_state.json.")
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    data_root = PROJECT_ROOT / "autism_data_anonymized" / "autism_data_anonymized"
    all_videos = discover_videos(data_root)
    # Select videos based on per‑split counts, subject‑level counts or max‑videos fallback
    videos = select_videos(
        data_root,
        args.train_asd,
        args.train_td,
        args.test_asd,
        args.test_td,
        args.max_videos,
        args.train_asd_subjects,
        args.train_td_subjects,
        args.test_asd_subjects,
        args.test_td_subjects,
        args.seed,
    )
    
    # Log selected counts for each group (debug)


    logging.info("Discovered %s video(s) in dataset", len(all_videos))
    logging.info("Selected %s video(s) for this run", len(videos))

    save_selected_videos(videos)

    state_path = LOGS_DIR / "pipeline_state.json"
    state = load_state(state_path) if args.resume else {"processed": []}
    processed_set = set(state.get("processed", []))

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, video_path in enumerate(videos, start=1):
        # Compute a unique identifier that includes split and label to avoid collisions
        unique_video_id = get_unique_video_id(video_path, data_root)
        # Use the unique identifier for all intermediate file names
        detection_csv = DETECTIONS_DIR / f"{unique_video_id}_detections.csv"
        tracked_csv = TRACKED_DIR / f"{unique_video_id}_tracked.csv"
        track_analysis_csv = TRACK_ANALYSIS_DIR / f"{unique_video_id}_track_analysis.csv"
        child_report_csv = CHILD_SEQUENCES_DIR / f"{unique_video_id}_child_report.csv"
        child_sequence_csv = CHILD_SEQUENCES_DIR / f"{unique_video_id}_child_sequence.csv"

        already_done = (
            detection_csv.is_file()
            and tracked_csv.is_file()
            and track_analysis_csv.is_file()
            and child_report_csv.is_file()
            and child_sequence_csv.is_file()
        )
        if already_done and not args.force:
            logging.info("[%s/%s] Skipping %s (already processed)", idx, len(videos), video_path.name)
            skipped_count += 1
            continue
        if unique_video_id in processed_set and not args.force:
            logging.info("[%s/%s] Skipping %s (in state file)", idx, len(videos), video_path.name)
            skipped_count += 1
            continue

        logging.info("[%s/%s] Processing %s", idx, len(videos), video_path.name)
        try:
            # Pass the unique identifier to detect_skeletons to generate correctly named outputs
            run_subscript("detect_skeletons.py", ["--input", str(video_path), "--unique-id", unique_video_id])
            run_subscript(
                "centroid_tracker.py",
                [
                    "--input",
                    str(detection_csv),
                    "--output",
                    str(tracked_csv),
                    "--max-distance",
                    "150",
                    "--max-disappeared",
                    "10",
                ],
            )
            run_subscript(
                "analyze_tracks.py",
                ["--input", str(tracked_csv), "--output", str(track_analysis_csv)],
            )
            run_subscript(
                "extract_child_track.py",
                ["--input", str(tracked_csv), "--output", str(child_report_csv)],
            )
            run_subscript(
                "extract_child_sequence.py",
                [
                    "--tracked",
                    str(tracked_csv),
                    "--report",
                    str(child_report_csv),
                    "--output-dir",
                    str(CHILD_SEQUENCES_DIR),
                ],
            )
            processed_set.add(unique_video_id)
            processed_count += 1
            save_state(state_path, {"processed": list(processed_set)})
        except Exception as e:
            logging.exception("Failed processing %s: %s", video_path.name, e)
            failed_count += 1
            continue

    backfill_child_sequences()

    try:
        run_subscript("filter_child_sequences.py", [])
        accepted_path = FILTERING_DIR / "accepted_videos.txt"
        if not accepted_path.is_file():
            raise RuntimeError("filter_child_sequences.py did not produce accepted_videos.txt")

        run_subscript("build_feature_dataset.py", [])
        features_path = FEATURES_DIR / "features.csv"
        if not features_path.is_file() or features_path.stat().st_size == 0:
            raise RuntimeError("build_feature_dataset.py did not produce a non-empty features.csv")

        run_subscript("create_labeled_dataset.py", [])
        labeled_path = FEATURES_DIR / "labeled_features.csv"
        if not labeled_path.is_file() or labeled_path.stat().st_size == 0:
            raise RuntimeError("create_labeled_dataset.py did not produce a non-empty labeled_features.csv")
    except Exception as e:
        logging.exception("Pipeline final stages failed: %s", e)
        sys.exit(1)

    accepted_path = FILTERING_DIR / "accepted_videos.txt"
    rejected_path = FILTERING_DIR / "rejected_videos.csv"
    features_path = FEATURES_DIR / "features.csv"
    labeled_path = FEATURES_DIR / "labeled_features.csv"

    accepted_count = 0
    if accepted_path.is_file():
        with open(accepted_path, encoding="utf-8") as f:
            accepted_count = sum(1 for _ in f)
    rejected_count = 0
    if rejected_path.is_file():
        with open(rejected_path, encoding="utf-8") as f:
            rejected_count = max(0, sum(1 for _ in f) - 1)
    feature_rows = 0
    if features_path.is_file():
        with open(features_path, encoding="utf-8") as f:
            feature_rows = max(0, sum(1 for _ in f) - 1)
    labeled_rows = 0
    if labeled_path.is_file():
        with open(labeled_path, encoding="utf-8") as f:
            labeled_rows = max(0, sum(1 for _ in f) - 1)

    summary_lines = [
        f"Total videos in dataset: {len(all_videos)}",
        f"Videos selected this run: {len(videos)}",
        f"Videos processed: {processed_count}",
        f"Videos skipped: {skipped_count}",
        f"Videos failed: {failed_count}",
        f"Accepted child sequences: {accepted_count}",
        f"Rejected child sequences: {rejected_count}",
        f"Total feature rows: {feature_rows}",
        f"Total labeled rows: {labeled_rows}",
    ]
    summary_path = REPORTS_ROOT / "pipeline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    # ----------------------------------------------------------------------
    # 7️⃣  Diagnostic audit report (counts by stage and group)
    # ----------------------------------------------------------------------
    # Build a stem → (split, label) map using the selected video list
    stem_to_group: dict[str, str] = {}
    for vid in videos:
        rel = vid.relative_to(data_root)
        split = rel.parts[0].replace("_set", "")  # training or testing
        label = rel.parts[1].lower()                # asd or td
        key = f"{split}_{label}"  # e.g., training_asd, testing_td
        stem_to_group[get_unique_video_id(vid, data_root)] = key

    # Helper to aggregate counts per group for a list of stems
    def aggregate_counts(stems: list[str]) -> dict[str, int]:
        counts = {"training_asd": 0, "training_td": 0, "testing_asd": 0, "testing_td": 0}
        for s in stems:
            grp = stem_to_group.get(s)
            if grp:
                counts[grp] += 1
        return counts

    # Stage 1: selected videos (already have `videos` list)
    selected_counts = {"training_asd": 0, "training_td": 0, "testing_asd": 0, "testing_td": 0}
    for vid in videos:
        rel = vid.relative_to(data_root)
        split = rel.parts[0].replace("_set", "")  # training or testing
        label = rel.parts[1].lower()                # asd or td
        key = f"{split}_{label}"
        if key in selected_counts:
            selected_counts[key] += 1

    # Stage 2‑4: count files in each output directory
    detection_stems = [p.stem.replace("_detections", "") for p in DETECTIONS_DIR.glob("*_detections.csv")]
    track_analysis_stems = [p.stem.replace("_track_analysis", "") for p in TRACK_ANALYSIS_DIR.glob("*_track_analysis.csv")]
    child_seq_stems = [p.stem.replace("_child_sequence", "") for p in CHILD_SEQUENCES_DIR.glob("*_child_sequence.csv")]

    detection_counts = aggregate_counts(detection_stems)
    track_counts = aggregate_counts(track_analysis_stems)
    child_seq_counts = aggregate_counts(child_seq_stems)

    # Stage 5: accepted videos (read accepted_videos.txt)
    accepted_stems = []
    if accepted_path.is_file():
        with open(accepted_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # accepted_videos.txt contains video IDs (stem) per line
                accepted_stems.append(line)
    accepted_counts = aggregate_counts(accepted_stems)

    # Stage 6: feature rows (features.csv) – each row has a video_id column
    feature_stems = []
    if features_path.is_file():
        with open(features_path, "r", encoding="utf-8") as f:
            header = next(f)
            idx = header.split(",").index("video_id")
            for row in f:
                cols = row.strip().split(",")
                if len(cols) <= idx:
                    continue
                feature_stems.append(cols[idx])
    feature_counts = aggregate_counts(feature_stems)

    # Stage 7: labeled rows (labeled_features.csv) – also indexed by video_id
    labeled_stems = []
    if labeled_path.is_file():
        with open(labeled_path, "r", encoding="utf-8") as f:
            header = next(f)
            idx = header.split(",").index("video_id")
            for row in f:
                cols = row.strip().split(",")
                if len(cols) <= idx:
                    continue
                labeled_stems.append(cols[idx])
    labeled_counts = aggregate_counts(labeled_stems)

    # Write audit CSV
    audit_path = REPORTS_ROOT / "pipeline_audit.csv"
    with open(audit_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["stage", "train_asd", "train_td", "test_asd", "test_td"])
        def row(stage, cnts):
            return [stage, cnts["training_asd"], cnts["training_td"], cnts["testing_asd"], cnts["testing_td"]]
        writer.writerow(row("selected_videos", selected_counts))
        writer.writerow(row("detections", detection_counts))
        writer.writerow(row("track_analysis", track_counts))
        writer.writerow(row("child_sequences", child_seq_counts))
        writer.writerow(row("accepted", accepted_counts))
        writer.writerow(row("feature_rows", feature_counts))
        writer.writerow(row("labeled_rows", labeled_counts))
    logging.info("Pipeline audit written to %s", audit_path)
    
    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
