#!/usr/bin/env python
"""run_pipeline_mp.py

Multiprocessing version of the ASD video preprocessing pipeline.
It mirrors the logic of ``run_pipeline.py`` but processes videos in parallel
using ``concurrent.futures.ProcessPoolExecutor``.  The script respects the
``--resume`` flag and a ``--workers`` option to control the number of parallel
processes.  All sub‑scripts are invoked via ``run_subscript`` which runs them as
separate processes, so each worker is isolated and safe on Windows.

Key safety measures:
- ``if __name__ == '__main__'`` guard (required for ``spawn`` start method).
- State file ``pipeline_state.json`` is read before workers start and written
  once after all workers finish, avoiding concurrent writes.
- ``force`` and ``resume`` semantics are identical to the original script.
- Errors in individual videos are caught and reported; they do not abort the
  entire pool.

Usage example:
    python C:/asd_project/scripts/run_pipeline_mp.py --resume --workers 6
"""

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess

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

log_file = LOGS_DIR / "pipeline_mp.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Helper utilities – identical to the original script
# ---------------------------------------------------------------------------

def discover_videos(data_root: Path) -> list[Path]:
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
    # Subject‑level selection logic (unchanged from the original script)
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
        for folder, subj_count, _display_name in buckets:
            if subj_count is None or subj_count <= 0:
                continue
            if not folder.is_dir():
                continue
            pool = sorted(folder.rglob("*.mp4"))
            subject_to_videos: dict[str, list[Path]] = {}
            for path in pool:
                subj_id = path.stem.split("_part_")[0]
                subject_to_videos.setdefault(subj_id, []).append(path)
            unique_subjects = sorted(subject_to_videos.keys())
            if len(unique_subjects) < subj_count:
                subj_count = len(unique_subjects)
            selected_subjects = sorted(random.Random(seed).sample(unique_subjects, subj_count))
            for subj in selected_subjects:
                selected.extend(subject_to_videos[subj])
        selected.sort()
        return selected

    # Explicit split counts logic (unchanged)
    if any(n is not None for n in (train_asd, train_td, test_asd, test_td)):
        buckets = [
            (data_root / "training_set" / "asd", train_asd),
            (data_root / "training_set" / "td", train_td),
            (data_root / "testing_set" / "asd", test_asd),
            (data_root / "testing_set" / "td", test_td),
        ]
        selected: list[Path] = []
        for folder, count in buckets:
            if count is None or count <= 0:
                continue
            if not folder.is_dir():
                continue
            pool = sorted(folder.rglob("*.mp4"))
            selected.extend(pool[:count])
        selected.sort()
        return selected

    # Fallback – use all videos (or limit with max_videos)
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

def run_subscript(script_name: str, args: list) -> None:
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

def get_unique_video_id(video_path: Path, data_root: Path) -> str:
    rel = video_path.relative_to(data_root)
    split_raw = rel.parts[0]
    split = "train" if split_raw.startswith("training") else "test"
    label = rel.parts[1].lower()
    video_id = video_path.stem
    return f"{split}_{label}_{video_id}"

# ---------------------------------------------------------------------------
# Worker function – executed in a separate process
# ---------------------------------------------------------------------------

def worker_process(video_path_str: str, data_root_str: str, force: bool) -> tuple[str, str | None]:
    """Process a single video.

    Returns a tuple ``(status, unique_video_id)`` where ``status`` is one of
    ``"processed"``, ``"skipped"`` or ``"failed"``.  ``unique_video_id`` is only
    provided for the ``"processed"`` case.
    """
    video_path = Path(video_path_str)
    data_root = Path(data_root_str)
    unique_video_id = get_unique_video_id(video_path, data_root)
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
    if already_done and not force:
        return ("skipped", None)
    try:
        # Run the five sub‑scripts in order
        run_subscript("detect_skeletons.py", ["--input", str(video_path), "--unique-id", unique_video_id])
        run_subscript(
            "centroid_tracker.py",
            ["--input", str(detection_csv), "--output", str(tracked_csv), "--max-distance", "150", "--max-disappeared", "10"],
        )
        run_subscript("analyze_tracks.py", ["--input", str(tracked_csv), "--output", str(track_analysis_csv)])
        run_subscript("extract_child_track.py", ["--input", str(tracked_csv), "--output", str(child_report_csv)])
        run_subscript(
            "extract_child_sequence.py",
            ["--tracked", str(tracked_csv), "--report", str(child_report_csv), "--output-dir", str(CHILD_SEQUENCES_DIR)],
        )
        return ("processed", unique_video_id)
    except Exception as e:
        logging.exception("Failed processing %s: %s", video_path.name, e)
        return ("failed", None)

# ---------------------------------------------------------------------------
# Main entry point (mirrors original script but uses a pool)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multiprocess ASD video preprocessing pipeline.")
    parser.add_argument("--max-videos", type=int, default=None, help="Process at most N videos (fallback when split counts not set).")
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
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: CPU count).")
    args = parser.parse_args()
    logging.info("Parsed arguments: %s", args)

    data_root = PROJECT_ROOT / "autism_data_anonymized" / "autism_data_anonymized"
    all_videos = discover_videos(data_root)
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
    logging.info("Discovered %s video(s) in dataset", len(all_videos))
    logging.info("Selected %s video(s) for this run", len(videos))

    # State handling
    state_path = LOGS_DIR / "pipeline_state.json"
    state = load_state(state_path) if args.resume else {"processed": []}
    processed_set = set(state.get("processed", []))

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Prepare work list – filter out already‑done videos before spawning workers
    work_items: list[tuple[str, str, bool]] = []  # (video_path_str, data_root_str, force)
    for video_path in videos:
        unique_video_id = get_unique_video_id(video_path, data_root)
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
            skipped_count += 1
            continue
        if unique_video_id in processed_set and not args.force:
            skipped_count += 1
            continue
        work_items.append((str(video_path), str(data_root), args.force))

    total_to_process = len(work_items)
    logging.info("Submitting %s videos to the pool (workers=%s)", total_to_process, args.workers)

    # ---------------------------------------------------------------------
    # Run pool
    # ---------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Map each work item to a future
        future_to_video = {executor.submit(worker_process, *item): item[0] for item in work_items}
        for future in as_completed(future_to_video):
            video_str = future_to_video[future]
            try:
                status, uid = future.result()
                if status == "processed":
                    processed_count += 1
                    processed_set.add(uid)  # type: ignore[arg-type]
                elif status == "skipped":
                    skipped_count += 1
                else:  # failed
                    failed_count += 1
                logging.info("[%s] %s", status.upper(), Path(video_str).name)
            except Exception as e:
                failed_count += 1
                logging.exception("Unexpected error processing %s", video_str)

    # Save updated state after the pool finishes
    save_state(state_path, {"processed": list(processed_set)})

    # ---------------------------------------------------------------------
    # Final stages – identical to the original script (backfill, filtering, etc.)
    # ---------------------------------------------------------------------
    try:
        # Backfill any missing child sequence files
        from pathlib import Path as _Path  # local import to avoid circular refs
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
                ["--tracked", str(tracked_path), "--report", str(report_path), "--output-dir", str(CHILD_SEQUENCES_DIR)],
            )
            created += 1
        if created:
            logging.info("Backfilled %s child sequence file(s)", created)

        # Filtering, feature extraction, labeling – same as original
        run_subscript("filter_child_sequences.py", [])
        accepted_path = FILTERING_DIR / "accepted_videos.txt"
        if not accepted_path.is_file():
            raise RuntimeError("filter_child_sequences.py did not produce accepted_videos.txt")
        run_subscript("build_feature_dataset.py", [])
        features_path = FEATURES_DIR / "features.csv"
        if not features_path.is_file() or features_path.stat().st_size == 0:
            raise RuntimeError("build_feature_dataset.py did not produce a non‑empty features.csv")
        run_subscript("create_labeled_dataset.py", [])
        labeled_path = FEATURES_DIR / "labeled_features.csv"
        if not labeled_path.is_file() or labeled_path.stat().st_size == 0:
            raise RuntimeError("create_labeled_dataset.py did not produce a non‑empty labeled_features.csv")
    except Exception as e:
        logging.exception("Pipeline final stages failed: %s", e)
        sys.exit(1)

    # ---------------------------------------------------------------------
    # Summary report (mirrors original script)
    # ---------------------------------------------------------------------
    accepted_path = FILTERING_DIR / "accepted_videos.txt"
    rejected_path = FILTERING_DIR / "rejected_videos.csv"
    features_path = FEATURES_DIR / "features.csv"
    labeled_path = FEATURES_DIR / "labeled_features.csv"

    def count_lines(path: Path, skip_header: bool = True) -> int:
        if not path.is_file():
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f) - (1 if skip_header else 0)

    accepted_count = count_lines(accepted_path, skip_header=False)
    rejected_count = count_lines(rejected_path)  # header present
    feature_rows = count_lines(features_path)
    labeled_rows = count_lines(labeled_path)

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
    logging.info("Pipeline summary written to %s", summary_path)
    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
