# diagnose_dataset.py
"""Diagnostic script to verify dataset discovery and video selection.

Prints the number of videos found in each of the four expected folders and the
counts of videos selected for each split/label based on the same logic used by
`run_pipeline.py`.
"""

import argparse
from pathlib import Path
import sys

# Import the central configuration (PROJECT_ROOT points to the repo root)
from path_config import PROJECT_ROOT


def count_videos(folder: Path) -> int:
    """Return the number of *.mp4 files under *folder* (recursively)."""
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.rglob("*.mp4"))


def discover_videos(data_root: Path) -> list[Path]:
    """Collect all video files under the four dataset folders."""
    videos = []
    for split in ("training_set", "testing_set"):
        for label in ("asd", "td"):
            folder = data_root / split / label
            if folder.is_dir():
                videos.extend(sorted(folder.rglob("*.mp4")))
    return videos


def select_videos(
    data_root: Path,
    train_asd: int | None,
    train_td: int | None,
    test_asd: int | None,
    test_td: int | None,
    max_videos: int | None,
) -> list[Path]:
    """Select a balanced set of videos, mirroring the logic in run_pipeline.py.

    If any of the per‑split counts are provided, those exact numbers are taken;
    otherwise the function falls back to returning all discovered videos (or
    truncating to ``max_videos`` when that argument is set).
    """
    if any(v is not None for v in (train_asd, train_td, test_asd, test_td)):
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
                print(f"[WARN] Missing folder: {folder}", file=sys.stderr)
                continue
            pool = sorted(folder.rglob("*.mp4"))
            if len(pool) < count:
                print(
                    f"[WARN] Requested {count} videos from {folder} but only found {len(pool)}",
                    file=sys.stderr,
                )
            selected.extend(pool[:count])
        selected.sort()
        return selected

    # No per‑split counts: fall back to full discovery (optionally limited)
    videos = discover_videos(data_root)
    if max_videos is not None:
        videos = videos[:max_videos]
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset discovery and selection diagnostics")
    parser.add_argument("--train-asd", type=int, default=None, help="Training ASD videos to sample")
    parser.add_argument("--train-td", type=int, default=None, help="Training TD videos to sample")
    parser.add_argument("--test-asd", type=int, default=None, help="Testing ASD videos to sample")
    parser.add_argument("--test-td", type=int, default=None, help="Testing TD videos to sample")
    parser.add_argument("--max-videos", type=int, default=None, help="Legacy max‑videos fallback")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "autism_data_anonymized" / "autism_data_anonymized"

    # 1️⃣ Print folder discovery counts
    print("Folder discovery counts:")
    for split in ("training_set", "testing_set"):
        for label in ("asd", "td"):
            folder = data_root / split / label
            count = count_videos(folder)
            print(f"{split}/{label} : {count}")
    print()

    # 2️⃣ Perform selection using the same logic as run_pipeline
    selected = select_videos(
        data_root,
        args.train_asd,
        args.train_td,
        args.test_asd,
        args.test_td,
        args.max_videos,
    )

    # Count selected videos per group
    counts = {
        "train_asd": 0,
        "train_td": 0,
        "test_asd": 0,
        "test_td": 0,
    }
    for vid in selected:
        rel = vid.relative_to(data_root)
        split = rel.parts[0]  # "training_set" or "testing_set"
        label = rel.parts[1].lower()  # "asd" or "td"
        key = ("train" if split.startswith("training") else "test") + "_" + label
        if key in counts:
            counts[key] += 1

    print("Selected videos:")
    for key in ("train_asd", "train_td", "test_asd", "test_td"):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
