#!/usr/bin/env python
"""Create a balanced subject subset metadata CSV for child-only video experiments."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced subset metadata CSV.")
    parser.add_argument("--input", default="outputs/features/labeled_features.csv")
    parser.add_argument("--output", default="outputs/vgg16_lstm/subset_2subjects_metadata.csv")
    parser.add_argument("--subjects-per-group", type=int, default=2)
    parser.add_argument(
        "--max-videos-per-subject",
        type=int,
        default=None,
        help="Optional cap for a faster smoke test. By default all videos for selected subjects are kept.",
    )
    return parser.parse_args()


def subject_id(video_id: str) -> str:
    return video_id.split("_part_")[0]


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    if value in {"1", "asd", "autism"}:
        return "asd"
    if value in {"0", "td", "typical"}:
        return "td"
    raise ValueError(f"Unknown label value: {label!r}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"No columns found in {input_path}")

    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        split = row["split"].strip().lower()
        label = normalize_label(row["label"])
        grouped[(split, label)][subject_id(row["video_id"])].append(row)

    selected_rows = []
    summary = []
    for split, label in [("train", "asd"), ("train", "td"), ("test", "asd"), ("test", "td")]:
        subjects = sorted(grouped[(split, label)])
        chosen_subjects = subjects[: args.subjects_per_group]
        if len(chosen_subjects) < args.subjects_per_group:
            raise RuntimeError(
                f"Only found {len(chosen_subjects)} subjects for split={split}, label={label}"
            )

        for subj in chosen_subjects:
            subj_rows = sorted(grouped[(split, label)][subj], key=lambda r: r["unique_video_id"])
            if args.max_videos_per_subject is not None:
                subj_rows = subj_rows[: args.max_videos_per_subject]
            selected_rows.extend(subj_rows)
            summary.append((split, label, subj, len(subj_rows)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Wrote {output_path} with {len(selected_rows)} rows")
    for split, label, subj, count in summary:
        print(f"{split:5s} {label:3s} {subj:8s} {count:4d} videos")


if __name__ == "__main__":
    main()
