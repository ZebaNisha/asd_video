#!/usr/bin/env python
"""Extract VGG16 features from child-only crops.

This uses the existing child tracker output:
    outputs/child_sequences/<unique_video_id>_child_sequence.csv

For each video, it crops the tracked child box from selected frames, resizes the
crop to 224x224, runs VGG16, and saves one sequence per video.

Output NPZ files contain:
    X: float32 array shaped (N, T, 512)
    y: int64 labels, ASD=1 and TD=0
    lengths: int64 number of extracted frames before padding
    video_ids: unique_video_id strings
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VGG16 features from tracked child crops.")
    parser.add_argument("--metadata", default="outputs/vgg16_lstm/subset_2subjects_10videos_metadata.csv")
    parser.add_argument("--child-seq-dir", default="outputs/child_sequences")
    parser.add_argument("--out-dir", default="outputs/vgg16_lstm/subset_2subjects_10videos")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_vgg16():
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

    try:
        model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    except Exception as exc:
        print(f"Warning: could not load ImageNet weights ({exc}); falling back to randomly initialized VGG16.")
        model = VGG16(weights=None, include_top=False, pooling="avg")
    return model, preprocess_input


def read_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_to_int(value: str) -> int:
    value = str(value).strip().lower()
    if value in {"1", "asd", "autism"}:
        return 1
    if value in {"0", "td", "typical"}:
        return 0
    raise ValueError(f"Unknown label value: {value!r}")


def load_child_boxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
    boxes = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = int(float(row["frame_number"]))
            cx = float(row["centroid_x"])
            cy = float(row["centroid_y"])
            w = float(row["bbox_width"])
            h = float(row["bbox_height"])
            boxes[frame] = (cx, cy, w, h)
    return boxes


def crop_child(frame: np.ndarray, box: tuple[float, float, float, float], margin: float) -> np.ndarray:
    height, width = frame.shape[:2]
    cx, cy, box_w, box_h = box
    pad_w = box_w * margin
    pad_h = box_h * margin
    x1 = max(0, int(round(cx - box_w / 2 - pad_w)))
    y1 = max(0, int(round(cy - box_h / 2 - pad_h)))
    x2 = min(width, int(round(cx + box_w / 2 + pad_w)))
    y2 = min(height, int(round(cy + box_h / 2 + pad_h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


def select_frames(frame_numbers: list[int], max_frames: int) -> list[int]:
    if len(frame_numbers) <= max_frames:
        return frame_numbers
    indices = np.linspace(0, len(frame_numbers) - 1, max_frames).round().astype(int)
    return [frame_numbers[i] for i in indices]


def extract_video_features(
    video_path: Path,
    boxes: dict[int, tuple[float, float, float, float]],
    max_frames: int,
    margin: float,
    model,
    preprocess_input,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    selected_frames = select_frames(sorted(boxes), max_frames)
    selected_set = set(selected_frames)
    crops = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((max_frames, 512), dtype=np.float32), 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in selected_set:
            crop = crop_child(frame, boxes[frame_idx], margin)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        frame_idx += 1
    cap.release()

    if not crops:
        return np.zeros((max_frames, 512), dtype=np.float32), 0

    images = np.asarray(crops, dtype=np.float32)
    images = preprocess_input(images)
    features = model.predict(images, batch_size=batch_size, verbose=0).astype(np.float32)

    padded = np.zeros((max_frames, features.shape[1]), dtype=np.float32)
    length = min(len(features), max_frames)
    padded[:length] = features[:length]
    return padded, length


def save_split(out_dir: Path, split: str, samples: list[np.ndarray], labels: list[int], lengths: list[int], ids: list[str]) -> None:
    out_path = out_dir / f"vgg16_child_{split}.npz"
    x = np.stack(samples).astype(np.float32) if samples else np.empty((0, 0, 512), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    seq_lengths = np.asarray(lengths, dtype=np.int64)
    video_ids = np.asarray(ids)
    np.savez_compressed(out_path, X=x, y=y, lengths=seq_lengths, video_ids=video_ids)
    print(f"Wrote {out_path}: X={x.shape}, y={y.shape}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "vgg16_child_feature_report.csv"

    train_out = out_dir / "vgg16_child_train.npz"
    test_out = out_dir / "vgg16_child_test.npz"
    if train_out.exists() and test_out.exists() and report_path.exists() and not args.force:
        print(f"Outputs already exist in {out_dir}. Use --force to regenerate.")
        return

    model, preprocess_input = load_vgg16()
    rows = read_metadata(Path(args.metadata))
    child_seq_dir = Path(args.child_seq_dir)

    by_split = {
        "train": {"X": [], "y": [], "lengths": [], "ids": []},
        "test": {"X": [], "y": [], "lengths": [], "ids": []},
    }
    report_rows = []

    for row in rows:
        unique_id = row.get("unique_video_id") or row["video_id"]
        split = str(row["split"]).strip().lower()
        if split not in by_split:
            report_rows.append({"unique_video_id": unique_id, "split": split, "status": "unknown_split", "length": 0})
            continue

        seq_path = child_seq_dir / f"{unique_id}_child_sequence.csv"
        if not seq_path.is_file():
            report_rows.append({"unique_video_id": unique_id, "split": split, "status": "missing_child_sequence", "length": 0})
            continue

        boxes = load_child_boxes(seq_path)
        if not boxes:
            report_rows.append({"unique_video_id": unique_id, "split": split, "status": "empty_child_sequence", "length": 0})
            continue

        video_path = Path(row["dataset_path"])
        features, length = extract_video_features(
            video_path,
            boxes,
            args.max_frames,
            args.crop_margin,
            model,
            preprocess_input,
            args.batch_size,
        )
        if length == 0:
            report_rows.append({"unique_video_id": unique_id, "split": split, "status": "video_or_crop_failed", "length": 0})
            continue

        by_split[split]["X"].append(features)
        by_split[split]["y"].append(label_to_int(row["label"]))
        by_split[split]["lengths"].append(length)
        by_split[split]["ids"].append(unique_id)
        report_rows.append({"unique_video_id": unique_id, "split": split, "status": "ok", "length": length})

    for split, data in by_split.items():
        save_split(out_dir, split, data["X"], data["y"], data["lengths"], data["ids"])

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["unique_video_id", "split", "status", "length"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
