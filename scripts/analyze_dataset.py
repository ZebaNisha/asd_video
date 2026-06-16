"""Analyze autism_data_anonymized videos and write dataset_stats.json."""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"c:\asd_project\autism_data_anonymized\autism_data_anonymized")
OUT_JSON = Path(r"c:\asd_project\dataset_stats.json")

MOTION_THRESHOLD = 0.35
MOTION_SAMPLES = 4
FACE_SAMPLES = 1  # middle frame only (speed)
_thread_local = threading.local()


@dataclass
class VideoResult:
    path: str
    group: str
    ok: bool = True
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    frames: int = 0
    corrupt_reason: str | None = None
    motion_score: float = 0.0
    face_counts: list[int] = field(default_factory=list)


def _sample_indices(frame_count: int, n: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    return [int(i) for i in np.linspace(0, frame_count - 1, n)]


def _count_faces(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    col_sums = np.sum(gray > 10, axis=0)
    active_cols = np.where(col_sums > 0)[0]
    if len(active_cols) == 0:
        return 0
    groups = []
    current_group = [active_cols[0]]
    for c in active_cols[1:]:
        if c - current_group[-1] <= 30:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)
    valid_groups = [g for g in groups if len(g) > 0 and (g[-1] - g[0]) >= 5]
    return len(valid_groups)


def analyze_video(path: Path, group: str) -> VideoResult:
    rel = path.relative_to(ROOT).as_posix()
    res = VideoResult(path=rel, group=group)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        res.ok = False
        res.corrupt_reason = "cannot_open"
        return res

    res.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    res.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res.fps = float(cap.get(cv2.CAP_PROP_FPS))
    res.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res.duration = res.frames / res.fps if res.fps > 0 else 0.0

    if res.frames <= 0:
        cap.release()
        res.ok = False
        res.corrupt_reason = "zero_frames"
        return res

    motion_idx = _sample_indices(res.frames, MOTION_SAMPLES)
    face_idx = set(_sample_indices(res.frames, FACE_SAMPLES))
    prev_gray = None
    diffs: list[float] = []
    readable = 0

    for idx in motion_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        readable += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
        prev_gray = gray
        if idx in face_idx:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res.face_counts.append(_count_faces(rgb))

    cap.release()

    if readable == 0:
        res.ok = False
        res.corrupt_reason = "unreadable_frames"
        return res

    res.motion_score = float(np.mean(diffs)) if diffs else 0.0
    return res


def bucket_duration(seconds: float) -> str:
    if seconds < 4.6:
        return "< 4.6 s"
    if seconds < 4.8:
        return "4.6 – 4.8 s"
    if seconds < 5.0:
        return "4.8 – 5.0 s"
    if seconds < 5.2:
        return "5.0 – 5.2 s"
    if seconds < 5.4:
        return "5.2 – 5.4 s"
    return ">= 5.4 s"


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks: list[tuple[Path, str]] = []
    for split in ("training_set", "testing_set"):
        for group in ("ASD", "TD"):
            folder = ROOT / split / group
            tasks.extend((p, group) for p in sorted(folder.glob("*.mp4")))

    print(f"Analyzing {len(tasks)} videos...")
    results: list[VideoResult] = []
    workers = min(8, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_video, p, g): p for p, g in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 2500 == 0:
                print(f"  {done}/{len(tasks)}")
            results.append(fut.result())

    ok_results = [r for r in results if r.ok]
    asd = sum(1 for r in results if r.group == "ASD")
    td = sum(1 for r in results if r.group == "TD")

    fps_dist = Counter(round(r.fps, 4) for r in ok_results)
    res_dist = Counter((r.width, r.height) for r in ok_results)
    dur_dist = Counter(bucket_duration(r.duration) for r in ok_results)

    corrupted = [r.path for r in results if not r.ok]
    low_motion = [r.path for r in ok_results if r.motion_score < MOTION_THRESHOLD]

    def max_faces(r: VideoResult) -> int:
        return max(r.face_counts) if r.face_counts else 0

    def min_faces(r: VideoResult) -> int:
        return min(r.face_counts) if r.face_counts else 0

    without_child = [r.path for r in ok_results if max_faces(r) == 0]
    only_examiner = [r.path for r in ok_results if max_faces(r) == 1]

    summary = {
        "total_videos": len(results),
        "asd_videos": asd,
        "td_videos": td,
        "fps_distribution": dict(fps_dist.most_common()),
        "resolution_distribution": {
            f"{w}x{h}": c for (w, h), c in res_dist.most_common()
        },
        "duration_distribution": dict(dur_dist),
        "duration_seconds": {
            "mean": float(np.mean([r.duration for r in ok_results])),
            "min": float(min(r.duration for r in ok_results)),
            "max": float(max(r.duration for r in ok_results)),
        },
        "corrupted_videos": {
            "count": len(corrupted),
            "by_reason": dict(Counter(r.corrupt_reason for r in results if not r.ok)),
            "paths": corrupted,
        },
        "videos_without_child": {
            "count": len(without_child),
            "paths": without_child,
        },
        "videos_with_only_examiner": {
            "count": len(only_examiner),
            "paths": only_examiner,
        },
        "videos_with_missing_motion": {
            "count": len(low_motion),
            "paths": low_motion,
            "threshold": MOTION_THRESHOLD,
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_dataset_md(summary)
    print(f"Wrote {OUT_JSON}")
    print("Corrupted:", len(corrupted))
    print("Missing motion:", len(low_motion))
    print("Without child (0 faces):", len(without_child))
    print("Only examiner (1 face):", len(only_examiner))


def write_dataset_md(summary: dict) -> None:
    md_path = Path(r"c:\asd_project\dataset.md")
    fps = summary["fps_distribution"]
    total_fps = sum(fps.values())
    dur = summary["duration_distribution"]
    total_dur = sum(dur.values())

    def table(counter: dict, total: int, limit: int = 20) -> str:
        lines = ["| Value | Count | % |", "|-------|------:|--:|"]
        for k, v in list(counter.items())[:limit]:
            pct = 100.0 * v / total if total else 0
            lines.append(f"| {k} | {v:,} | {pct:.1f}% |")
        if len(counter) > limit:
            lines.append(f"| … ({len(counter) - limit} more) | | |")
        return "\n".join(lines)

    corr = summary["corrupted_videos"]
    no_child = summary["videos_without_child"]
    examiner = summary["videos_with_only_examiner"]
    motion = summary["videos_with_missing_motion"]
    ds = summary["duration_seconds"]

    body = f"""# Autism Data Anonymized — Dataset Summary

**Root path:** `{ROOT}`

**Generated:** automated scan of all `.mp4` files (`scripts/analyze_dataset.py`).

## Video counts

| Group | Videos |
|-------|-------:|
| **ASD** | {summary['asd_videos']:,} |
| **TD** | {summary['td_videos']:,} |
| **Total** | {summary['total_videos']:,} |

Layout: `training_set/` and `testing_set/`, each with `ASD/` and `TD/` (4,840 videos per folder).

## FPS distribution

{table(fps, total_fps)}

Mean FPS (unweighted over videos): **{sum(float(k) * v for k, v in fps.items()) / total_fps:.2f}**

## Resolution distribution

{table(summary['resolution_distribution'], summary['total_videos'])}

## Duration distribution

| Stat | Seconds |
|------|--------:|
| Mean | {ds['mean']:.4f} |
| Min | {ds['min']:.4f} |
| Max | {ds['max']:.4f} |

### Duration buckets

{table(dur, total_dur)}

## Quality checks

| Category | Count | Notes |
|----------|------:|-------|
| **Corrupted videos** | {corr['count']:,} | Cannot open, zero frames, or unreadable samples |
| **Videos without child** | {no_child['count']:,} | No child detected by the sampled-frame heuristic |
| **Videos with only examiner** | {examiner['count']:,} | Exactly one face in sampled frame (heuristic) |
| **Videos with missing motion** | {motion['count']:,} | Mean frame diff < {motion['threshold']} |

### Corrupted videos

"""
    if corr["count"] == 0:
        body += "None detected.\n"
    else:
        body += f"Reasons: `{corr['by_reason']}`\n\n"
        for p in corr["paths"][:50]:
            body += f"- `{p}`\n"
        if corr["count"] > 50:
            body += f"\n… and {corr['count'] - 50} more (see `dataset_stats.json`).\n"

    body += "\n### Videos without child\n\n"
    if no_child["count"] == 0:
        body += "None detected.\n"
    else:
        for p in no_child["paths"][:30]:
            body += f"- `{p}`\n"
        if no_child["count"] > 30:
            body += f"\n… and {no_child['count'] - 30} more (see `dataset_stats.json`).\n"

    body += "\n### Videos with only examiner\n\n"
    if examiner["count"] == 0:
        body += "None detected.\n"
    else:
        for p in examiner["paths"][:30]:
            body += f"- `{p}`\n"
        if examiner["count"] > 30:
            body += f"\n… and {examiner['count'] - 30} more (see `dataset_stats.json`).\n"

    body += "\n### Videos with missing motion\n\n"
    if motion["count"] == 0:
        body += "None detected.\n"
    else:
        for p in motion["paths"][:30]:
            body += f"- `{p}`\n"
        if motion["count"] > 30:
            body += f"\n… and {motion['count'] - 30} more (see `dataset_stats.json`).\n"

    body += """
## Methodology

- **Corrupted:** OpenCV cannot open the file, reports zero frames, or cannot decode any sampled frame.
- **Missing motion:** Mean absolute grayscale difference between consecutive motion samples < threshold.
- **Without child / only examiner:** Sampled-frame heuristics; these are automated checks, not manual ADOS labels. Clips with occlusion or multiple people may be misclassified.

Full path lists: `dataset_stats.json`.
"""
    md_path.write_text(body, encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
