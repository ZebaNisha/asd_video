"""
Evaluate label agreement between selected videos and pipeline labels.

Compares ground truth from `reports/selected_videos.csv` to labels in
`outputs/features/labeled_features.csv`, aligned on `video_id`.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(r"C:\asd_project")
SELECTED_PATH = PROJECT_ROOT / "reports" / "selected_videos.csv"
LABELED_PATH = PROJECT_ROOT / "outputs" / "features" / "labeled_features.csv"


def label_to_int(value) -> int:
    """Map asd/td or 0/1 to integer label (ASD=1, TD=0)."""
    if pd.isna(value):
        return -1
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in ("1", "asd"):
        return 1
    if text in ("0", "td"):
        return 0
    raise ValueError(f"Unknown label: {value!r}")


def load_selected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        if "video_path" not in df.columns:
            raise KeyError(f"{path} must contain video_path or video_id")
        df = df.copy()
        df["video_id"] = df["video_path"].apply(lambda p: Path(str(p)).stem)
    df["label_gt"] = df["label"].apply(label_to_int)
    return df[["video_id", "label_gt"]].drop_duplicates("video_id")


def load_labeled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        raise KeyError(f"{path} must contain video_id")
    out = df.copy()
    out["label_pred"] = out["label"].apply(label_to_int)
    return out[["video_id", "label_pred"]].drop_duplicates("video_id")


def main() -> None:
    if not SELECTED_PATH.is_file():
        raise FileNotFoundError(f"Missing {SELECTED_PATH}")
    if not LABELED_PATH.is_file():
        raise FileNotFoundError(f"Missing {LABELED_PATH}")

    selected_df = load_selected(SELECTED_PATH)
    labeled_df = load_labeled(LABELED_PATH)

    merged = pd.merge(selected_df, labeled_df, on="video_id", how="inner")
    if merged.empty:
        raise RuntimeError(
            "No matching video_id rows between selected_videos.csv and labeled_features.csv"
        )

    missing_in_labeled = set(selected_df["video_id"]) - set(merged["video_id"])
    if missing_in_labeled:
        print(f"Warning: {len(missing_in_labeled)} selected videos not in labeled_features.csv")

    y_true = merged["label_gt"]
    y_pred = merged["label_pred"]

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    agreement = (y_true == y_pred).mean()

    print("\n=== Label evaluation (selected vs labeled_features) ===")
    print(f"Samples compared: {len(merged)}")
    print(f"Label agreement:  {agreement:.4f}")
    print(f"Precision (ASD=1): {precision:.4f}")
    print(f"Recall    (ASD=1): {recall:.4f}")
    print(f"F1-score  (ASD=1): {f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
