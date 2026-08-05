#!/usr/bin/env python
"""Rename generated outputs from Subj_D2_<id> to Subj_D2<id> to fix subject majority voting."""

import os
from pathlib import Path

PROJECT_ROOT = Path("c:/asd_project").resolve()
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "dataset2"

def rename_files_in_dir(directory):
    if not directory.is_dir():
        return
    for item in directory.iterdir():
        if item.is_file():
            name = item.name
            if "Subj_D2_" in name:
                new_name = name.replace("Subj_D2_", "Subj_D2")
                new_path = item.with_name(new_name)
                item.rename(new_path)
                print(f"Renamed: {item.name} -> {new_name}")

def main():
    print("Renaming dataset2 output files...")
    rename_files_in_dir(OUTPUTS_ROOT / "detections")
    rename_files_in_dir(OUTPUTS_ROOT / "tracked")
    rename_files_in_dir(OUTPUTS_ROOT / "child_sequences")
    print("Renaming completed successfully.")

if __name__ == "__main__":
    main()
