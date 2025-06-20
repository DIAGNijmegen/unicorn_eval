import os
import json
import random
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

"""
Example usage:
python generate_mapping.py --root_dir /path/to/local_data --shots 12 --output mapping.csv

Root directory structure:
/path/to/local_data/
├── Task01_classifying_he_prostate_biopsies_into_isup_scores
│   ├── shots-public
│   │   ├── case_01
│   │   │   ├── images
│   │   │   └── unicorn-task-description.json
│   │   │   └── inputs.json
│   ├── shots-public-labels
"""


def generate_mapping(root_dir, n_shots, output_path="mapping.csv"):
    rows = []

    root = Path(root_dir)

    # Traverse each task directory
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue

        task_dir = task_dir / "shots-public"
        case_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
        task_rows = []

        for case_dir in case_dirs:
            task_json_path = case_dir / "unicorn-task-description.json"
            if not task_json_path.exists():
                print(f"Warning: unicorn-task-description.json not found in {case_dir}")
                continue

            with open(task_json_path, "r") as f:
                task_info = json.load(f)

            row = {
                "case_id": case_dir.name,
                "task_name": task_info["task_name"],
                "task_type": task_info["task_type"],
                "domain": task_info["domain"],
                "modality": task_info["modality"],
            }
            task_rows.append(row)

        # Randomly split shots and cases
        random.shuffle(task_rows)
        for i, row in enumerate(task_rows):
            row["split"] = "shot" if i < n_shots else "case"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Mapping saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mapping.csv from unicorn folder structure"
    )
    parser.add_argument(
        "--root_dir", help="Path to the root folder containing task folders"
    )
    parser.add_argument(
        "--shots", type=int, default=12, help="Number of shot cases per task"
    )
    parser.add_argument("--output", default="mapping.csv", help="Output path for CSV")

    args = parser.parse_args()
    generate_mapping(args.root_dir, args.shots, args.output)
