import json
import argparse
import pandas as pd
from pathlib import Path


""" 
Example usage:
python generate_predictions.py --csv /path/to/mapping.csv --output /path/to/predictions.json
"""

INPUT_SLUGS_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": [
        "prostate-tissue-biopsy-whole-slide-image"
    ],
    "Task02_classifying_lung_nodule_malignancy_in_ct": [
        "chest-ct-region-of-interest-cropout"
    ],
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": [
        "prostatectomy-tissue-whole-slide-image"
    ],
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": [
        "ihc-staining-for-pd-l1"
    ],
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": [
        "transverse-t2-prostate-mri"
    ],
    "Task07_detecting_lung_nodules_in_thoracic_ct": [
        "chest-ct"
    ],
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task09_segmenting_rois_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task10_segmenting_lesions_within_vois_in_ct": [
        "stacked-3d-ct-volumes-of-lesions"
    ],
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": [
        "sagittal-spine-mri"
    ],
}

MODEL_OUTPUT_SLUG_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "image-neural-representation",
    "Task02_classifying_lung_nodule_malignancy_in_ct": "image-neural-representation",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "image-neural-representation",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "image-neural-representation",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "patch-neural-representation",
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": "patch-neural-representation",
    "Task07_detecting_lung_nodules_in_thoracic_ct": "patch-neural-representation",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "patch-neural-representation",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "patch-neural-representation",
    "Task10_segmenting_lesions_within_vois_in_ct": "patch-neural-representation",
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": "patch-neural-representation",
}

def generate_predictions_from_csv(csv_path: Path, output_path: Path):
    df = pd.read_csv(csv_path)
    use_custom_pk = "pk" in df.columns
    use_custom_name = "case_id" in df.columns

    if not use_custom_name:
        raise ValueError("CSV must contain a 'case_id' column.")

    grouped = df.groupby("task_name")
    predictions = []

    for task_name, group in grouped:
        if task_name not in INPUT_SLUGS_DICT:
            print(f"Warning: unknown task '{task_name}' — skipping.")
            continue

        input_slugs = INPUT_SLUGS_DICT[task_name]
        output_slug = MODEL_OUTPUT_SLUG_DICT[task_name]

        for _, row in group.iterrows():
                name = row["case_id"]
                pk = row["pk"] if use_custom_pk else name
                predictions.append({
                    "pk": Path(pk).stem,
                    "inputs": [{
                        "image": {"name": name},
                        "interface": {"slug": input_slugs[0]}
                    }],
                    "outputs": [{
                        "interface": {
                            "slug": output_slug,
                            "relative_path": f"{output_slug}.json"
                        }
                    }]
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions.json from a CSV file.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Path to save predictions.json")

    args = parser.parse_args()
    generate_predictions_from_csv(args.csv, args.output)