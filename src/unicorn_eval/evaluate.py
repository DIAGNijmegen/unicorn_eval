#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat

import numpy as np
import openslide
import pandas as pd

from unicorn_eval.helpers import get_max_workers
from unicorn_eval.utils import (
    adapt_features,
    evaluate_predictions,
    extract_data,
    extract_embeddings_and_labels,
    normalize_metric,
    write_json_file,
)

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUNDTRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

ADAPTOR_SLUGS_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "adaptor-pathology-classification",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "adaptor-pathology-regression",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "adaptor-pathology-classification",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "adaptor-pathology-detection",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "adaptor-pathology-detection",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "adaptor-pathology-segmentation",
}

INPUT_SLUGS_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": [
        "prostate-tissue-biopsy-whole-slide-image"
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
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task09_segmenting_rois_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task20_generating_caption_from_wsi": [
        "he-staining"
    ]
}

MODEL_OUTPUT_SLUG_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "image-neural-representation",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "image-neural-representation",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "image-neural-representation",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "patch-neural-representation",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "patch-neural-representation",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "patch-neural-representation",
    "Task20_generating_caption_from_wsi": "nlp-predictions-dataset",
}

LABEL_SLUG_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "isup-grade.json",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "overall-survival-years.json",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "pd-l1-tps-binned.json",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "cell-classification.json",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "mitotic-figures.json",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "images/tumor-stroma-and-other/{case_id}.tif",
    "Task20_generating_caption_from_wsi": "nlp-predictions-dataset.json",
}

EXTRA_LABEL_SLUG_DICT = {
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": [
        "event.json"
    ],
}


def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    mapping = pd.read_csv(GROUNDTRUTH_DIRECTORY / "mapping.csv")

    image_name = None
    for slug_inputs in INPUT_SLUGS_DICT.values():
        for slug_input in slug_inputs:
            try:
                image_name = get_image_name(
                    values=job["inputs"],
                    slug=slug_input,
                )
            except Exception as e:
                continue

    assert image_name is not None, "No image found in predictions.json"
    case_name = Path(image_name).stem

    case_info = mapping[mapping.case_id == case_name]
    task_name = case_info.task_name.values[0]
    modality = case_info.modality.values[0]

    if modality == "vision":

        prediction = None

        slug_embedding = MODEL_OUTPUT_SLUG_DICT[task_name]

        # find the location of the results
        location_neural_representation = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug=slug_embedding,
        )

        # read the results

        result_algorithm = load_json_file(
            location=location_neural_representation,
        )[0]

        if slug_embedding == "image-neural-representation":
            embeddings = np.array(result_algorithm["features"]).astype(np.float32)
            coordinates, spacing, patch_size, image_size, image_spacing = None, None, None, None, None
        elif slug_embedding == "patch-neural-representation":
            embeddings, coordinates, spacing, patch_size, image_size, image_spacing = extract_data(result_algorithm)

    elif modality == "vision-language":

        model_output_slug = MODEL_OUTPUT_SLUG_DICT[task_name]

        embeddings = None
        coordinates = None
        spacing, patch_size, image_size, image_spacing = None, None, None, None

        # find the location of the results
        location_prediction = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug=model_output_slug,
        )

        # read the results
        prediction = load_json_file(
            location=location_prediction,
        )[0]

    case_specific_ground_truth_dir = GROUNDTRUTH_DIRECTORY / task_name / case_name

    slug_label = LABEL_SLUG_DICT[task_name]
    label_path = case_specific_ground_truth_dir / slug_label

    if label_path.suffix == ".json":
        label = load_json_file(location=label_path)
    elif label_path.suffix == ".tif":
        label_path = Path(str(label_path).replace("{case_id}", case_name))
        label = load_tif_file(location=label_path)
    else:
        raise ValueError(f"Unsupported file format: {label_path.suffix}")

    extra_labels = None
    extra_slug_labels = EXTRA_LABEL_SLUG_DICT.get(task_name, [])
    use_extra_labels = len(extra_slug_labels) > 0
    if use_extra_labels:
        extra_labels = {}
        for extra_slug_label in extra_slug_labels:
            slug_name = Path(extra_slug_label).stem
            extra_label_path = case_specific_ground_truth_dir / extra_slug_label
            extra_labels[slug_name] = load_json_file(location=extra_label_path)

        # convert extra_labels dictionary to a structured numpy array
        dtype = [(key, type(value)) for key, value in extra_labels.items()]
        extra_labels = np.array([tuple(extra_labels.values())], dtype=dtype)

    case_info_dict = case_info.to_dict(orient="records")[0]
    case_info_dict["embeddings"] = embeddings
    case_info_dict["coordinates"] = coordinates
    case_info_dict["spacing"] = spacing
    case_info_dict["image_spacing"] = image_spacing
    case_info_dict["image_size"] = image_size
    case_info_dict["patch_size"] = patch_size
    case_info_dict["prediction"] = prediction
    case_info_dict["label"] = label
    case_info_dict["extra_labels"] = extra_labels

    return case_info_dict


def get_cases_extra_labels_detection(cases_image_sizes, cases_image_spacings):
    case_extra_labels = {}
    for case_id, image_size in cases_image_sizes.items():
        spacing = cases_image_spacings[case_id]

        width, height = image_size
        spacing_x_um, spacing_y_um = spacing

        spacing_x_mm = spacing_x_um / 1000.0
        spacing_y_mm = spacing_y_um / 1000.0
        pixel_area_mm2 = spacing_x_mm * spacing_y_mm
        image_area_mm2 = width * height * pixel_area_mm2

        case_extra_labels[case_id] = image_area_mm2

    return case_extra_labels


def print_directory_contents(path: Path | str):
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir():
            print_directory_contents(child)
        else:
            print(child)


def read_adaptors():
    # read the adaptors that are used for this submission
    adaptors = {}
    for task_name, slug in ADAPTOR_SLUGS_DICT.items():
        adaptor_path = INPUT_DIRECTORY / f"{slug}.json"
        if adaptor_path.exists():
            with open(adaptor_path) as f:
                adaptors[task_name] = json.loads(f.read())
    return adaptors


def read_predictions():
    # the prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # this tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # reads a json file
    with open(location) as f:
        return json.loads(f.read())

def load_tif_file(*, location):
    slide = openslide.OpenSlide(location)
    print("Image dimensions:", slide.dimensions)
    level_0 = slide.read_region((0, 0), 0, slide.dimensions)
 #   save_tif(level_0, location.stem)
    level_0_np = np.array(level_0)
    class_labels = level_0_np[:, :, 0]  # shape: (H, W)
    return class_labels

def write_metrics(*, metrics):
    # write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


def write_combined_metrics(*, metric_dict: dict[dict]) -> None:
    metrics = {"metrics": {}, "normalized_metrics": {}}
    predictions = {"predictions": []}

    for task_name, task_metrics in metric_dict.items():
        case_prediction = [
            p.tolist() if isinstance(p, np.ndarray) else p
            for p in task_metrics["predictions"]
        ]
        predictions["predictions"].extend(case_prediction)

        for metric_name, metric_value in task_metrics["metrics"].items():
            task_identifier = task_name.split("_")[0]
            metrics["metrics"][f"{task_identifier}_{metric_name}"] = metric_value
            metrics["normalized_metrics"][f"{task_identifier}_{metric_name}"] = normalize_metric(task_name, metric_value)

        for metric_name, metric_value in task_metrics["additional_metrics"].items():
            task_identifier = task_name.split("_")[0]
            metrics["metrics"][f"{task_identifier}_{metric_name}"] = metric_value

    # aggregate metrics when there are multiple tasks
    metrics["metrics"]["mean"] = np.mean([metric_value for _, metric_value in metrics["normalized_metrics"].items()])

    write_json_file(
        location=OUTPUT_DIRECTORY / "metrics.json",
        content=metrics,
    )

    write_json_file(
        location=OUTPUT_DIRECTORY / "predictions.json",
        content=predictions,
    )


def main():
    print("input folder contents:")
    print_directory_contents(INPUT_DIRECTORY)
    print("=+=" * 10)
    print("grountruth folder contents:")
    print_directory_contents(GROUNDTRUTH_DIRECTORY)
    print("=+=" * 10)

    metrics = {}
    adaptors = read_adaptors()
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # use concurrent workers to process the predictions more efficiently
    max_workers = get_max_workers()
    with Pool(processes=max_workers) as pool:
        processed_results = pool.map(process, predictions)
    task_values = extract_embeddings_and_labels(processed_results)

    task_metrics = {}

    for task_name, results in task_values.items():

        modality = results["modality"]
        case_labels = results["case_labels"]
        case_ids = results["case_ids"]
        task_type = results["task_type"]

        if modality == "vision":

            adaptor_name = adaptors[task_name]

            shot_embeddings = results["shot_embeddings"]
            shot_labels = results["shot_labels"]
            shot_extra_labels = results["shot_extra_labels"]
            shot_ids = results["shot_ids"]
            case_embeddings = results["case_embeddings"]

            case_extra_labels = results["case_extra_labels"]
            patch_size = results["patch_size"]
            case_image_size = results["cases_image_sizes"]

            if task_type in ["classification", "regression"]:

                if len(shot_embeddings.shape) > 2:
                    shot_embeddings = shot_embeddings.squeeze(1)

                if len(case_embeddings.shape) > 2:
                    case_embeddings = case_embeddings.squeeze(1)

            elif task_type == "detection":

                case_extra_labels = get_cases_extra_labels_detection(results["cases_image_sizes"], results["cases_image_spacings"])

            predictions = adapt_features(
                adaptor_name=adaptor_name,
                task_type=task_type,
                shot_features=shot_embeddings,
                shot_names=shot_ids,
                shot_labels=shot_labels,
                test_features=case_embeddings,
                shot_coordinates=results["shot_coordinates"],
                test_coordinates=results["cases_coordinates"],
                test_names=case_ids,
                patch_size=patch_size,
                test_image_sizes=case_image_size,
                shot_extra_labels=shot_extra_labels,
            )

        elif modality == "vision-language":
            predictions = [pred["text"] for pred in results["prediction"]]
            case_labels = [label["text"] for case in results["case_labels"] for label in case]
            case_extra_labels = None

        metrics = evaluate_predictions(
            task_name=task_name,
            case_ids=case_ids,
            test_predictions=predictions,
            test_labels=case_labels,
            test_extra_labels=case_extra_labels,
        )
        task_metrics[task_name] = metrics

    write_combined_metrics(metric_dict=task_metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())