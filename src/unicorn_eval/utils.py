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

import gc
import json
import logging
from functools import partial
from typing import Any
from pathlib import Path
from pprint import pformat

import numpy as np
import openslide
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sksurv.metrics import concordance_index_censored

from unicorn_eval.adaptors import (KNN, ConvDetector, DensityMap, KNNRegressor,
                                   LinearProbing, LinearProbingRegressor,
                                   LogisticRegression, MultiLayerPerceptron,
                                   MultiLayerPerceptronRegressor,
                                   PatchNoduleRegressor, WeightedKNN,
                                   WeightedKNNRegressor)
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v1 import \
    LinearUpsampleConv3D_V1
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2 import (
    ConvUpsampleSegAdaptor, LinearUpsampleConv3D_V2)
from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v1 import (
    SegmentationUpsampling, SegmentationUpsampling3D)
from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v2.main import \
    SegmentationUpsampling3D_V2
from unicorn_eval.adaptors.segmentation.mevis_conv_segmentation_3d.v1.main import \
    ConvSegmentation3D
from unicorn_eval.metrics.dice import compute_dice_score
from unicorn_eval.metrics.f1_score import compute_f1
from unicorn_eval.metrics.picai_score import compute_picai_score
from unicorn_eval.metrics.sensitivity import compute_cpm
from unicorn_eval.metrics.spider import compute_spider_score
from unicorn_eval.metrics.uls import compute_uls_score
from unicorn_eval.metrics.vision_language import \
    compute_average_language_metric


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUNDTRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

METRIC_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": {
        "name": "cohen-kappa-quadratic",
        "fn": partial(cohen_kappa_score, weights="quadratic"),
        "range": (0, 1),
    },
    "Task02_classifying_lung_nodule_malignancy_in_ct": {
        "name": "auc",
        "fn": roc_auc_score,
        "range": (0.5, 1),
    },
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": {
        "name": "c-index",
        "fn": concordance_index_censored,
        "range": (0.5, 1),
    },
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": {
        "name": "cohen-kappa-quadratic",
        "fn": partial(cohen_kappa_score, weights="quadratic"),
        "range": (0, 1),
    },
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": {
        "name": "f1",
        "fn": compute_f1,
        "range": (0, 1),
    },
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": {
        "name": "picai",
        "fn": compute_picai_score,
        "range": (0.25, 1),
    },
    "Task07_detecting_lung_nodules_in_thoracic_ct": {
        "name": "sensitivity",
        "fn": compute_cpm,
        "range": (0, 1),
    },
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": {
        "name": "f1",
        "fn": compute_f1,
        "range": (0, 1),
    },
    "Task09_segmenting_rois_in_breast_cancer_wsis": {
        "name": "dice",
        "fn": compute_dice_score,
        "range": (0.2548, 1),
    },
    "Task10_segmenting_lesions_within_vois_in_ct": {
        "name": "uls_score",
        "fn": compute_uls_score,
        "range": (0, 1),
    },
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": {
        "name": "spider_score",
        "fn": compute_spider_score,
        "range": (0, 1),
    },
    "Task12_predicting_histopathology_sample_origin": {
        "name": "unweighted-kappa",
        "range": (0, 1),
    },
    "Task13_classifying_pulmonary_nodule_presence": {
        "name": "auc",
        "range": (0.5, 1),
    },
    "Task14_classifying_kidney_abnormality": {
        "name": "auc",
        "range": (0.5, 1),
    },
    "Task15_hip_kellgren_lawrence_score": {
        "name": "unweighted-kappa",
        "range": (0, 1),
    },
    "Task16_classifying_colon_histopathology_diagnosis": {
        "name": "macro-auc",
        "range": (0.5, 1),
    },
    "Task17_predicting_lesion_size_measurements": {
        "name": "rsmape",
        "range": (0.7580, 1),
    },
    "Task18_predicting_prostate_volume_psa_and_psa_density": {
        "name": "rsmape",
        "range": (0.7668, 1),
    },
    "Task19_anonymizing_report": {
        "name": "redaction_score",
        "range": (0, 1),
    },
    "Task20_generating_caption_from_wsi": {
        "name": "average_language_metric",
        "fn": compute_average_language_metric,
        "range": (0, 1),
    },
}

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
        "transverse-t2-prostate-mri",
    ],
    "Task07_detecting_lung_nodules_in_thoracic_ct": [
        "chest-ct",
    ],
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task09_segmenting_rois_in_breast_cancer_wsis": [
        "histopathology-region-of-interest-cropout"
    ],
    "Task10_segmenting_lesions_within_vois_in_ct": ["stacked-3d-ct-volumes-of-lesions"],
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": [
        "sagittal-spine-mri"
    ],
    "Task20_generating_caption_from_wsi": ["he-staining"],
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
    "Task20_generating_caption_from_wsi": "nlp-predictions-dataset",
}

LABEL_SLUG_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "isup-grade.json",
    "Task02_classifying_lung_nodule_malignancy_in_ct": "lung-nodule-malignancy-risk.json",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "overall-survival-years.json",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "pd-l1-tps-binned.json",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "cell-classification.json",
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": "images/transverse-cspca-label/{case_id}.mha",
    "Task07_detecting_lung_nodules_in_thoracic_ct": "nodule-locations.json",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "mitotic-figures.json",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "images/tumor-stroma-and-other/{case_id}.tif",
    "Task10_segmenting_lesions_within_vois_in_ct": "images/ct-binary-uls/{case_id}.mha",
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": "images/sagittal-spine-mr-segmentation/{case_id}.mha",
    "Task20_generating_caption_from_wsi": "nlp-predictions-dataset.json",
}

EXTRA_LABEL_SLUG_DICT = {
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": [
        "event.json",
        "cohort.json",
    ],
    "Task07_detecting_lung_nodules_in_thoracic_ct": ["diameter.json"],
}


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
    logging.info(f"Image dimensions: {slide.dimensions}")
    level_0 = slide.read_region((0, 0), 0, slide.dimensions)
    #   save_tif(level_0, location.stem)
    level_0_np = np.array(level_0)
    class_labels = level_0_np[:, :, 0]  # shape: (H, W)
    return class_labels


def load_mha_file(*, path: Path | str):
    class_labels = sitk.ReadImage(str(path))

    if class_labels is None:
        raise ValueError("Failed to load class labels from MHA file.")

    return (
        sitk.GetArrayFromImage(class_labels),
        list(class_labels.GetSize()),
        list(class_labels.GetOrigin()),
        list(class_labels.GetSpacing()),
        list(class_labels.GetDirection()),
    )


def get_image_name(*, values, slug):
    # this tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def read_inputs(input_dir: Path, case_names: list[str]):
    # the prediction file tells us the location of the users' predictions
    with open(input_dir / "predictions.json") as f:
        inputs = json.loads(f.read())

    filtered_inputs = []
    for elem in inputs:
        case_name = None
        for slug_inputs in INPUT_SLUGS_DICT.values():
            for slug_input in slug_inputs:
                try:
                    image_name = get_image_name(
                        values=elem["inputs"], slug=slug_input
                    )
                    case_name = Path(image_name).stem
                    # remove suffixes "_adc", "_t2w", "_hbv" and "_tissue" from the case name if present
                    for suffix in ["_adc", "_t2w", "_hbv", "_tissue"]:
                        if case_name.endswith(suffix):
                            case_name = case_name[: -len(suffix)]
                    break
                except Exception:
                    continue
            if case_name:
                break

        if case_name is None:
            # skip if we can't determine case name (language task)
            continue
        if case_name in case_names:
            filtered_inputs.append(elem)

    # free up memory
    del inputs
    gc.collect()

    return filtered_inputs


def process(job):
    """Processes a single algorithm job, looking at the outputs"""

    embeddings = None
    prediction = None
    coordinates = None
    spacing = None
    patch_size = None
    patch_spacing = None
    image_size = None
    image_spacing = None
    image_origin = None
    image_direction = None
    feature_grid_resolution = None

    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    mapping_path = GROUNDTRUTH_DIRECTORY / "mapping.csv"
    try:
        mapping = pd.read_csv(
            mapping_path, dtype={"case_id": str}
        )  # ensure case_id is string to enable leading zeros
    except FileNotFoundError:
        # if the mapping file is not found, we assume that the evaluation is for a language task
        # and we do not need the mapping
        logging.error(f"{mapping_path} not found, cannot group by task.")
        return {}

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

    if image_name is None:
        # if no image_name is found, it corresponds to a pure language task
        # for which we already have written the metrics
        return None

    case_name = Path(image_name).stem

    # remove suffixes "_adc", "_t2w", and "_hbv" from the case name if present
    for suffix in ["_adc", "_t2w", "_hbv"]:
        if case_name.endswith(suffix):
            case_name = case_name[: -len(suffix)]

    case_info = mapping[mapping.case_id == case_name]
    if case_info.empty:
        raise ValueError(f"Case {case_name} not found in mapping.")

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
        neural_representations = load_json_file(
            location=location_neural_representation,
        )

        features = []
        if slug_embedding == "image-neural-representation":
            for neural_representation in neural_representations:
                feature = neural_representation["features"]
                feature = np.array(feature).astype(np.float32)
                features.append(feature)
            embeddings = np.concatenate(features)

        elif slug_embedding == "patch-neural-representation":
            # TODO: better handle the case when there are multiple encoded inputs for a case
            # right now we concatenate the features
            # and use the first coordinates, spacing, patch_size, image_size, and image_spacing
            first = True
            for neural_representation in neural_representations:
                (
                    feature,
                    curr_coordinates,
                    curr_spacing,
                    curr_patch_size,
                    curr_patch_spacing,
                    curr_feature_grid_resolution,
                    curr_image_size,
                    curr_image_spacing,
                    curr_image_origin,
                    curr_image_direction,
                ) = extract_data(neural_representation)
                features.append(feature)
                if first:
                    coordinates = curr_coordinates
                    spacing = curr_spacing
                    patch_size = curr_patch_size
                    patch_spacing = curr_patch_spacing
                    feature_grid_resolution = curr_feature_grid_resolution
                    image_size = curr_image_size
                    image_spacing = curr_image_spacing
                    image_origin = curr_image_origin
                    image_direction = curr_image_direction
                    first = False
                else:
                    assert np.all(
                        coordinates == curr_coordinates
                    ), "Coordinates do not match between images of the same case"
                    assert np.all(
                        spacing == curr_spacing
                    ), "Spacing does not match between images of the same case"
                    assert np.all(
                        patch_size == curr_patch_size
                    ), "Patch size does not match between images of the same case"
                    assert np.all(
                        patch_spacing == curr_patch_spacing
                    ), "Patch spacing does not match between images of the same case"
                    assert np.all(
                        image_size == curr_image_size
                    ), "Image size does not match between images of the same case"
                    assert np.all(
                        image_spacing == curr_image_spacing
                    ), "Image spacing does not match between images of the same case"
                    assert np.all(
                        image_origin == curr_image_origin
                    ), "Image origin does not match between images of the same case"
                    assert np.all(
                        image_direction == curr_image_direction
                    ), "Image direction does not match between images of the same case"
            embeddings = np.concatenate(features)

    elif modality == "vision-language":

        model_output_slug = MODEL_OUTPUT_SLUG_DICT[task_name]

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

    label_size, label_origin, label_spacing, label_direction = None, None, None, None
    if label_path.suffix == ".json":
        label = load_json_file(location=label_path)
    elif label_path.suffix == ".tif":
        label_path = Path(str(label_path).replace("{case_id}", case_name))
        label = load_tif_file(location=label_path)
    elif label_path.suffix == ".mha":
        label_path = Path(str(label_path).replace("{case_id}", case_name))
        label, label_size, label_origin, label_spacing, label_direction = load_mha_file(
            path=label_path,
        )
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
            if extra_label_path.exists():
                extra_labels[slug_name] = load_json_file(location=extra_label_path)
            else:
                logging.warning(f"extra label file not found: {extra_label_path}")
                extra_labels[slug_name] = None

        # convert extra_labels dictionary to a structured numpy array
        dtype = [(key, type(value)) for key, value in extra_labels.items()]
        extra_labels = np.array([tuple(extra_labels.values())], dtype=dtype)

    case_info_dict = case_info.to_dict(orient="records")[0]
    case_info_dict["embeddings"] = embeddings
    case_info_dict["coordinates"] = coordinates
    case_info_dict["spacing"] = spacing
    case_info_dict["image_spacing"] = image_spacing
    case_info_dict["image_size"] = image_size
    case_info_dict["image_origin"] = image_origin
    case_info_dict["image_direction"] = image_direction
    case_info_dict["patch_size"] = patch_size
    case_info_dict["patch_spacing"] = patch_spacing
    case_info_dict["feature_grid_resolution"] = feature_grid_resolution
    case_info_dict["prediction"] = prediction
    case_info_dict["label"] = label
    case_info_dict["extra_labels"] = extra_labels
    case_info_dict["label_spacing"] = label_spacing
    case_info_dict["label_size"] = label_size
    case_info_dict["label_origin"] = label_origin
    case_info_dict["label_direction"] = label_direction

    return case_info_dict


def adapt_features(
    *,
    adaptor_name: str,
    task_type: str,
    shot_features: np.ndarray,
    shot_labels: np.ndarray,
    test_features: np.ndarray,
    shot_coordinates: list[np.ndarray] | None = None,
    test_coordinates: list[np.ndarray] | None = None,
    shot_names: list[str] | None = None,
    test_names: list[str] | None = None,
    global_patch_size: list[int] | int | None = 224,
    global_patch_spacing: list[float] | float | None = None,
    shot_patch_sizes: dict[str, list[int] | int] | None = None,
    test_patch_sizes: dict[str, list[int] | int] | None = None,
    shot_patch_spacings: dict[str, list[float] | float] | None = None,
    test_patch_spacings: dict[str, list[float] | float] | None = None,
    feature_grid_resolution: list[int] | None = None,
    test_image_sizes: dict[str, list[int]] | None = None,
    test_image_spacing: dict[str, list[float]] | None = None,
    test_image_origins: dict[str, list[float]] | None = None,
    test_image_directions: dict[str, list[float]] | None = None,
    test_label_sizes: dict[str, list[int]] | None = None,
    test_label_spacing: dict[str, list[float]] | None = None,
    test_label_origins: dict[str, list[float]] | None = None,
    test_label_directions: dict[str, list[float]] | None = None,
    shot_image_sizes: dict[str, list[int]] | None = None,
    shot_image_spacing: dict[str, list[float]] | None = None,
    shot_image_origins: dict[str, list[float]] | None = None,
    shot_image_directions: dict[str, list[float]] | None = None,
    shot_label_spacing: dict[str, list[float]] | None = None,
    shot_label_origins: dict[str, list[float]] | None = None,
    shot_label_directions: dict[str, list[float]] | None = None,
    shot_extra_labels: np.ndarray | None = None,
    return_probabilities: bool = False,
) -> np.ndarray:
    num_shots = len(shot_features)

    if "-nn" in adaptor_name:
        k = int(adaptor_name.split("-")[0])
        k = min(k, num_shots)
        if "weighted" in adaptor_name:
            if task_type == "classification":
                adaptor = WeightedKNN(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k,
                    return_probabilities=return_probabilities,
                )
            elif task_type == "regression":
                adaptor = WeightedKNNRegressor(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k,
                )
        else:
            if task_type == "classification":
                adaptor = KNN(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k,
                    return_probabilities=return_probabilities,
                )
            elif task_type == "regression":
                adaptor = KNNRegressor(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k,
                )

    elif adaptor_name == "logistic-regression":
        assert task_type == "classification"
        adaptor = LogisticRegression(
            shot_features=shot_features,
            shot_labels=shot_labels,
            test_features=test_features,
            max_iterations=1000,
            C=1.0,
            solver="lbfgs",
            return_probabilities=return_probabilities,
        )

    elif "linear-probing" in adaptor_name:
        if task_type == "classification":
            adaptor = LinearProbing(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                num_epochs=100,
                learning_rate=0.001,
                return_probabilities=return_probabilities,
            )
        elif task_type == "regression":
            survival = False
            if "survival" in adaptor_name:
                survival = True
            adaptor = LinearProbingRegressor(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                survival=survival,
                num_epochs=100,
                learning_rate=0.001,
            )

    elif "linear-classification" in adaptor_name:
        assert task_type == "classification", "Linear classification is only supported for classification tasks."
        adaptor = LinearProbing(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_extra_labels=shot_extra_labels,
            test_features=test_features,
            num_epochs=100,
            learning_rate=0.001,
            return_probabilities=return_probabilities,
        )

    elif "mlp" in adaptor_name:
        if task_type == "classification":
            adaptor = MultiLayerPerceptron(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
                return_probabilities=return_probabilities,
            )
        elif task_type == "regression":
            survival = False
            if "survival" in adaptor_name:
                survival = True
            adaptor = MultiLayerPerceptronRegressor(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                survival=survival,
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
            )

    elif adaptor_name == "patch-nodule-regressor":
        adaptor = PatchNoduleRegressor(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_ids=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_ids=test_names,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            shot_image_spacings=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            hidden_dim=64,
            num_epochs=50,
            lr=0.001,
        )

    elif adaptor_name == "density-map":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = DensityMap(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            global_patch_size=global_patch_size[0],
            heatmap_size=16,
        )

    elif adaptor_name == "conv-detector":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = ConvDetector(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            patch_sizes=global_patch_size,
        )

    elif adaptor_name == "segmentation-upsampling":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = SegmentationUpsampling(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            global_patch_size=global_patch_size[0],
            global_patch_spacing=global_patch_spacing[0],
        )
    elif adaptor_name == "linear-upsample-conv3d":
        adaptor = LinearUpsampleConv3D_V1(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
        )
    elif adaptor_name == "linear-upsample-conv3d-v2":
        adaptor = LinearUpsampleConv3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
        )
    elif adaptor_name == "conv3d-linear-upsample":
        adaptor = LinearUpsampleConv3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            decoder_cls=ConvUpsampleSegAdaptor,
        )

    elif adaptor_name == "segmentation-upsampling-3d":
        adaptor = SegmentationUpsampling3D(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
        )

    elif adaptor_name == "segmentation-upsampling-3d-v2":
        adaptor = SegmentationUpsampling3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
        )

    elif adaptor_name == "conv-segmentation-3d":
        adaptor = ConvSegmentation3D(  # All args copied from segmentation-upsampling-3d
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,  # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            feature_grid_resolution=feature_grid_resolution,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
        )

    elif adaptor_name == "detection-by-linear-upsample-conv3d":
        adaptor = LinearUpsampleConv3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            return_binary=False,
        )

    elif adaptor_name == "detection-by-conv3d-linear-upsample":
        adaptor = LinearUpsampleConv3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            return_binary=False,
            decoder_cls=ConvUpsampleSegAdaptor,
        )

    elif adaptor_name == "detection-by-segmentation-upsampling-3d":
        adaptor = SegmentationUpsampling3D(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            return_binary=False,
        )

    elif adaptor_name == "detection-by-segmentation-upsampling-3d-v2":
        adaptor = SegmentationUpsampling3D_V2(
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            return_binary=False,
        )

    elif adaptor_name == "conv-detection-segmentation-3d":
        adaptor = ConvSegmentation3D(  # All args copied from detection-by-segmentation-upsampling-3d
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
            shot_labels=shot_labels,
            shot_label_spacing=shot_label_spacing,
            shot_label_origins=shot_label_origins,
            shot_label_directions=shot_label_directions,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            feature_grid_resolution=feature_grid_resolution,
            shot_patch_sizes=shot_patch_sizes,
            test_patch_sizes=test_patch_sizes,
            shot_patch_spacings=shot_patch_spacings,
            test_patch_spacings=test_patch_spacings,
            return_binary=False,
        )

    else:
        raise ValueError(f"Unknown adaptor: {adaptor_name}")

    adaptor.fit()
    predictions = adaptor.predict()
    return predictions

#########

def get_adaptor(
    *,
    adaptor_name: str,
    task_type: str,
    num_shots: int,
    global_patch_size: list[int] | int | None = 224,
    global_patch_spacing: list[float] | float | None = None,
    feature_grid_resolution: list[int] | None = None,
    return_probabilities: bool = False,
) -> np.ndarray:

    if "-nn" in adaptor_name:
        k = int(adaptor_name.split("-")[0])
        k = min(k, num_shots)
        if "weighted" in adaptor_name:
            if task_type == "classification":
                adaptor = WeightedKNN(
                    k=k,
                    return_probabilities=return_probabilities,
                )
            elif task_type == "regression":
                adaptor = WeightedKNNRegressor(k=k)
        else:
            if task_type == "classification":
                adaptor = KNN(
                    k=k,
                    return_probabilities=return_probabilities,
                )
            elif task_type == "regression":
                adaptor = KNNRegressor(k=k)

    elif adaptor_name == "logistic-regression":
        assert task_type == "classification"
        adaptor = LogisticRegression(
            max_iterations=1000,
            C=1.0,
            solver="lbfgs",
            return_probabilities=return_probabilities,
        )

    elif "linear-probing" in adaptor_name:
        if task_type == "classification":
            adaptor = LinearProbing(
                num_epochs=100,
                learning_rate=0.001,
                return_probabilities=return_probabilities,
            )
        elif task_type == "regression":
            survival = False
            if "survival" in adaptor_name:
                survival = True
            adaptor = LinearProbingRegressor(
                survival=survival,
                num_epochs=100,
                learning_rate=0.001,
            )

    elif "linear-classification" in adaptor_name:
        assert task_type == "classification", "Linear classification is only supported for classification tasks."
        adaptor = LinearProbing(
            num_epochs=100,
            learning_rate=0.001,
            return_probabilities=return_probabilities,
        )

    elif "mlp" in adaptor_name:
        if task_type == "classification":
            adaptor = MultiLayerPerceptron(
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
                return_probabilities=return_probabilities,
            )
        elif task_type == "regression":
            survival = False
            if "survival" in adaptor_name:
                survival = True
            adaptor = MultiLayerPerceptronRegressor(
                survival=survival,
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
            )

    elif adaptor_name == "patch-nodule-regressor":
        adaptor = PatchNoduleRegressor(
            hidden_dim=64,
            num_epochs=50,
            lr=0.001,
        )

    elif adaptor_name == "density-map":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = DensityMap(
            global_patch_size=global_patch_size[0],
            heatmap_size=16,
        )

    elif adaptor_name == "conv-detector":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = ConvDetector(
            patch_sizes=global_patch_size,
        )

    elif adaptor_name == "segmentation-upsampling":
        assert global_patch_size is not None, f"Global patch size must be specified for {adaptor_name} adaptor."
        adaptor = SegmentationUpsampling(
            global_patch_size=global_patch_size[0],
            global_patch_spacing=global_patch_spacing[0],
        )
    elif adaptor_name == "linear-upsample-conv3d":
        adaptor = LinearUpsampleConv3D_V1(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
        )
    elif adaptor_name == "linear-upsample-conv3d-v2":
        adaptor = LinearUpsampleConv3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
        )
    elif adaptor_name == "conv3d-linear-upsample":
        adaptor = LinearUpsampleConv3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            decoder_cls=ConvUpsampleSegAdaptor,
        )

    elif adaptor_name == "segmentation-upsampling-3d":
        adaptor = SegmentationUpsampling3D(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
        )

    elif adaptor_name == "segmentation-upsampling-3d-v2":
        adaptor = SegmentationUpsampling3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
        )

    elif adaptor_name == "conv-segmentation-3d":
        adaptor = ConvSegmentation3D(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            feature_grid_resolution=feature_grid_resolution,
        )

    elif adaptor_name == "detection-by-linear-upsample-conv3d":
        adaptor = LinearUpsampleConv3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            return_binary=False,
        )

    elif adaptor_name == "detection-by-conv3d-linear-upsample":
        adaptor = LinearUpsampleConv3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=None,
            return_binary=False,
            decoder_cls=ConvUpsampleSegAdaptor,
        )

    elif adaptor_name == "detection-by-segmentation-upsampling-3d":
        adaptor = SegmentationUpsampling3D(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            return_binary=False,
        )

    elif adaptor_name == "detection-by-segmentation-upsampling-3d-v2":
        adaptor = SegmentationUpsampling3D_V2(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            return_binary=False,
        )

    elif adaptor_name == "conv-detection-segmentation-3d":
        adaptor = ConvSegmentation3D(
            global_patch_size=global_patch_size,
            global_patch_spacing=global_patch_spacing,
            feature_grid_resolution=feature_grid_resolution,
            return_binary=False,
        )

    else:
        raise ValueError(f"Unknown adaptor: {adaptor_name}")

    return adaptor


def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types."""
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


def evaluate_predictions(
    task_name, case_ids, test_predictions, test_labels, test_extra_labels=None, save_predictions: bool = False
):

    metrics = {
        "predictions": [],  # list to store individual case results
        "metrics": {},  # dictionary to store main metric
        "additional_metrics": {},  # dictionary to store additional metrics
    }

    if save_predictions:
        if task_name == "Task07_detecting_lung_nodules_in_thoracic_ct":
            # Only store references, not copies
            prediction_entry = {
                "case_id": convert_numpy_types(case_ids),
                "ground_truth": convert_numpy_types(test_labels),
                "prediction": convert_numpy_types(test_predictions),
            }
            metrics["predictions"].append(prediction_entry)
        else:
            # Use generator to avoid building a large list in memory
            for case_id, prediction, ground_truth in zip(case_ids, test_predictions, test_labels):
                ground_truth = convert_numpy_types(ground_truth)
                prediction = convert_numpy_types(prediction)
                metrics["predictions"].append(
                    {
                        "case_id": case_id,
                        "ground_truth": convert_numpy_types(ground_truth),
                        "prediction": convert_numpy_types(prediction),
                    }
                )

    # handle metric computation based on task_name
    metric_name = METRIC_DICT[task_name]["name"]
    metric_fn = METRIC_DICT[task_name]["fn"]
    metric_dict = {}
    additional_metric_dict = {}
    if task_name == "Task01_classifying_he_prostate_biopsies_into_isup_scores":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task02_classifying_lung_nodule_malignancy_in_ct":
        malignancy_risk = test_predictions[:, 1]
        metric_value = metric_fn(test_labels, malignancy_risk)
        metric_dict[metric_name] = metric_value
    elif (
        task_name
        == "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies"
    ):
        events = test_extra_labels["event"].astype(bool)
        cohorts = test_extra_labels["cohort"]
        if len(np.unique(list(set(cohorts)))) > 1:
            cohort_metrics = []
            for c in np.unique(cohorts):
                cohort_mask = cohorts == c
                cohort_metric = metric_fn(events[cohort_mask], test_labels[cohort_mask], -test_predictions[cohort_mask])[0]
                cohort_metrics.append(cohort_metric)
            metric_value = np.mean(cohort_metrics)
        else:
            metric_value = metric_fn(events, test_labels, -test_predictions)[0]
        metric_dict[metric_name] = metric_value
    elif (
        task_name
        == "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi"
    ):
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif (
        task_name
        == "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer"
    ):
        metric_value = metric_fn(test_labels, test_predictions, 20) # Data at 0.5um/px, 10um distance
        metric_dict[metric_name] = metric_value
    elif (
        task_name
        == "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams"
    ):
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task07_detecting_lung_nodules_in_thoracic_ct":
        metric_value = metric_fn(
            case_ids, test_predictions, test_labels, test_extra_labels
        )
        metric_dict[metric_name] = metric_value
    elif task_name == "Task08_detecting_mitotic_figures_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions, 30) # Data at 0.25um/px, 7.5um distance
        metric_dict[metric_name] = metric_value
    elif task_name == "Task09_segmenting_rois_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task10_segmenting_lesions_within_vois_in_ct":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif (
        task_name == "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri"
    ):
        metric_value = metric_fn(test_labels, test_predictions, case_ids)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task20_generating_caption_from_wsi":
        language_metric_dict = metric_fn(test_labels, test_predictions)  # a dictionary
        metric_dict[metric_name] = language_metric_dict.pop(metric_name)
        additional_metric_dict.update(language_metric_dict)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    metrics["metrics"] = metric_dict
    metrics["additional_metrics"] = additional_metric_dict

    return metrics


def process_image_representation(data):
    # stack embeddings
    data["shot_embeddings"] = np.vstack(data["shot_embeddings"])
    data["case_embeddings"] = np.vstack(data["case_embeddings"])
    # convert labels to numpy arrays
    data["shot_labels"] = np.array(data["shot_labels"])
    data["case_labels"] = np.array(data["case_labels"])
    if data["shot_extra_labels"] and data["shot_extra_labels"][0] is not None:
        data["shot_extra_labels"] = np.concatenate(data["shot_extra_labels"], axis=0)
    else:
        data["shot_extra_labels"] = None
    if data["case_extra_labels"] and data["case_extra_labels"][0] is not None:
        data["case_extra_labels"] = np.concatenate(data["case_extra_labels"], axis=0)
    else:
        data["case_extra_labels"] = None
    return data


def process_detection_pathology(
    data,
):

    def extract_points(labels):
        """
        Pull out coordinate tuples from a list of GT dictionaries.

        * Keeps the first two coordinates when available.

        Returns
        -------
        list[list[tuple]]
            Two‑level list: ``[case_idx][pt_idx] -> tuple``.
        """
        pts_all = []
        for gt in labels:
            case_pts = []
            for p in gt.get("points", []):
                pt = p.get("point")
                if pt is None:
                    continue
                case_pts.append(tuple(pt[:2]))
            pts_all.append(case_pts)
        return pts_all

    data["shot_labels"] = extract_points(data["shot_labels"])
    data["case_labels"] = extract_points(data["case_labels"])

    extra_list = data.get("case_extra_labels")
    if not extra_list or extra_list[0] is None:
        data["case_extra_labels"] = None
        return data

    data["case_extra_labels"] = np.concatenate(extra_list, axis=0)

    return data


def process_detection_radiology(data, task_name: str | None = None):

    def extract_points(labels):
        """
        Pull out coordinate tuples from a list of GT dictionaries.

        * Keeps the first three coordinates when available.
        * Falls back to the first two coordinates for 2‑D points.

        Returns
        -------
        list[list[tuple]]
            Two‑level list: ``[case_idx][pt_idx] -> tuple``.
        """
        pts_all = []
        for gt in labels:
            case_pts = []
            for p in gt.get("points", []):
                pt = p.get("point")
                if pt is None:
                    continue
                case_pts.append(tuple(pt[:3]) if len(pt) >= 3 else tuple(pt[:2]))
            pts_all.append(case_pts)
        return pts_all

    data["shot_labels"] = extract_points(data["shot_labels"])
    data["case_labels"] = extract_points(data["case_labels"])

    extra_list = data.get("case_extra_labels")
    if not extra_list or extra_list[0] is None:
        data["case_extra_labels"] = None
        return data

    if task_name == "Task07_detecting_lung_nodules_in_thoracic_ct":
        # build: [{'point': …, 'diameter': …, 'name': …}, …]
        diameter_records = []
        for case_id, case_extra in enumerate(extra_list):
            if isinstance(case_extra, dict):
                # expected structure: {<study_id>: {'points': […]}}
                nested = next(iter(case_extra.values()), {})
                for idx, p in enumerate(nested.get("points", [])):
                    diameter_records.append(
                        {
                            "point": tuple(p["point"][:3]),
                            "diameter": float(p["diameter"]),
                            "name": p.get("name", f"case{case_id}_pt{idx}"),
                        }
                    )
            elif isinstance(case_extra, (list, np.ndarray)):
                first_tuple = case_extra[0]
                if len(first_tuple) >= 1:
                    element = first_tuple[0]

                    if element is None:
                        logging.info("nothing to process in this case (got [(None,)])")

                    elif isinstance(element, dict):
                        for idx, d in enumerate(element.get("points")):
                            diameter_records.append(
                                {
                                    "point": None,
                                    "diameter": float(d.get("diameter")),
                                    "name": f"case{case_id}_pt{idx}",
                                }
                            )

            elif isinstance(case_extra, (int, float)):
                diameter_records.append(
                    {
                        "point": None,
                        "diameter": float(case_extra),
                        "name": f"case{case_id}",
                    }
                )

            else:
                raise ValueError(f"Unsupported extra_label type: {type(case_extra)}")

        data["case_extra_labels"] = diameter_records

    else:
        data["case_extra_labels"] = np.concatenate(extra_list, axis=0)

    return data


def extract_embeddings_and_labels(processed_results, task_name):
    """Extract embeddings and labels for a given task."""
    task_data = {
        "task_type": None,
        "modality": None,
        "domain": None,
        "global_patch_size": None,
        "global_patch_spacing": None,
        "feature_grid_resolution": None,
        "prediction": [],
        "shot_embeddings": [],
        "shot_coordinates": [],
        "shot_image_spacings": {},
        "shot_image_origins": {},
        "shot_image_directions": {},
        "shot_image_sizes": {},
        "shot_patch_sizes": {},
        "shot_patch_spacings": {},
        "shot_label_sizes": {},
        "shot_label_spacings": {},
        "shot_label_origins": {},
        "shot_label_directions": {},
        "shot_labels": [],
        "shot_extra_labels": [],
        "shot_ids": [],
        "case_embeddings": [],
        "cases_coordinates": [],
        "case_labels": [],
        "case_extra_labels": [],
        "case_ids": [],
        "cases_image_sizes": {},
        "cases_image_spacings": {},
        "cases_image_origins": {},
        "cases_image_directions": {},
        "cases_patch_sizes": {},
        "cases_patch_spacings": {},
        "cases_label_sizes": {},
        "cases_label_spacings": {},
        "cases_label_origins": {},
        "cases_label_directions": {},
    }

    valid_results_found = False

    for result in processed_results:
        if result is None:
            # skip language tasks
            continue

        # only process results for this specific task
        if result["task_name"] != task_name:
            continue

        valid_results_found = True

        # initialize task data with first valid result
        if task_data["task_type"] is None:
            task_data["task_type"] = result["task_type"]
            task_data["modality"] = result["modality"]
            task_data["domain"] = result["domain"]
            task_data["feature_grid_resolution"] = result["feature_grid_resolution"]

            # Check if all cases have the same patch size and spacing
            all_patch_sizes = [result["patch_size"] for result in processed_results]
            all_patch_spacings = [result["patch_spacing"] for result in processed_results]

            # Set global values if all are the same, otherwise None
            task_data["global_patch_size"] = all_patch_sizes[0] if all_patch_sizes and all(ps == all_patch_sizes[0] for ps in all_patch_sizes) else None
            task_data["global_patch_spacing"] = all_patch_spacings[0] if all_patch_spacings and all(ps == all_patch_spacings[0] for ps in all_patch_spacings) else None


        if result["split"] == "shot":
            task_data["shot_embeddings"].append(result["embeddings"])
            task_data["shot_labels"].append(result["label"])
            task_data["shot_extra_labels"].append(result.get("extra_labels"))
            task_data["shot_ids"].append(result["case_id"])
            task_data["shot_coordinates"].append(result["coordinates"])
            shot_id = result["case_id"]
            task_data["shot_image_sizes"][shot_id] = result["image_size"]
            task_data["shot_image_spacings"][shot_id] = result["image_spacing"]
            task_data["shot_image_origins"][shot_id] = result["image_origin"]
            task_data["shot_image_directions"][shot_id] = result["image_direction"]
            task_data["shot_patch_spacings"][shot_id] = result["patch_spacing"]
            task_data["shot_patch_sizes"][shot_id] = result["patch_size"]
            task_data["shot_label_spacings"][shot_id] = result["label_spacing"]
            task_data["shot_label_sizes"][shot_id] = result["label_size"]
            task_data["shot_label_origins"][shot_id] = result["label_origin"]
            task_data["shot_label_directions"][shot_id] = result["label_direction"]
        elif result["split"] == "case":
            task_data["case_embeddings"].append(result["embeddings"])
            task_data["case_labels"].append(result["label"])
            task_data["case_extra_labels"].append(result.get("extra_labels"))
            task_data["prediction"].append(result.get("prediction"))
            task_data["case_ids"].append(result["case_id"])
            task_data["cases_coordinates"].append(result["coordinates"])
            case_id = result["case_id"]
            task_data["cases_image_spacings"][case_id] = result["image_spacing"]
            task_data["cases_image_sizes"][case_id] = result["image_size"]
            task_data["cases_image_origins"][case_id] = result["image_origin"]
            task_data["cases_image_directions"][case_id] = result["image_direction"]
            task_data["cases_patch_sizes"][case_id] = result["patch_size"]
            task_data["cases_patch_spacings"][case_id] = result["patch_spacing"]
            task_data["cases_label_spacings"][case_id] = result["label_spacing"]
            task_data["cases_label_sizes"][case_id] = result["label_size"]
            task_data["cases_label_origins"][case_id] = result["label_origin"]
            task_data["cases_label_directions"][case_id] = result["label_direction"]

    if not valid_results_found:
        return None

    # post-process the task data
    task_type = task_data["task_type"]
    task_domain = task_data["domain"]

    if task_type in ["classification", "regression"]:
        task_data = process_image_representation(task_data)
    elif task_type == "detection":
        if task_domain == "pathology":
            task_data = process_detection_pathology(task_data)
        elif task_domain in ["CT", "MR"]:
            if task_name != "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams":
                task_data = process_detection_radiology(task_data, task_name)
        else:
            raise ValueError(f"Unknown task domain: {task_domain}")

    return task_data


def extract_embeddings(result):
    """Extract embeddings from a single result."""
    data = {
        "task_type": result["task_type"],
        "modality": result["modality"],
        "domain": result["domain"],
        "feature_grid_resolution": result["feature_grid_resolution"],
    }

    data["embeddings"] = result["embeddings"]
    data["name"] = result["case_id"]
    data["coordinates"] = result["coordinates"]
    data["image_size"] = result["image_size"]
    data["image_spacing"] = result["image_spacing"]
    data["image_origin"] = result["image_origin"]
    data["image_direction"] = result["image_direction"]
    data["patch_spacing"] = result["patch_spacing"]
    data["patch_size"] = result["patch_size"]

    return data


def extract_data(patch_neural_representation):
    # Extract metadata
    metadata: dict[str, Any] = patch_neural_representation["meta"]
    spacing = metadata["patch-spacing"]
    patch_size = metadata["patch-size"]
    patch_spacing = metadata["patch-spacing"]
    feature_grid_resolution = metadata.get("feature-grid-resolution", [1]*len(patch_size))
    image_size = metadata["image-size"]
    image_spacing = metadata["image-spacing"]
    image_origin = metadata["image-origin"]
    image_direction = metadata["image-direction"]

    # Extract patches
    patches = patch_neural_representation["patches"]

    # Extract features and coordinates
    features = np.array([p["features"] for p in patches]).astype(np.float32)
    coordinates = np.array([p["coordinates"] for p in patches])

    return (
        features,
        coordinates,
        spacing,
        patch_size,
        patch_spacing,
        feature_grid_resolution,
        image_size,
        image_spacing,
        image_origin,
        image_direction,
    )


def normalize_metric(task_name, metric_value):
    min_value, max_value = METRIC_DICT[task_name]["range"]
    normalized_value = (metric_value - min_value) / (max_value - min_value)
    return normalized_value


def sanitize_json_content(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_json_content(v) for v in obj]
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(
        obj,
        (
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ),
    ):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, content):
    # Writes a json file with the sanitized content
    content = sanitize_json_content(content)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
