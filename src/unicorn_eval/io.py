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
from pathlib import Path
from pprint import pformat

import numpy as np
import openslide
import pandas as pd
import SimpleITK as sitk

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUNDTRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

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


def load_tif_file(*, location):
    slide = openslide.OpenSlide(location)
    logging.info(f"Image dimensions: {slide.dimensions}")
    level_0 = slide.read_region((0, 0), 0, slide.dimensions)
    #   save_tif(level_0, location.stem)
    level_0_np = np.array(level_0)
    class_labels = level_0_np[:, :, 0]  # shape: (H, W)
    return class_labels


def load_mha_file(*, path: Path | str, return_as_path: bool = False):
    class_labels = sitk.ReadImage(str(path))
    size = class_labels.GetSize()
    origin = class_labels.GetOrigin()
    spacing = class_labels.GetSpacing()
    direction = class_labels.GetDirection()
    label = str(path)
    if not return_as_path:
        label = sitk.GetArrayFromImage(class_labels)

    if class_labels is None:
        raise ValueError("Failed to load class labels from MHA file.")

    return (
        label,
        list(size),
        list(origin),
        list(spacing),
        list(direction),
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

    # remove suffixes "_adc", "_t2w", "_hbv" and "_tissue" from the case name if present
    for suffix in ["_adc", "_t2w", "_hbv", "_tissue"]:
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


def extract_data(patch_neural_representation):
    # Extract metadata
    metadata: dict[str] = patch_neural_representation["meta"]
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