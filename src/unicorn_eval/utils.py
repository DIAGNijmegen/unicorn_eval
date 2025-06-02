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
from functools import partial

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sksurv.metrics import concordance_index_censored

from unicorn_eval.adaptors import (
    KNN,
    DensityMap,
    KNNRegressor,
    LinearProbing,
    LinearProbingRegressor,
    LogisticRegression,
    MultiLayerPerceptron,
    MultiLayerPerceptronRegressor,
    PatchNoduleRegressor,
    SegmentationUpsampling,
    SegmentationUpsampling3D,
    WeightedKNN,
    WeightedKNNRegressor,
)
from unicorn_eval.metrics.dice import compute_dice_score
from unicorn_eval.metrics.f1_score import compute_f1
from unicorn_eval.metrics.picai_score import compute_picai_score
from unicorn_eval.metrics.sensitivity import compute_cpm
from unicorn_eval.metrics.spider import compute_spider_score
from unicorn_eval.metrics.uls import compute_uls_score
from unicorn_eval.metrics.vision_language import compute_average_language_metric

METRIC_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": {
        "name": "cohen-kappa-quadratic",
        "fn": partial(cohen_kappa_score, weights="quadratic"),
        "range": (-1, 1),
    },
    "Task02_classifying_lung_nodule_malignancy_in_ct": {
        "name": "auc",
        "fn": roc_auc_score,
        "range": (0, 1),
    },
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": {
        "name": "c-index",
        "fn": concordance_index_censored,
        "range": (0, 1),
    },
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": {
        "name": "cohen-kappa-quadratic",
        "fn": partial(cohen_kappa_score, weights="quadratic"),
        "range": (-1, 1),
    },
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": {
        "name": "f1",
        "fn": compute_f1,
        "range": (0, 1),
    },
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": {
        "name": "picai",
        "fn": compute_picai_score,
        "range": (0,1),
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
        "range": (0, 1),
    },
    "Task10_segmenting_lesions_within_vois_in_ct": {
        "name": "uls_score",
        "fn": compute_uls_score,
        "range": (0, 1),
    },
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": {
        "name": "spider_score",
        "fn": compute_spider_score,
        "range": (0,1),
    },
    "Task20_generating_caption_from_wsi": {
        "name": "average_language_metric",
        "fn": compute_average_language_metric,
        "range": (0, 1),
    },
}


def adapt_features(
    *,
    adaptor_name: str,
    task_type: str,
    shot_features: np.ndarray,
    shot_labels: np.ndarray,
    test_features: np.ndarray,
    shot_coordinates=None,
    test_coordinates=None,
    shot_names=None,
    test_names=None,
    patch_size=224,
    test_image_sizes=None,
    test_image_spacing=None,
    test_image_origins=None,
    test_image_directions=None,
    test_label_sizes=None,
    test_label_spacing=None,
    test_label_origins=None,
    test_label_directions=None,
    shot_image_sizes=None,
    shot_image_spacing=None,
    shot_image_origins=None,
    shot_image_directions=None,
    shot_label_spacing=None,
    shot_label_origins=None,
    shot_label_directions=None,
    shot_extra_labels=None,
    return_probabilities=False,
):
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
                    k=k
                )
        else:
            if task_type == "classification":
                adaptor = KNN(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k
                )
            elif task_type == "regression":
                adaptor = KNNRegressor(
                    shot_features=shot_features,
                    shot_labels=shot_labels,
                    test_features=test_features,
                    k=k
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
        adaptor = DensityMap(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            patch_size=patch_size[0],
            heatmap_size=16,
        )

    elif adaptor_name == "segmentation-upsampling":
        adaptor = SegmentationUpsampling(
            shot_features=shot_features,
            shot_labels=shot_labels,
            shot_coordinates=shot_coordinates,
            shot_names=shot_names,
            test_features=test_features,
            test_coordinates=test_coordinates,
            test_names=test_names,
            test_image_sizes=test_image_sizes,
            patch_size=patch_size[0],
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
            test_names=test_names, # try to remove this input
            test_image_sizes=test_image_sizes,
            test_image_origins=test_image_origins,
            test_image_spacings=test_image_spacing,
            test_image_directions=test_image_directions,
            test_label_sizes=test_label_sizes,
            test_label_spacing=test_label_spacing,
            test_label_origins=test_label_origins,
            test_label_directions=test_label_directions,
            patch_size=patch_size,
            shot_image_sizes=shot_image_sizes,
            shot_image_spacing=shot_image_spacing,
            shot_image_origins=shot_image_origins,
            shot_image_directions=shot_image_directions,
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
            patch_size=patch_size,
            return_binary=False,
        )

    else:
        raise ValueError(f"Unknown adaptor: {adaptor_name}")

    adaptor.fit()
    predictions = adaptor.predict()
    return predictions


def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types."""
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


def evaluate_predictions(task_name, case_ids, test_predictions, test_labels, test_extra_labels=None, test_image_spacing=None):

    metrics = {
        "predictions": [],          # list to store individual case results
        "metrics": {},              # dictionary to store main metric
        "additional_metrics": {},   # dictionary to store additional metrics
    }

    # To get all nodule predictions for Task 7
    if task_name == "Task07_detecting_lung_nodules_in_thoracic_ct":
        # One entry containing the full arrays
        metrics["predictions"].append(
            {
                "case_id": convert_numpy_types(case_ids),
                "ground_truth": convert_numpy_types(test_labels),
                "prediction": convert_numpy_types(test_predictions),
            }
        )
    else:
        iterable = zip(case_ids, test_predictions, test_labels)
        for case_id, prediction, ground_truth in iterable:
            ground_truth = convert_numpy_types(ground_truth)
            prediction = convert_numpy_types(prediction)
            metrics["predictions"].append(
                {
                    "case_id": case_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
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
        malignancy_risk = test_predictions[:,1]
        metric_value = metric_fn(test_labels, malignancy_risk)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies":
        events = test_extra_labels["event"].astype(bool)
        metric_value = metric_fn(events, test_labels, -test_predictions)[0]
        metric_dict[metric_name] = metric_value
    elif task_name == "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer":
        metric_value = metric_fn(test_labels, test_predictions, 8)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task07_detecting_lung_nodules_in_thoracic_ct":
        metric_value = metric_fn(case_ids, test_predictions, test_labels, test_extra_labels)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task08_detecting_mitotic_figures_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions, 16)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task09_segmenting_rois_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task10_segmenting_lesions_within_vois_in_ct":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri":
        metric_value = metric_fn(test_labels, test_predictions, test_image_spacing, case_ids)
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


def process_detection(data, task_name: str | None = None):
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
                        print("nothing to process in this case (got [(None,)])")

                    elif isinstance(element, dict):
                        for idx, d in enumerate(element.get('points')):
                            diameter_records.append(
                                    {"point": None, "diameter": float(d.get('diameter')), "name": f"case{case_id}_pt{idx}"}
                            )

            elif isinstance(case_extra, (int, float)):
                diameter_records.append(
                    {"point": None, "diameter": float(case_extra), "name": f"case{case_id}"}
                )

            else:
                raise ValueError(f"Unsupported extra_label type: {type(case_extra)}")

        data["case_extra_labels"] = diameter_records

    else:
        data["case_extra_labels"] = np.concatenate(extra_list, axis=0)

    return data


def extract_embeddings_and_labels(processed_results):
    tasks = {}

    for result in processed_results:
        task_name = result["task_name"]

        if task_name not in tasks:
            tasks[task_name] = {
                "task_type": result["task_type"],
                "modality": result["modality"],
                "spacing": result["spacing"],
                "patch_size": result["patch_size"],
                "prediction": [],
                "shot_embeddings": [],
                "shot_coordinates": [],
                "shot_image_spacings": {},
                "shot_image_origins": {},
                "shot_image_directions": {},
                "shot_image_sizes": {},
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
                "cases_label_sizes": {},
                "cases_label_spacings": {},
                "cases_label_origins": {},
                "cases_label_directions": {},
            }

        if result["split"] == "shot":
            tasks[task_name]["shot_embeddings"].append(result["embeddings"])
            tasks[task_name]["shot_labels"].append(result["label"])
            tasks[task_name]["shot_extra_labels"].append(result.get("extra_labels"))
            tasks[task_name]["shot_ids"].append(result["case_id"])
            tasks[task_name]["shot_coordinates"].append(result["coordinates"])
            shot_id = result["case_id"]
            tasks[task_name]["shot_image_sizes"][shot_id] = result["image_size"]
            tasks[task_name]["shot_image_spacings"][shot_id] = result["image_spacing"]
            tasks[task_name]["shot_image_origins"][shot_id] = result["image_origin"]
            tasks[task_name]["shot_image_directions"][shot_id] = result["image_direction"]
            tasks[task_name]["shot_label_spacings"][shot_id] = result["label_spacing"]
            tasks[task_name]["shot_label_sizes"][shot_id] = result["label_size"]
            tasks[task_name]["shot_label_origins"][shot_id] = result["label_origin"]
            tasks[task_name]["shot_label_directions"][shot_id] = result["label_direction"]
        elif result["split"] == "case":
            tasks[task_name]["case_embeddings"].append(result["embeddings"])
            tasks[task_name]["case_labels"].append(result["label"])
            tasks[task_name]["case_extra_labels"].append(result.get("extra_labels"))
            tasks[task_name]["prediction"].append(result.get("prediction"))
            tasks[task_name]["case_ids"].append(result["case_id"])
            tasks[task_name]["cases_coordinates"].append(result["coordinates"])
            case_id = result["case_id"]
            tasks[task_name]["cases_image_spacings"][case_id] = result["image_spacing"]
            tasks[task_name]["cases_image_sizes"][case_id] = result["image_size"]
            tasks[task_name]["cases_image_origins"][case_id] = result["image_origin"]
            tasks[task_name]["cases_image_directions"][case_id] = result["image_direction"]
            tasks[task_name]["cases_label_spacings"][case_id] = result["label_spacing"]
            tasks[task_name]["cases_label_sizes"][case_id] = result["label_size"]
            tasks[task_name]["cases_label_origins"][case_id] = result["label_origin"]
            tasks[task_name]["cases_label_directions"][case_id] = result["label_direction"]

    # now post-process each task
    for task_name, task_data in tasks.items():
        task_type = task_data["task_type"]
        if task_type in ["classification", "regression"]:
            tasks[task_name] = process_image_representation(task_data)
        elif task_type == "detection":
            if not task_name == "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams":
                tasks[task_name] = process_detection(task_data, task_name)
    return tasks


def extract_data(patch_neural_representation):
    # Extract metadata
    spacing = patch_neural_representation["meta"]["patch-spacing"]
    patch_size = patch_neural_representation["meta"]["patch-size"]
    image_size = patch_neural_representation["meta"]["image-size"]
    image_spacing = patch_neural_representation["meta"]["image-spacing"]
    image_origin = patch_neural_representation["meta"]["image-origin"]
    image_direction = patch_neural_representation["meta"]["image-direction"]

    # Extract patches
    patches = patch_neural_representation["patches"]

    # Extract features and coordinates
    features = np.array([p["features"] for p in patches]).astype(np.float32)
    coordinates = np.array([p["coordinates"] for p in patches])

    return features, coordinates, spacing, patch_size, image_size, image_spacing, image_origin, image_direction


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
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, content):
    # Writes a json file with the sanitized content
    content = sanitize_json_content(content)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))