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
from sklearn.metrics import cohen_kappa_score
from sksurv.metrics import concordance_index_censored

from unicorn_eval.adaptors import (
    KNN,
    KNNRegressor,
    TwoLayerPerceptron,
    TwoLayerPerceptronRegressor,
    DensityMap,
    LinearProbing,
    LinearProbingRegressor,
    LogisticRegression,
    SegmentationUpsampling,
    WeightedKNN,
    WeightedKNNRegressor,
)
from unicorn_eval.metrics.dice import compute_dice_score
from unicorn_eval.metrics.f1_score import compute_f1
from unicorn_eval.metrics.vision_language import compute_average_language_metric

METRIC_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": {
        "name": "cohen-kappa-quadratic",
        "fn": partial(cohen_kappa_score, weights="quadratic"),
        "range": (-1, 1),
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
        "name": "froc",
        "fn": compute_f1,
        "range": (0, 1),
    },
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": {
        "name": "froc",
        "fn": compute_f1,
        "range": (0, 1),
    },
    "Task09_segmenting_rois_in_breast_cancer_wsis": {
        "name": "dice",
        "fn": compute_dice_score,
        "range": (0, 1),
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
    shot_extra_labels=None,
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
                    k=k
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
            adaptor = TwoLayerPerceptron(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
            )
        elif task_type == "regression":
            survival = False
            if "survival" in adaptor_name:
                survival = True
            adaptor = TwoLayerPerceptronRegressor(
                shot_features=shot_features,
                shot_labels=shot_labels,
                shot_extra_labels=shot_extra_labels,
                test_features=test_features,
                survival=survival,
                hidden_dim=256,
                num_epochs=100,
                learning_rate=0.001,
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


def evaluate_predictions(task_name, case_ids, test_predictions, test_labels, test_extra_labels=None):

    metrics = {
        "predictions": [],  # list to store individual case results
        "metrics": {}, # dictionary to store main metric
        "additional_metrics": {},  # dictionary to store additional metrics
    }

    for case_id, prediction, ground_truth in zip(case_ids, test_predictions, test_labels):
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
    elif task_name == "Task08_detecting_mitotic_figures_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions, 16)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task09_segmenting_rois_in_breast_cancer_wsis":
        metric_value = metric_fn(test_labels, test_predictions)
        metric_dict[metric_name] = metric_value
    elif task_name == "Task20_generating_caption_from_wsi":
        language_metric_dict = metric_fn(test_labels, test_predictions) # a dictionnary
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


def process_detection(data):
    def extract_points(labels):
        return [np.array([(p["point"][0], p["point"][1]) for p in gt["points"]]).tolist() for gt in labels]
    # add comment
    data["shot_labels"] = extract_points(data["shot_labels"])
    data["case_labels"] = extract_points(data["case_labels"])
    if data["case_extra_labels"] and data["case_extra_labels"][0] is not None:
        data["case_extra_labels"] = np.concatenate(data["case_extra_labels"], axis=0)
    else:
        data["case_extra_labels"] = None
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
            }

        if result["split"] == "shot":
            tasks[task_name]["shot_embeddings"].append(result["embeddings"])
            tasks[task_name]["shot_labels"].append(result["label"])
            tasks[task_name]["shot_extra_labels"].append(result["extra_labels"])
            tasks[task_name]["shot_ids"].append(result["case_id"])
            tasks[task_name]["shot_coordinates"].append(result["coordinates"])
        elif result["split"] == "case":
            tasks[task_name]["case_embeddings"].append(result["embeddings"])
            tasks[task_name]["case_labels"].append(result["label"])
            tasks[task_name]["case_extra_labels"].append(result["extra_labels"])
            tasks[task_name]["prediction"].append(result["prediction"])
            tasks[task_name]["case_ids"].append(result["case_id"])
            tasks[task_name]["cases_coordinates"].append(result["coordinates"])
            case_id = result["case_id"]
            image_size = result["image_size"]
            tasks[task_name]["cases_image_spacings"][case_id] = result["image_spacing"]
            tasks[task_name]["cases_image_sizes"][case_id] = image_size

    for task_name, task_data in tasks.items():
        task_type = task_data["task_type"]
        if task_type in ["classification", "regression"]:
            tasks[task_name] = process_image_representation(task_data)
        elif task_type == "detection":
            tasks[task_name] = process_detection(task_data)

    return tasks


def extract_data(patch_neural_representation):
    # Extract metadata
    spacing = patch_neural_representation["meta"]["patch-spacing"]
    patch_size = patch_neural_representation["meta"]["patch-size"]
    image_size = patch_neural_representation["meta"]["image-size"]
    image_spacing = patch_neural_representation["meta"]["image-spacing"]

    # Extract patches
    patches = patch_neural_representation["patches"]

    # Extract features and coordinates
    features = np.array([p["features"] for p in patches]).astype(np.float32)
    coordinates = np.array([p["coordinates"] for p in patches])

    return features, coordinates, spacing, patch_size, image_size, image_spacing


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