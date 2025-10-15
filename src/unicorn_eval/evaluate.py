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

import copy
import gc
import json
import logging
import multiprocessing
import random
import re
import shutil
import argparse
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from dragon_eval import DragonEval
from dragon_eval.evaluation import REGRESSION_EPSILON, TASK_TYPE, EvalType

from unicorn_eval.helpers import get_max_workers
from unicorn_eval.io import (
    GROUNDTRUTH_DIRECTORY,
    INPUT_DIRECTORY,
    OUTPUT_DIRECTORY,
    process,
    read_inputs,
    write_json_file,
)
from unicorn_eval.utils import (
    METRIC_DICT,
    evaluate_predictions,
    extract_embeddings_and_labels,
    extract_embeddings_labels_and_predictions,
    extract_labels,
    get_adaptor,
    normalize_metric,
    set_all_seeds,
    set_lowest_possible_metric,
)

# Matches BMP PUA (U+E000–U+F8FF) and the supplementary PUA ranges (planes 15 & 16)
PUA_REGEX = re.compile(r"[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")


ADAPTOR_SLUGS_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": "adaptor-pathology-classification",
    "Task02_classifying_lung_nodule_malignancy_in_ct": "adaptor-radiology-classification",
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": "adaptor-pathology-regression",
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": "adaptor-pathology-classification",
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": "adaptor-pathology-detection",
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": "adaptor-radiology-detection-segmentation",
    "Task07_detecting_lung_nodules_in_thoracic_ct": "adaptor-radiology-detection-points",
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": "adaptor-pathology-detection",
    "Task09_segmenting_rois_in_breast_cancer_wsis": "adaptor-pathology-segmentation",
    "Task10_segmenting_lesions_within_vois_in_ct": "adaptor-radiology-segmentation",
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": "adaptor-radiology-segmentation",
}

DETERMINISTIC_ADAPTORS = [
    "1-nn",
    "5-nn",
    "20-nn",
    "1-nn-weighted",
    "5-nn-weighted",
    "20-nn-weighted",
]

REQUIRES_PROBABILITIES_DICT = {
    "Task01_classifying_he_prostate_biopsies_into_isup_scores": False,
    "Task02_classifying_lung_nodule_malignancy_in_ct": True,
    "Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies": False,
    "Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi": False,
    "Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer": False,
    "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams": False,
    "Task07_detecting_lung_nodules_in_thoracic_ct": False,
    "Task08_detecting_mitotic_figures_in_breast_cancer_wsis": False,
    "Task09_segmenting_rois_in_breast_cancer_wsis": False,
    "Task10_segmenting_lesions_within_vois_in_ct": False,
    "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri": False,
}

LANGUAGE_TASK_NAMES = [
    "Task12_predicting_histopathology_sample_origin",
    "Task13_classifying_pulmonary_nodule_presence",
    "Task14_classifying_kidney_abnormality",
    "Task15_hip_kellgren_lawrence_score",
    "Task16_classifying_colon_histopathology_diagnosis",
    "Task17_predicting_lesion_size_measurements",
    "Task18_predicting_prostate_volume_psa_and_psa_density",
    "Task19_anonymizing_report",
]

TASK_TYPE.update(
    {
        "Task12_predicting_histopathology_sample_origin": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
        "Task13_classifying_pulmonary_nodule_presence": EvalType.BINARY_CLASSIFICATION,
        "Task14_classifying_kidney_abnormality": EvalType.BINARY_CLASSIFICATION,
        "Task15_hip_kellgren_lawrence_score": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
        "Task16_classifying_colon_histopathology_diagnosis": EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK,
        "Task17_predicting_lesion_size_measurements": EvalType.REGRESSION,
        "Task18_predicting_prostate_volume_psa_and_psa_density": EvalType.REGRESSION,
        "Task19_anonymizing_report": EvalType.TEXT_TARGET,
    }
)

REGRESSION_EPSILON.update(
    {
        "Task17_predicting_lesion_size_measurements": 4,
        "Task18_predicting_prostate_volume_psa_and_psa_density": np.array(
            [4, 0.4, 0.04]
        ),
    }
)


def print_directory_contents(path: Path | str):
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir():
            print_directory_contents(child)
        else:
            logging.info(str(child))


def read_adaptors():
    # read the adaptors that are used for this submission
    adaptors = {}
    for task_name, slug in ADAPTOR_SLUGS_DICT.items():
        adaptor_path = INPUT_DIRECTORY / f"{slug}.json"
        if adaptor_path.exists():
            with open(adaptor_path) as f:
                adaptors[task_name] = json.loads(f.read())
    return adaptors


def write_combined_metrics(
    *, metric_dict: dict[str, dict], save_predictions: bool = True
) -> None:
    metrics = {"metrics": {}, "std": {}, "normalized_metrics": {}, "additional_std": {}}
    predictions = {"predictions": []}

    for task_name, task_metrics in metric_dict.items():

        task_identifier = task_name.split("_")[0]

        for metric_name, metric_value in task_metrics["metrics"].items():
            metrics["metrics"][f"{task_identifier}_{metric_name}"] = metric_value
            metrics["normalized_metrics"][f"{task_identifier}_{metric_name}"] = (
                normalize_metric(task_name, metric_value)
            )

        for metric_name, std_value in task_metrics.get(
            "std", {}
        ).items():
            metrics["std"][f"{task_identifier}_{metric_name}"] = std_value

        for metric_name, metric_value in task_metrics.get(
            "additional_metrics", {}
        ).items():
            metrics["metrics"][f"{task_identifier}_{metric_name}"] = metric_value

        for metric_name, std_value in task_metrics.get(
            "additional_std", {}
        ).items():
            metrics["std"][f"{task_identifier}_{metric_name}"] = std_value

        if save_predictions:
            case_prediction = [
                p.tolist() if isinstance(p, np.ndarray) else p
                for p in task_metrics["predictions"]
            ]
            predictions["predictions"].extend(case_prediction)

    # aggregate metrics when there are multiple tasks
    metrics["metrics"]["mean"] = np.mean(
        [metric_value for _, metric_value in metrics["normalized_metrics"].items()]
    )

    logging.info(f"metrics={metrics}")
    write_json_file(
        location=OUTPUT_DIRECTORY / "metrics.json",
        content=metrics,
    )

    if save_predictions:
        write_json_file(
            location=OUTPUT_DIRECTORY / "predictions.json",
            content=predictions,
        )


def reformat_language_metrics(metrics: dict) -> dict:
    """
    Reformat the language metrics to match the expected format for write_combined_metrics.
    """
    # If empty, return an empty dictionary
    if not metrics:
        return {}
    else:
        return {
            task: {"metrics": {task: values["mean"]}}
            for task, values in metrics["aggregates"].items()
            if task != "overall"
        }


def prepare_predictions_language(input_dir: Path, output_dir: Path, gt_dir: Path):
    """
    Map the predictions with random filenames to the correct task and fold.

    New behavior:
    - If a prediction file matches a GT task except for a few missing cases:
        * If missing <= 5% of that task's total UIDs, we fill the gaps by
          duplicating a randomly chosen existing prediction and replacing only
          the 'uid' field with each missing UID.
        * If missing > 5%, we raise a ValueError.
    - If a prediction file contains UIDs not present in the matched GT task,
      we warn and skip that file.
    """
    task_uids = {}
    logging.info(f"Scanning ground truth directory: {gt_dir}")
    for gt_file in gt_dir.rglob("*.json"):
        if any(task_name in str(gt_file) for task_name in LANGUAGE_TASK_NAMES):
            with open(gt_file, "r") as f:
                entries = json.load(f)
            matched_task = next(
                task for task in LANGUAGE_TASK_NAMES if task in str(gt_file)
            )
            uids = set(entry["uid"] for entry in entries)
            task_uids[matched_task] = uids
            logging.info(
                f"→ Found GT task '{matched_task}' with {len(uids)} UIDs from {gt_file}"
            )

    for pred_file in input_dir.rglob("*/output/nlp-predictions-dataset.json"):
        if pred_file.name in ["predictions.json", "inputs.json"]:
            continue

        logging.info(f"\nProcessing prediction file: {pred_file}")
        with open(pred_file, "r") as f:
            entries = json.load(f)

        uids = set(entry["uid"] for entry in entries)
        matched = False

        for task, gt_uids in task_uids.items():
            # Only proceed if all predicted UIDs are part of this GT task.
            extra_uids = uids - gt_uids
            if extra_uids:
                # This prediction file cannot belong to this task.
                continue

            # Check how many are missing vs GT.
            missing_uids = gt_uids - uids
            total_gt = len(gt_uids)

            if not missing_uids and uids == gt_uids:
                # Perfect match
                output_file = (
                    output_dir / f"{task}-fold0" / "nlp-predictions-dataset.json"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(entries, f)
                logging.info(
                    f"✓ Matched to task '{task}', wrote {len(entries)} entries → {output_file}"
                )
                matched = True
                break

            # If we reach here, there are no extra UIDs, but some may be missing.
            if missing_uids:
                missing_ratio = len(missing_uids) / float(total_gt)
                percent = round(missing_ratio * 100, 3)

                if missing_ratio > 0.05:
                    # Too many missing → error
                    raise ValueError(
                        f"Prediction file {pred_file} matched task '{task}' but is missing "
                        f"{len(missing_uids)}/{total_gt} UIDs ({percent}%), which exceeds the 5% threshold."
                    )

                # Fill small gaps by duplicating a random existing prediction and replacing only the 'uid'.
                logging.info(
                    f"• '{task}' missing {len(missing_uids)}/{total_gt} UIDs ({percent}%). "
                    f"Filling gaps by duplicating random predictions."
                )
                if not entries:
                    raise ValueError(
                        f"Cannot fill missing UIDs for {pred_file}: no existing entries to sample from."
                    )

                for missing_uid in missing_uids:
                    template = copy.deepcopy(random.choice(entries))
                    template["uid"] = missing_uid
                    entries.append(template)

                # Sanity: now we should match exactly
                final_uids = set(e["uid"] for e in entries)
                assert (
                    final_uids == gt_uids
                ), "Internal error: after gap-filling, UIDs still do not match GT."

                output_file = (
                    output_dir / f"{task}-fold0" / "nlp-predictions-dataset.json"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(entries, f)
                logging.info(
                    f"✓ Matched to task '{task}' after filling, wrote {len(entries)} entries → {output_file}"
                )
                matched = True
                break

        if not matched:
            logging.info(
                f"⚠ No matching ground truth found for {pred_file} "
                f"(either extra UIDs present or task mismatch). "
                f"Pred UID count = {len(uids)}"
            )


def _strip_pua(s: str, stats: dict) -> str:
    if not s:
        return s
    hits = PUA_REGEX.findall(s)
    if hits:
        stats["total_removed"] += len(hits)
        # save a few examples of what we saw (as code points)
        if len(stats["examples"]) < 10:
            stats["examples"].extend(
                [f"U+{ord(ch):04X}" for ch in hits[: 10 - len(stats["examples"])]]
            )
    return PUA_REGEX.sub("", s)


def _clean_obj(obj, stats: dict):
    """Recursively clean strings inside dicts/lists/tuples; leave other types untouched."""
    if isinstance(obj, str):
        return _strip_pua(obj, stats)
    elif isinstance(obj, list):
        return [_clean_obj(x, stats) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_obj(x, stats) for x in obj)
    elif isinstance(obj, dict):
        # Clean both keys and values (keys are rarely non-ASCII, but just in case)
        new_dict = {}
        for k, v in obj.items():
            new_k = _clean_obj(k, stats) if isinstance(k, str) else k
            new_v = _clean_obj(v, stats)
            new_dict[new_k] = new_v
        return new_dict
    else:
        return obj


def evaluate_language_predictions():
    input_dir = (
        INPUT_DIRECTORY
        if INPUT_DIRECTORY.exists()
        else Path("unicorn/test-predictions")
    )
    # Make a temp folder
    temp_metric_output_path = Path("/opt/app/predictions/language_results/metrics.json")
    temp_metric_output_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = (
        Path("/opt/app/predictions")
        if Path("/opt/app/predictions").exists()
        else Path("unicorn/workdir")
    )
    prepare_predictions_language(
        input_dir=input_dir,
        output_dir=workdir,
        gt_dir=GROUNDTRUTH_DIRECTORY,
    )

    # determine which task we are evaluating
    files = list(GROUNDTRUTH_DIRECTORY.glob("*.json"))
    task_names = [
        path.stem.replace(".json", "")
        for path in files
        if path.stem.replace(".json", "") in LANGUAGE_TASK_NAMES
    ]

    if task_names:
        # Remove restricted unicode characters from input of task 19
        if "Task19_anonymizing_report" in task_names:
            input_path = (
                workdir
                / "Task19_anonymizing_report-fold0"
                / "nlp-predictions-dataset.json"
            )
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Clean the loaded JSON
                stats = {"total_removed": 0, "examples": []}
                cleaned = _clean_obj(data, stats)

                # Only write if anything changed
                if stats["total_removed"] > 0:
                    # Make a backup before overwriting
                    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
                    if not backup_path.exists():
                        shutil.copy2(input_path, backup_path)

                    # Write cleaned JSON (preserve UTF-8 as-is)
                    with open(input_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned, f, ensure_ascii=False, indent=2)

                    print(
                        f"[PUA scrub] Removed {stats['total_removed']} private-use characters "
                        f"from '{input_path.name}'. Examples: {', '.join(stats['examples']) or 'n/a'}"
                    )
                else:
                    print(
                        f"[PUA scrub] No private-use characters found in '{input_path.name}'."
                    )

        # evaluate
        DragonEval(
            ground_truth_path=GROUNDTRUTH_DIRECTORY,
            predictions_path=workdir,
            output_file=temp_metric_output_path,
            folds=[0],
            tasks=task_names,
        ).evaluate()

        # load the metrics
        with open(temp_metric_output_path, "r") as f:
            return json.load(f)

    else:
        logging.info("No language tasks found in the ground truth files.")
        return {}


def process_task_in_subprocess(
    task_name, mapping, adaptors, save_predictions, metrics_path, use_multiprocessing=True
):
    logging.info(f"Processing task in subprocess: {task_name}")

    max_workers = get_max_workers()
    modality = mapping[(mapping.task_name == task_name)]["modality"].values[0]

    if modality == "vision":

        # ensure we have an adaptor for this task
        if task_name not in adaptors:
            raise Exception(f"No adaptor found for task {task_name}")

        adaptor_name = adaptors[task_name]
        return_probabilities = REQUIRES_PROBABILITIES_DICT[task_name]

        num_run = 5
        if adaptor_name in DETERMINISTIC_ADAPTORS:
            num_run = 1
        first_metric, first_additional_metric = True, True
        metrics = {}
        running_metrics = {"metrics": {}, "additional_metrics": {}}

        for seed in range(num_run):

            logging.info(
                f"Run {seed+1}/{num_run} for task {task_name} using {adaptor_name}"
            )
            set_all_seeds(seed)

            # only load few shots for the given task
            task_shots = mapping[
                (mapping.task_name == task_name) & (mapping.split == "shot")
            ]["case_id"].tolist()
            shot_inputs = read_inputs(input_dir=INPUT_DIRECTORY, case_names=task_shots)

            if use_multiprocessing:
                pool = multiprocessing.Pool(processes=max_workers)
                shots = pool.map(process, shot_inputs)
                pool.close()
                pool.join()
            else:
                shots = [process(shot) for shot in shot_inputs]

            del shot_inputs
            gc.collect()
            shot_informations = extract_embeddings_and_labels(shots, task_name)
            del shots
            gc.collect()

            if shot_informations is None:
                logging.info(f"No shots found for task {task_name}, skipping.")
                logging.info("=+=" * 10)
                return 0

            else:

                global_patch_size = shot_informations["global_patch_size"]
                global_patch_spacing = shot_informations["global_patch_spacing"]
                feature_grid_resolution = shot_informations["feature_grid_resolution"]

                shot_embeddings = shot_informations["embeddings"]
                shot_coordinates = shot_informations["coordinates"]
                shot_labels = shot_informations["labels"]
                shot_extra_labels = shot_informations["extra_labels"]
                shot_ids = shot_informations["ids"]
                shot_image_sizes = shot_informations["image_sizes"]
                shot_image_spacings = shot_informations["image_spacings"]
                shot_image_origins = shot_informations["image_origins"]
                shot_image_directions = shot_informations["image_directions"]
                shot_patch_sizes = shot_informations["patch_sizes"]
                shot_patch_spacings = shot_informations["patch_spacings"]
                shot_label_spacings = shot_informations["label_spacings"]
                shot_label_origins = shot_informations["label_origins"]
                shot_label_directions = shot_informations["label_directions"]

                task_type = shot_informations["task_type"]
                if task_type in ["classification", "regression"]:
                    save_predictions = True
                    if len(shot_embeddings.shape) > 2:
                        shot_embeddings = shot_embeddings.squeeze(1)

                num_shots = len(shot_ids)

                adaptor = get_adaptor(
                    adaptor_name=adaptor_name,
                    task_type=task_type,
                    num_shots=num_shots,
                    feature_grid_resolution=feature_grid_resolution,
                    global_patch_size=global_patch_size,
                    global_patch_spacing=global_patch_spacing,
                    return_probabilities=return_probabilities,
                    seed=seed,
                )

                adaptor.fit(
                    shot_features=shot_embeddings,
                    shot_labels=shot_labels,
                    shot_ids=shot_ids,
                    shot_coordinates=shot_coordinates,
                    shot_patch_sizes=shot_patch_sizes,
                    shot_patch_spacings=shot_patch_spacings,
                    shot_extra_labels=shot_extra_labels,
                    shot_image_sizes=shot_image_sizes,
                    shot_image_spacings=shot_image_spacings,
                    shot_image_origins=shot_image_origins,
                    shot_image_directions=shot_image_directions,
                    shot_label_spacings=shot_label_spacings,
                    shot_label_origins=shot_label_origins,
                    shot_label_directions=shot_label_directions,
                )

                del (
                    shot_embeddings,
                    shot_labels,
                    shot_extra_labels,
                    shot_ids,
                    shot_coordinates,
                    shot_image_sizes,
                    shot_image_spacings,
                    shot_image_origins,
                    shot_image_directions,
                    shot_patch_sizes,
                    shot_patch_spacings,
                    shot_label_spacings,
                    shot_label_origins,
                    shot_label_directions,
                    shot_informations,
                )
                gc.collect()

                task_cases = mapping[
                    (mapping.task_name == task_name) & (mapping.split == "case")
                ]["case_id"].tolist()
                case_inputs = read_inputs(
                    input_dir=INPUT_DIRECTORY, case_names=task_cases
                )

                if use_multiprocessing:
                    pool = multiprocessing.Pool(processes=max_workers)
                    cases = pool.map(process, case_inputs)
                    pool.close()
                    pool.join()
                else:
                    cases = [process(case) for case in case_inputs]

                del case_inputs
                gc.collect()
                case_information = extract_labels(cases, task_name)
                del cases
                gc.collect()

                if case_information is None:
                    raise ValueError(f"No cases found for task {task_name}")
                case_labels = case_information["labels"]

                predictions = adaptor.predict(case_information["ids"])

                run_metrics = evaluate_predictions(
                    task_name=task_name,
                    case_ids=case_information["ids"],
                    test_predictions=predictions,
                    test_labels=case_labels,
                    test_extra_labels=case_information.get("extra_labels"),
                    save_predictions=save_predictions,
                )

                del case_information, case_labels, predictions
                gc.collect()

                # store metrics
                for metric_name, metric_value in run_metrics["metrics"].items():
                    if first_metric:
                        first_metric = False
                        running_metrics["metrics"][metric_name] = [metric_value]
                    else:
                        running_metrics["metrics"][metric_name].append(metric_value)
                for metric_name, metric_value in run_metrics[
                    "additional_metrics"
                ].items():
                    if first_additional_metric:
                        first_additional_metric = False
                        running_metrics["additional_metrics"][
                            metric_name
                        ] = [metric_value]
                    else:
                        running_metrics["additional_metrics"][
                            metric_name
                        ].append(metric_value)

        # average metrics
        metrics = {
            "metrics": {k: np.mean(v) for k, v in running_metrics["metrics"].items()},
            "std": {k: np.std(v) for k, v in running_metrics["metrics"].items()},
            "additional_metrics": {k: np.mean(v) for k, v in running_metrics["additional_metrics"].items()},
            "additional_std": {k: np.std(v) for k, v in running_metrics["additional_metrics"].items()},
        }

    elif modality == "vision-language":

        task_cases = mapping[
            (mapping.task_name == task_name) & (mapping.split == "case")
        ]["case_id"].tolist()
        case_inputs = read_inputs(input_dir=INPUT_DIRECTORY, case_names=task_cases)

        if use_multiprocessing:
            pool = multiprocessing.Pool(processes=max_workers)
            cases = pool.map(process, case_inputs)
            pool.close()
            pool.join()
        else:
            cases = [process(case) for case in case_inputs]

        case_information = extract_embeddings_labels_and_predictions(cases, task_name)
        if case_information is None:
            raise ValueError(f"No cases found for task {task_name}")

        predictions = [pred["text"] for pred in case_information["prediction"]]
        case_labels = [
            label["text"] for case in case_information["labels"] for label in case
        ]

        metrics = evaluate_predictions(
            task_name=task_name,
            case_ids=case_information["ids"],
            test_predictions=predictions,
            test_labels=case_labels,
            test_extra_labels=case_information.get("extra_labels"),
            save_predictions=save_predictions,
        )

        del case_information, predictions, case_labels
        gc.collect()

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # save metrics
    write_json_file(location=metrics_path, content=metrics)


def main():

    parser = argparse.ArgumentParser(description="Unicorn Eval runner")
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        help="List of task names to evaluate",
    )
    args = parser.parse_args()
    task_names = args.tasks

    logging.info("Input folder contents:")
    print_directory_contents(INPUT_DIRECTORY)
    logging.info("=+=" * 10)
    logging.info("Groundtruth folder contents:")
    print_directory_contents(GROUNDTRUTH_DIRECTORY)
    logging.info("=+=" * 10)

    print("Evaluating language predictions")
    task_metrics = reformat_language_metrics(evaluate_language_predictions())
    print("=+=" * 10)

    metrics = {}
    adaptors = read_adaptors()

    # get mapping to determine list of tasks to evaluate
    mapping_path = GROUNDTRUTH_DIRECTORY / "mapping.csv"
    try:
        mapping = pd.read_csv(mapping_path, dtype={"case_id": str})
        if task_names is not None:
            all_tasks = task_names
            assert all(
                task in mapping["task_name"].unique() for task in all_tasks
            ), "One or more specified task names are not present in the mapping.csv file."
        else:
            all_tasks = mapping["task_name"].unique()
        save_predictions = False

        for task_name in all_tasks:
            use_multiprocessing = True
            if task_name == "Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams":
                use_multiprocessing = False  # disable multiprocessing for this task due to memory issues
            print(f"Processing task: {task_name} (in subprocess)")
            metrics_path = OUTPUT_DIRECTORY / f"{task_name}.json"

            # create a queue to capture errors from the subprocess
            error_queue = multiprocessing.Queue()
            def wrapped_process_task():
                try:
                    process_task_in_subprocess(
                        task_name, mapping, adaptors, save_predictions, metrics_path, use_multiprocessing
                    )
                except Exception as e:
                    # capture the exception and traceback, then put it in the queue
                    error_queue.put((task_name, traceback.format_exc()))

            # start the subprocess
            p = multiprocessing.Process(target=wrapped_process_task)
            p.start()
            p.join()

            # check if there was an error in the subprocess
            if not error_queue.empty():
                task_name, error_traceback = error_queue.get()
                print(f"⚠️ Error processing task {task_name}")
                print(f"🛠️ Traceback:\n{error_traceback}")
                metrics = set_lowest_possible_metric(task_name)
                write_json_file(location=metrics_path, content=metrics)
                print(f"🚨 Setting lowest possible metric for task {task_name} due to error.")
            elif not metrics_path.exists():
                print(f"❌ Metrics file not found for task {task_name}.")
                print(f"🚨 Setting lowest possible metric for task {task_name}.")
                metrics = set_lowest_possible_metric(task_name)
                write_json_file(location=metrics_path, content=metrics)
            else:
                print(f"✅ Successfully completed processing task: {task_name}")

            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                task_metrics[task_name] = metrics
            print("=+=" * 10)

        # set lowest possible metric for skipped tasks
        for task_name in METRIC_DICT.keys():
            if task_name not in task_metrics:
                task_metrics[task_name] = set_lowest_possible_metric(task_name)

        logging.info(f"Writing metrics for {len(task_metrics)} tasks...")
        write_combined_metrics(metric_dict=task_metrics, save_predictions=False)
        logging.info("Metrics written successfully.")
        return 0

    except FileNotFoundError:
        logging.info(f"Writing metrics for {len(task_metrics)} tasks...")
        write_combined_metrics(metric_dict=task_metrics, save_predictions=False)
        logging.info("Metrics written successfully.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
