import SimpleITK
from collections import OrderedDict, defaultdict
import numpy as np
import SimpleITK as sitk
import pandas

from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)

from typing import Iterable, Optional, List, Union
import json
from pathlib import Path


def surface_distances(
    manual: Iterable[bool],
    automatic: Iterable[bool],
    voxel_spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> Iterable[float]:
    """Computes the surface distances (positive numbers) from all border voxels of a binary object in two images."""
    manual_mask = np.asarray(manual, dtype="bool")
    automatic_mask = np.asarray(automatic, dtype="bool")

    if np.count_nonzero(manual_mask) == 0 or np.count_nonzero(automatic_mask) == 0:
        raise ValueError(
            "Cannot compute surface distance if there are no foreground pixels in the image"
        )

    if connectivity is None:
        connectivity = manual_mask.ndim

    # Extract border using erosion
    footprint = generate_binary_structure(manual_mask.ndim, connectivity)
    manual_border = manual_mask ^ binary_erosion(
        manual_mask, structure=footprint, iterations=1
    )
    automatic_border = automatic_mask ^ binary_erosion(
        automatic_mask, structure=footprint, iterations=1
    )

    # Compute average surface distance
    dt = distance_transform_edt(~manual_border, sampling=voxel_spacing)
    return dt[automatic_border]


def average_surface_distance(
    manual: Iterable[bool],
    automatic: Iterable[bool],
    voxel_spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
    symmetric: bool = True,
) -> float:
    """
    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    manual
        Reference masks (binary)

    automatic
        Masks that is compared to the reference mask

    voxel_spacing
        Spacing between elements in the images

    connectivity
        The neighbourhood/connectivity considered when determining the surface of the binary objects. Values between 1 and ndim are valid.
        Defaults to ndim, which is full connectivity even along the diagonal.

    symmetric
        Whether the surface distance are calculated from manual to automatic mask, or symmetrically in both directions

    Returns
    -------
    float
        Average surface distance
    """
    sd1 = surface_distances(manual, automatic, voxel_spacing, connectivity)
    if not symmetric:
        return np.asarray(sd1).mean()

    sd2 = surface_distances(automatic, manual, voxel_spacing, connectivity)
    return float(np.concatenate((sd1, sd2)).mean())


def dice_score(mask1: Iterable[bool], mask2: Iterable[bool]) -> float:
    """Dice volume overlap score for two binary masks"""
    m1 = np.asarray(mask1, dtype="bool").flatten()
    m2 = np.asarray(mask2, dtype="bool").flatten()

    try:
        return (
            2
            * np.count_nonzero(m1 & m2)
            / float(np.count_nonzero(m1) + np.count_nonzero(m2))
        )
    except ZeroDivisionError:
        raise ValueError("Cannot compute dice score on empty masks")


class Spider:

    def __init__(self, gts, preds, spacings, case_ids):
        self.ground_truths = gts
        self.inputs = preds
        self.spacings = spacings
        self.case_ids = case_ids

    def score_case(self, gt, pred, spacing):

        mask_manual = gt.astype(np.int64)
        mask_automatic = pred.astype(np.int64)

        # Construct containers for the per-scan results
        all_dice_scores = defaultdict(list)
        all_surface_distances = defaultdict(list)

        # Check if manual and automatic mask have the same dimensions
        if mask_manual.shape != mask_automatic.shape:
            print(
                " > Manual and automatic masks have different shapes: {} vs {}".format(
                    mask_manual.shape, mask_automatic.shape
                )
            )

        # build lookup table for all labels
        label_lut = OrderedDict()
        all_labels_manual = sorted(list(np.unique(mask_manual[mask_manual > 0])))

        for label_manual in all_labels_manual:
            # Determine label in automatic mask with which this label overlaps the most
            overlap_automatic = mask_automatic[mask_manual == label_manual]
            overlap_automatic_foreground = overlap_automatic > 0
            if np.any(overlap_automatic_foreground):
                label_automatic = np.bincount(
                    overlap_automatic[overlap_automatic_foreground]
                ).argmax()
                label_lut[label_manual] = label_automatic

        dice_scores_vert = []
        dice_scores_discs = []
        total_vert = 0
        total_discs = 0
        missed_vert = 0
        missed_discs = 0
        detection_threshold = 0.1

        for label_manual in all_labels_manual:
            if label_manual not in label_lut:
                score = 0
            else:
                label_automatic = label_lut[label_manual]
                mask1 = mask_manual == label_manual
                mask2 = mask_automatic == label_automatic
                if not mask1.any() and not mask2.any():
                    score = 1.0
                elif not mask1.any() or not mask2.any():
                    score = 0.0
                else:
                    score = dice_score(mask1, mask2)

            if "dice_score_SC" in locals():
                pass
            else:
                dice_score_SC = 999

            if label_manual > 0 and label_manual < 100:
                total_vert += 1
                if score < detection_threshold:
                    missed_vert += 1
                else:
                    dice_scores_vert.append(score)
            elif label_manual > 200:
                total_discs += 1
                if score < detection_threshold:
                    missed_discs += 1
                else:
                    dice_scores_discs.append(score)
            elif label_manual == 100:
                dice_score_SC = score

            all_dice_scores[label_manual].append(score)

        if dice_scores_vert:
            dice_score_vert = np.mean(dice_scores_vert)
        else:
            dice_score_vert = 0.0

        if dice_scores_discs:
            dice_score_discs = np.mean(dice_scores_discs)
        else:
            dice_score_discs = 0.0

        scores = [v for vs in all_dice_scores.values() for v in vs]
        if scores:
            overall_dice_score = np.mean(scores)
        else:
            overall_dice_score = 0.0

        detection_rate_vert = (total_vert - missed_vert) / total_vert
        detection_rate_discs = (total_discs - missed_discs) / total_discs

        # Calculate mean absolute surface distances
        surface_distances_vert = []
        surface_distances_discs = []
        for label_manual in all_labels_manual:
            if label_manual not in label_lut:
                distance = np.nan
            else:
                label_automatic = label_lut[label_manual]
                distance = average_surface_distance(
                    mask_manual == label_manual,
                    mask_automatic == label_automatic,
                    spacing,
                )

            if "surface_distance_SC" in locals():
                pass
            else:
                surface_distance_SC = 999

            if label_manual > 0 and label_manual < 100:
                surface_distances_vert.append(distance)
            elif label_manual > 200:
                surface_distances_discs.append(distance)
            elif label_manual == 100:
                surface_distance_SC = distance
            all_surface_distances[label_manual].append(distance)

        surface_distance_vert = np.mean(surface_distances_vert)
        surface_distance_discs = np.mean(surface_distances_discs)
        overal_surface_distance = np.mean(
            [v for vs in all_surface_distances.values() for v in vs]
        )

        return {
            "DiceScoreVertebrae": dice_score_vert,
            "DiceScoreDiscs": dice_score_discs,
            "DiceScoreSpinalCanal": dice_score_SC,
            "OveralDiceScore": overall_dice_score,
            "DetectionRateVertebrae": detection_rate_vert,
            "DetectionRateDiscs": detection_rate_discs,
            "ASDVertebrae": surface_distance_vert,
            "ASDDiscs": surface_distance_discs,
            "ASDSpinalCanal": surface_distance_SC,
            "OveralASD": overal_surface_distance,
        }

    def compute_metrics(self):

        metric_accumulator = []
        gts = self.ground_truths
        for i, gt in enumerate(gts):
            metric = self.score_case(
                gt, self.inputs[i], self.spacings.get(self.case_ids[i])
            )
            metric_accumulator.append(metric)

        df = pandas.DataFrame(metric_accumulator)
        metric_columns = [
            "DiceScoreVertebrae",
            "DiceScoreDiscs",
            "DiceScoreSpinalCanal",
            "OveralDiceScore",
            "DetectionRateVertebrae",
            "DetectionRateDiscs",
            "ASDVertebrae",
            "ASDDiscs",
            "ASDSpinalCanal",
            "OveralASD",
        ]

        results_metric = {}
        for metric_column in metric_columns:
            results_metric[metric_column] = {
                "mean": df[metric_column].mean(),
                "std": df[metric_column].std(),
            }

        return results_metric.get("OveralDiceScore").get("mean")


def compute_spider_score(test_labels, test_predictions, test_image_spacing, case_ids):
    evaluator = Spider(test_labels, test_predictions, test_image_spacing, case_ids)
    return evaluator.compute_metrics()
