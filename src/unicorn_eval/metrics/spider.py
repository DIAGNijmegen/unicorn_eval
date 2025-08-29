from collections import OrderedDict, defaultdict
from typing import Iterable

import numpy as np
import pandas


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

    def __init__(self, gts, preds, case_ids):
        self.ground_truths = gts
        self.inputs = preds
        self.case_ids = case_ids

    def score_case(self, gt, pred):

        mask_manual = gt.astype(np.int64)
        mask_automatic = pred.astype(np.int64)

        # Construct containers for the per-scan results
        all_dice_scores = defaultdict(list)

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

        return {
            "DiceScoreVertebrae": dice_score_vert,
            "DiceScoreDiscs": dice_score_discs,
            "DiceScoreSpinalCanal": dice_score_SC,
            "OverallDiceScore": overall_dice_score,
            "DetectionRateVertebrae": detection_rate_vert,
            "DetectionRateDiscs": detection_rate_discs
        }

    def compute_metrics(self):
        metric_accumulator = []
        gts = self.ground_truths
        for i, gt in enumerate(gts):
            metric = self.score_case(
                gt, self.inputs[i]
            )
            metric_accumulator.append(metric)

        df = pandas.DataFrame(metric_accumulator)
        metric_columns = [
            "DiceScoreVertebrae",
            "DiceScoreDiscs",
            "DiceScoreSpinalCanal",
            "OverallDiceScore",
            "DetectionRateVertebrae",
            "DetectionRateDiscs",
        ]

        results_metric = {}
        for metric_column in metric_columns:
            results_metric[metric_column] = {
                "mean": df[metric_column].mean(),
                "std": df[metric_column].std(),
            }

        return results_metric["OverallDiceScore"]["mean"]


def compute_spider_score(test_labels, test_predictions, case_ids):
    evaluator = Spider(test_labels, test_predictions, case_ids)
    return evaluator.compute_metrics()
