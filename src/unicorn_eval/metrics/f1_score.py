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

import numpy as np
from evalutils.evalutils import score_detection


def score(gt_coords, pred_coords, dist_thresh):
    """
    Compute TP, FP, FN, and F1-score for all ROIs.
    """

    n_empty_rois = 0
    no_pred_rois = 0
    tps, fns, fps = 0, 0, 0

    for i, (gt, pred) in enumerate(zip(gt_coords, pred_coords)):
        print(f"[ROI {i+1}] Ground Truths: {len(gt)}, Predictions: {len(pred)}")

        if len(pred) == 0:
            fns += len(gt)  # no tp or fp
            no_pred_rois += 1
        elif len(gt) == 0:
            fps += len(pred)  # no tp or fn
            if i == 0:
                n_empty_rois += 1
        else:
            det_score = score_detection(
                ground_truth=gt, predictions=pred, radius=dist_thresh
            )
            tps += det_score.true_positives
            fns += det_score.false_negatives
            fps += det_score.false_positives

        print(f"  → TP: {tps}, FN: {fns}, FP: {fps}")

    print(
        f"\nCompleted {len(gt_coords)} ROIs — Empty GTs: {n_empty_rois}, No Predictions: {no_pred_rois}"
    )

    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def do_prints(gts, preds_list):
    print(f"\n[INFO] Ground Truths for {len(gts)} files")
    print(f"[INFO] Total ROIs during inference: {len(preds_list)}\n")

    for i, gt in enumerate(gts):
        print(f"  GT File {i+1}: {len(gt)} cells")

    print(f"\n[INFO] Predictions for {len(preds_list)} files")
    for i, pr in enumerate(preds_list):
        pred_count = len(pr) if isinstance(pr, list) else int(bool(pr))
        print(f"  Prediction File {i+1}: {pred_count} predictions")


def compute_f1(gts, preds_list, dist_thresh):
    do_prints(gts, preds_list)

    if not preds_list or np.sum([len(pr) for pr in preds_list]) == 0:
        print("[WARN] No predictions found!")
        return 0.0

    f1_score = score(gts, preds_list, dist_thresh)

    print(f"\n[RESULTS] ROIs Processed: {len(gts)}")
    print(f"[RESULTS] F1 Score: {f1_score:.5f}")
    return f1_score
