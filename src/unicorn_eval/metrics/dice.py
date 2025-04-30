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
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix


def list_not_in(lst1, lst2):
    """Returns values in lst1 not in lst2"""
    return [v for v in lst1 if v not in lst2]


class CmScorer:
    def __init__(
        self,
        class_map,
        incremental=False,
        ignore_gt_zeros=True,
        gt_remap={},
        pred_remap={},
        remap_inplace=False,
    ):
        """
        class_map: {label: name}
        incremental: accumulate metrics across calls
        ignore_gt_zeros: ignore 0s in the ground truth
        gt_remap / pred_remap: remap values in GT or prediction
        """
        self._ignore_gt_zeros = ignore_gt_zeros
        self._incremental = incremental
        self.gt_remap = gt_remap
        self.pred_remap = pred_remap
        self.remap_inplace = remap_inplace
        self.class_map = class_map
        self.reset()

    def reset(self):
        self.cm = None

    def _remap(self, arr, old_new_map):
        if old_new_map.values() in old_new_map.keys() or not self.remap_inplace:
            arr_new = arr.copy()
            for old_val, new_val in old_new_map.items():
                arr_new[arr == old_val] = new_val
            return arr_new
        else:
            for old_val, new_val in old_new_map.items():
                arr[arr == old_val] = new_val
            return arr

    def __call__(self, gt, pred):
        if not self._incremental:
            self.reset()
        gt = self._remap(gt, self.gt_remap)
        pred = self._remap(pred, self.pred_remap)

        if self._ignore_gt_zeros:
            mask = gt != 0
            gt = gt[mask]
            pred = pred[mask]

        class_labels = list(sorted(self.class_map.keys()))
        class_names = [self.class_map[k] for k in class_labels]

        pred_labels = sorted(list(set(pred)))
        surplus = list_not_in(pred_labels, class_labels)
        if surplus:
            raise ValueError(f"Unknown prediction labels: {surplus}")

        cm_arr = confusion_matrix(gt, pred, labels=class_labels)
        matrix = {
            class_names[i]: {
                class_names[j]: int(cm_arr[i, j]) for j in range(len(class_names))
            }
            for i in range(len(class_names))
        }

        cm = ConfusionMatrix(matrix=matrix)
        if self.cm is None:
            self.cm = cm
        else:
            self.cm = self.cm.combine(cm)
        return self._get_score(cm)

    def _get_score(self, cm):
        stats = cm.class_stat.get("F1", {})
        stats["cm"] = cm.to_array()
        stats["classes"] = cm.classes
        return stats

    def get_score(self):
        return self._get_score(self.cm)


class TigerSegmScorer(CmScorer):
    def __init__(self, incremental=True):
        # gt_remap = {4: 3, 5: 3, 6: 2, 7: 3}
        pred_remap = {k: 3 for k in range(256)}
        pred_remap.update({1: 1, 2: 2})
        class_map = {1: "Tumor", 2: "Stroma", 3: "Other"}
        super().__init__(
            class_map=class_map,
            incremental=incremental,
            pred_remap=pred_remap,
            ignore_gt_zeros=True,
        )


def compute_dice_score(gts, preds):
    scorer = TigerSegmScorer(incremental=True)

    for i, gt in enumerate(gts):
        pred = preds[i]
        scorer(gt.astype(np.uint8), pred.astype(np.uint8))

    final_score = scorer.get_score()
    avg_tumor_stroma_dice = (final_score["Tumor"] + final_score["Stroma"]) / 2
    print(f"Score: {final_score}")
    print(f"Avg Tumor-Stroma Dice: {avg_tumor_stroma_dice:.4f}")
    return avg_tumor_stroma_dice
