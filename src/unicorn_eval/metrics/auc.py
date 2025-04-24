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

from sklearn.metrics import roc_auc_score


class ModelEvaluator:
    def __init__(self, predictions, ground_truths):
        # predictions: list of tuples (scan_id, predicted_value)
        # ground_truths: dict {scan_id: true_label}
        self.predictions = predictions
        self.ground_truths = ground_truths

    def compute_auc(self):
        y_pred = []
        y_true = []

        for scan_id, pred_value in self.predictions:
            if scan_id in self.ground_truths:
                y_pred.append(pred_value)
                y_true.append(self.ground_truths[scan_id])

        if not y_true:
            raise ValueError("No valid labels found to compute AUC.")

        auc_score = roc_auc_score(y_true, y_pred)
        return float(f"{auc_score:.4f}")
