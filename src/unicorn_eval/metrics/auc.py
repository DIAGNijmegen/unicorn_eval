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
from sklearn.metrics import roc_auc_score


class RocAuc:
    """
    Compute ROC‑AUC from two 1‑D arrays: y_pred (predictions) and y_true (ground truth).

    """

    def __init__(self, ground_truths, predictions):
        predictions = self._coerce_predictions(predictions)

        self.y_pred = np.asarray(predictions, dtype=float)
        self.y_true = np.asarray(ground_truths, dtype=float)

        if self.y_pred.ndim != 1 or self.y_true.ndim != 1:
            raise ValueError("predictions and ground_truths must be 1‑D arrays.")
        if self.y_pred.shape[0] != self.y_true.shape[0]:
            raise ValueError("predictions and ground_truths must be the same length.")

    @staticmethod
    def _coerce_predictions(predictions):
        """
        Convert *predictions* to a plain 1‑D float array.

        If *predictions* is a structured array, the method looks first for a
        field named ``'prediction'``.  If that is absent, the first float field
        encountered is used.

        Returns
        -------
        np.ndarray
            1‑D float array of prediction scores.

        Raises
        ------
        ValueError
            If no suitable float field is found in a structured array.
        """
        arr = np.asarray(predictions)

        # Structured array?  Then extract the relevant float field.
        if arr.dtype.names:
            if "prediction" in arr.dtype.names:
                arr = arr["prediction"]
            else:
                # Fallback: grab the first float‑typed field
                float_field = next(
                    (name for name in arr.dtype.names
                     if np.issubdtype(arr.dtype[name], np.floating)),
                    None
                )
                if float_field is None:
                    raise ValueError(
                        "Structured 'predictions' array must contain at least one float field."
                    )
                arr = arr[float_field]

        return arr

    def compute_auc(self) -> float:
        """Return the ROC‑AUC."""
        return roc_auc_score(self.y_true, self.y_pred)


def compute_roc_auc(ground_truths, predictions):
    auc = RocAuc(ground_truths, predictions).compute_auc()
    print(f"AUC: {auc:.4f}")
    return auc
