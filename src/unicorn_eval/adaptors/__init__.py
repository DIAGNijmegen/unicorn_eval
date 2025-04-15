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

from unicorn_eval.adaptors.classification import (
    knn_probing,
    weighted_knn_probing,
    logistic_regression,
    linear_probing,
    mlp,
)
from unicorn_eval.adaptors.detection import density_map
from unicorn_eval.adaptors.segmentation import segmentation_upsampling

__all__ = [
    "knn_probing",
    "logistic_regression",
    "weighted_knn_probing",
    "linear_probing",
    "mlp",
    "density_map",
    "segmentation_upsampling",
]
