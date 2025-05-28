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
    KNN,
    WeightedKNN,
    LogisticRegression,
    LinearProbing,
    MultiLayerPerceptron,
)
from unicorn_eval.adaptors.regression import (
    KNNRegressor,
    WeightedKNNRegressor,
    LinearProbingRegressor,
    MultiLayerPerceptronRegressor,
)

from unicorn_eval.adaptors.detection import DensityMap, PatchNoduleRegressor
from unicorn_eval.adaptors.segmentation import (
    SegmentationUpsampling,
    SegmentationUpsampling3D,
)

__all__ = [
    "KNN",
    "WeightedKNN",
    "LogisticRegression",
    "LinearProbing",
    "MultiLayerPerceptron",
    "KNNRegressor",
    "WeightedKNNRegressor",
    "LinearProbingRegressor",
    "MultiLayerPerceptronRegressor",
    "DensityMap",
    "PatchNoduleRegressor",
    "SegmentationUpsampling",
    "SegmentationUpsampling3D",
]
