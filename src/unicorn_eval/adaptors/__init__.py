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

from unicorn_eval.adaptors.classification import (KNN, LinearProbing,
                                                  LogisticRegression,
                                                  MultiLayerPerceptron,
                                                  WeightedKNN)
from unicorn_eval.adaptors.detection import (ConvDetector, DensityMap,
                                             PatchNoduleRegressor)
from unicorn_eval.adaptors.regression import (KNNRegressor,
                                              LinearProbingRegressor,
                                              MultiLayerPerceptronRegressor,
                                              WeightedKNNRegressor)
from unicorn_eval.adaptors.segmentation import (ConvSegmentation3D,
                                                SegmentationUpsampling,
                                                SegmentationUpsampling3D)
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v1 import \
    LinearUpsampleConv3D_V1
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2 import \
    LinearUpsampleConv3D_V2

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
    "ConvDetector",
    "PatchNoduleRegressor",
    "SegmentationUpsampling",
    "SegmentationUpsampling3D",
    "ConvSegmentation3D",
    "LinearUpsampleConv3D_V1",
    "LinearUpsampleConv3D_V2",
]
