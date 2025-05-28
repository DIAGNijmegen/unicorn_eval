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

from abc import ABC, abstractmethod
import numpy as np


class CaseLevelTaskAdaptor(ABC):
    """
    Abstract base class for case-level tasks such as classification or regression.
    This class provides a blueprint for implementing adaptors that operate on a case level,
    where each case is represented by its features and corresponding labels.

    Attributes:
        shot_features (np.ndarray): Feature matrix for the few-shots.
        shot_labels (np.ndarray): Labels corresponding to the few-shots.
        test_features (np.ndarray): Feature matrix for the test set.
        shot_extra_labels (np.ndarray, optional): Additional labels for the few-shots, if applicable. Defaults to None.
    """

    def __init__(
        self,
        shot_features: np.ndarray,
        shot_labels: np.ndarray,
        test_features: np.ndarray,
        shot_extra_labels: np.ndarray = None,
    ):
        """
        Initializes the CaseLevelTaskAdaptor with few-shot and test data.

        Args:
            shot_features (np.ndarray): Feature matrix for the few-shots.
            shot_labels (np.ndarray): Labels corresponding to the few-shots.
            test_features (np.ndarray): Feature matrix for the test set.
            shot_extra_labels (np.ndarray, optional): Additional labels for the few-shots, if applicable.
        """
        self.shot_features = shot_features
        self.shot_labels = shot_labels
        self.test_features = test_features
        self.shot_extra_labels = shot_extra_labels

    @abstractmethod
    def fit(self):
        """
        Abstract method to fit the model using the few-shot data.
        The implementation should use `shot_features` and `shot_labels` to fit the model.
        """
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Abstract method to make predictions using the test data.
        Returns:
            np.ndarray: Predictions for the test set based on `test_features`.
        """
        pass


class PatchLevelTaskAdaptor(ABC):
    """
    Abstract base class for dense prediction tasks such as detection or segmentation.
    This class provides a blueprint for implementing adaptors that operate on a patch level,
    where each case is represented by its patch features and coordinates, and corresponding labels.

    Attributes:
        shot_features (np.ndarray): Feature matrix for the few-shots.
        shot_labels (np.ndarray): Labels corresponding to the few-shots.
        shot_coordinates (np.ndarray): Spatial coordinates of the patches associated to each few-shot.
        test_features (np.ndarray): Feature matrix for the test set.
        test_coordinates (np.ndarray): Spatial coordinates of the patches associated to each test case.
        shot_extra_labels (np.ndarray, optional): Additional labels for the few-shots, if applicable. Defaults to None.
    """

    def __init__(
        self,
        shot_features: np.ndarray,
        shot_labels: np.ndarray,
        shot_coordinates: np.ndarray,
        test_features: np.ndarray,
        test_coordinates: np.ndarray,
        shot_extra_labels: np.ndarray = None,
    ):
        """
        Initializes the base adaptor with the provided features, labels, and coordinates.

        Args:
            shot_features (np.ndarray): Feature matrix for the few-shots.
            shot_labels (np.ndarray): Labels corresponding to the few-shots.
            shot_coordinates (np.ndarray): Spatial coordinates of the patches associated to each few-shot.
            test_features (np.ndarray): Feature matrix for the test set.
            test_coordinates (np.ndarray): Spatial coordinates of the patches associated to each test case.
            shot_extra_labels (np.ndarray, optional): Additional labels for the few-shots, if applicable. Defaults to None.
        """
        self.shot_features = shot_features
        self.shot_labels = shot_labels
        self.shot_coordinates = shot_coordinates
        self.test_features = test_features
        self.test_coordinates = test_coordinates
        self.shot_extra_labels = shot_extra_labels

    @abstractmethod
    def fit(self):
        """
        Abstract method to fit the model using the few-shot data.
        The implementation should use `shot_features`, `shot_coordinates` and `shot_labels` to fit the model.
        """
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Abstract method to make predictions using the test data.
        Returns:
            np.ndarray: Predictions for the test set based on `test_features`.
        """
        pass
