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


class Adaptor(ABC):
    """
    Abstract base class for all adaptors.
    This class provides a blueprint for implementing adaptors that can handle both case-level and patch-level tasks.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Abstract method to fit the model using the few-shot data.
        The implementation should use the provided arguments to fit the model.
        """

    @abstractmethod
    def predict(self, test_ids: list[str]) -> np.ndarray:
        """
        Abstract method to make predictions using the test data.
        Returns:
            np.ndarray: Predictions for the test set.
        """


class CaseLevelTaskAdaptor(Adaptor):
    """
    Abstract base class for case-level tasks such as classification or regression.
    This class provides a blueprint for implementing adaptors that operate on a case level,
    where each case is represented by its features and corresponding labels.
    """

    def __init__(self):
        """
        Initializes the adaptor instance.

        This constructor can be used to store hyperparameters (such as number of epochs or learning rate, if relevant) for later use.
        If there are no hyperparameters, this method can remain empty.
        """

    @abstractmethod
    def fit(
        self,
        shot_features: np.ndarray,
        shot_labels: np.ndarray,
        shot_extra_labels: np.ndarray | None = None,
    ) -> None:
        """
        Abstract method to fit the model using the few-shot data.
        The implementation should use `shot_features`, `shot_labels` (and potentially `shot_extra_labels`) to fit the model.
        """

    @abstractmethod
    def predict(self, test_ids: list[str]) -> np.ndarray:
        """
        Abstract method to make predictions using the test data.
        Returns:
            np.ndarray: Predictions for the test set based on `test_features`.
        """


class PatchLevelTaskAdaptor(Adaptor):
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

    def __init__(self):
        """
        Initializes the adaptor instance.

        This constructor can be used to store hyperparameters (such as number of epochs or learning rate, if relevant) for later use.
        If there are no hyperparameters, this method can remain empty.
        """
        pass

    @abstractmethod
    def fit(
        self,
        shot_features: np.ndarray,
        shot_coordinates: np.ndarray,
        shot_labels: np.ndarray,
        shot_extra_labels: np.ndarray | None = None,
    ) -> None:
        """
        Abstract method to fit the model using the few-shot data.
        The implementation should use `shot_features`, `shot_coordinates`, `shot_labels` (and potentially `shot_extra_labels`) to fit the model.
        """

    @abstractmethod
    def predict(self, test_ids: list[str]) -> np.ndarray:
        """
        Abstract method to make predictions using the test data.
        Returns:
            np.ndarray: Predictions for the test set based on `test_features` and `test_coordinates`.
        """
