from abc import ABC, abstractmethod
import numpy as np


class BaseAdaptor(ABC):
    """
    Base class for coarse-grained tasks like classification or regression.
    Expects train_feats, train_labels, and test_feats.
    """

    def __init__(self, train_feats: np.ndarray, train_labels: np.ndarray, test_feats: np.ndarray):
        self.train_feats = train_feats
        self.train_labels = train_labels
        self.test_feats = test_feats

    @abstractmethod
    def fit(self):
        """
        Fit the model using train_feats and train_labels.
        """
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Predict using test_feats.
        Returns:
            np.ndarray: Predictions for the test set.
        """
        pass


class DenseAdaptor(BaseAdaptor):
    """
    Base class for dense prediction tasks like detection or segmentation.
    Expects train_feats, train_labels, test_feats, train_coordinates, and test_coordinates.
    """

    def __init__(
        self,
        train_feats: np.ndarray,
        train_labels: np.ndarray,
        test_feats: np.ndarray,
        train_coordinates: np.ndarray,
        test_coordinates: np.ndarray,
    ):
        super().__init__(train_feats, train_labels, test_feats)
        self.train_coordinates = train_coordinates
        self.test_coordinates = test_coordinates

    @abstractmethod
    def fit(self):
        """
        Fit the model using train_feats, train_labels, and train_coordinates.
        """
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Predict using test_feats and test_coordinates.
        Returns:
            np.ndarray: Predictions for the test set.
        """
        pass