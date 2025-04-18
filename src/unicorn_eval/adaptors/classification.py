from typing import Literal, Union, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from unicorn_eval.adaptors.base import BaseAdaptor


def preprocess_features(
    train_feats: np.ndarray,
    test_feats: np.ndarray,
    center: bool = True,
    normalize_feats: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess feature vectors by centering and normalizing, optionally converting to NumPy.

    Args:
        train_feats: Training feature array (N_train, D)
        test_feats: Test feature array (N_test, D)
        center: Whether to subtract mean of training features
        normalize_feats: Whether to apply L2 normalization

    Returns:
        Preprocessed (train_feats, test_feats) as torch.Tensor or np.ndarray
    """
    if center:
        mean_feat = train_feats.mean(dim=0, keepdims=True)
        train_feats = train_feats - mean_feat
        test_feats = test_feats - mean_feat

    if normalize_feats:
        train_feats = train_feats / np.linalg.norm(train_feats, axis=-1, keepdims=True)
        test_feats = test_feats / np.linalg.norm(test_feats, axis=-1, keepdims=True)

    return train_feats, test_feats


class KNN(BaseAdaptor):
    """
    A class to perform K-Nearest Neighbors (KNN) probing for classification or regression tasks.
    This class provides functionality to preprocess features and apply KNN models for
    classification or regression tasks. It supports feature centering and L2 normalization.
    Attributes:
        k (int): Number of neighbors to consider for KNN.
        task_type (Literal["classification", "regression"]): The type of task to perform.
        num_workers (int): Number of parallel jobs for sklearn models. Default is 8.
        center_feats (bool): Whether to subtract the mean from features. Default is False.
        normalize_feats (bool): Whether to L2 normalize features. Default is False.
    Methods:
        fit(train_feats: np.ndarray, train_labels: np.ndarray):
            Fits the KNN model using the provided training features and labels.
        predict(test_feats: np.ndarray) -> np.ndarray:
            Predicts the labels or values for the provided test features.
        preprocess_features(train_feats: np.ndarray, test_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Preprocesses the training and test features by centering and/or normalizing them.
    """
    def __init__(self, train_feats, train_labels, test_feats, k, task_type, num_workers=8, center_feats=False, normalize_feats=False):
        super().__init__(train_feats, train_labels, test_feats)
        self.k = k
        self.task_type = task_type
        self.num_workers = num_workers
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats
        self.model = None

    def fit(self):
        train_feats, _ = preprocess_features(
            self.train_feats, self.test_feats, center=self.center_feats, normalize_feats=self.normalize_feats
        )

        if self.task_type == "classification":
            self.model = KNeighborsClassifier(n_neighbors=self.k, n_jobs=self.num_workers)
        elif self.task_type == "regression":
            self.model = KNeighborsRegressor(n_neighbors=self.k, n_jobs=self.num_workers)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self.model.fit(train_feats, self.train_labels)

    def predict(self) -> np.ndarray:
        _, test_feats = preprocess_features(
            self.train_feats, self.test_feats, center=self.center_feats, normalize_feats=self.normalize_feats
        )

        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before `predict`.")

        return self.model.predict(test_feats)


class WeightedKNN(BaseAdaptor):
    """
    WeightedKNN is a k-Nearest Neighbors (k-NN) based adaptor that supports weighted similarity
    for classification, ordinal classification, and regression tasks. It allows customization of
    distance metrics, feature preprocessing, and output formats.
    Attributes:
        train_feats (np.ndarray): Training feature matrix.
        train_labels (np.ndarray): Labels corresponding to the training features.
        test_feats (np.ndarray): Test feature matrix.
        k (int): Number of nearest neighbors to consider.
        task_type (str): Type of task, one of ["classification", "ordinal-classification", "regression"].
        metric (str or callable): Similarity metric to use. Options are "cosine", "euclidean", or a callable function.
        center_feats (bool): Whether to center the features during preprocessing.
        normalize_feats (bool): Whether to normalize the features during preprocessing.
        return_probabilities (bool): Whether to return class probabilities for classification tasks.
        class_values (np.ndarray or None): Array of possible class values for regression tasks.
    Methods:
        __init__(train_feats, train_labels, test_feats, k, task_type, metric="cosine", center_feats=False, normalize_feats=False, return_probabilities=False, class_values=None):
            Initializes the WeightedKNN with the given parameters.
        fit():
            Preprocesses the features and sets up the similarity function and class-related attributes
            based on the task type.
        predict() -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
            Predicts the output for the test features based on the k-nearest neighbors. For classification
            tasks, it can optionally return class probabilities.
    """
    def __init__(self, train_feats, train_labels, test_feats, k, task_type, metric="cosine", center_feats=False, normalize_feats=False, return_probabilities=False, class_values=None):
        super().__init__(train_feats, train_labels, test_feats)
        self.k = k
        self.task_type = task_type
        self.metric = metric
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats
        self.return_probabilities = return_probabilities
        self.class_values = class_values
        self.train_feats = None
        self.test_feats = None
        self.similarity_fn = None
        self.unique_classes = None
        self.class_to_idx = None
        self.num_classes = None

        assert not (
            task_type == "regression" and return_probabilities
        ), "Cannot return probabilities for regression."


    def fit(self):
        self.train_feats, self.test_feats = preprocess_features(
            self.train_feats, self.test_feats, center=self.center_feats, normalize_feats=self.normalize_feats
        )

        # define similarity function
        if callable(self.metric):
            self.similarity_fn = self.metric
        elif self.metric == "cosine":
            self.similarity_fn = lambda x, y: cosine_similarity(x, y)
        elif self.metric == "euclidean":
            self.similarity_fn = lambda x, y: 1.0 / (euclidean_distances(x, y) + 1e-8)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        if self.task_type in ["classification", "ordinal-classification"]:
            self.unique_classes = np.unique(self.train_labels)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
            self.num_classes = len(self.unique_classes)

    def predict(self) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if self.train_feats is None or self.test_feats is None or self.similarity_fn is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before `predict`.")

        predictions = []
        all_probs = []

        for test_point in self.test_feats:
            sim = self.similarity_fn(test_point.reshape(1, -1), self.train_feats).flatten()
            k_indices = np.argsort(-sim)[:self.k]
            k_labels = self.train_labels[k_indices]
            k_similarities = sim[k_indices]

            if self.task_type == "regression":
                weighted_avg = np.sum(k_labels * k_similarities) / (np.sum(k_similarities) + 1e-8)
                if self.class_values is not None:
                    diffs = np.abs(self.class_values - weighted_avg)
                    class_label = self.class_values[np.argmin(diffs)]
                    predictions.append(class_label)
                else:
                    predictions.append(weighted_avg)

            elif self.task_type in ["classification", "ordinal-classification"]:
                class_weights = np.zeros(self.num_classes)
                for label, sim in zip(k_labels, k_similarities):
                    class_weights[self.class_to_idx[label]] += sim

                class_probs = class_weights / (np.sum(class_weights) + 1e-8)
                all_probs.append(class_probs)

                if self.task_type == "ordinal-classification":
                    expected_val = np.dot(class_probs, self.unique_classes)
                    predicted_class = int(np.round(expected_val))
                else:
                    predicted_class = self.unique_classes[np.argmax(class_probs)]

                predictions.append(predicted_class)

        predictions = np.array(predictions)
        if self.return_probabilities and self.task_type in ["classification", "ordinal-classification"]:
            return predictions, np.vstack(all_probs)
        return predictions


class LogisticRegression(BaseAdaptor):
    """
    An adaptor for logistic regression that extends the BaseAdaptor class. This class
    provides functionality to train a logistic regression model and make predictions
    using the provided training and testing features.
    Attributes:
        train_feats (np.ndarray): The feature matrix for training the model.
        train_labels (np.ndarray): The labels corresponding to the training features.
        test_feats (np.ndarray): The feature matrix for testing the model.
        max_iter (int): The maximum number of iterations for the solver to converge. Default is 1000.
        C (float): Inverse of regularization strength; smaller values specify stronger regularization. Default is 1.0.
        solver (str): The algorithm to use in the optimization problem. Default is "lbfgs".
    Methods:
        fit():
            Trains the logistic regression model using the training features and labels.
        predict() -> np.ndarray:
            Predicts the labels for the test features using the trained model.
    """
    def __init__(self, train_feats, train_labels, test_feats, max_iter=1000, C=1.0, solver="lbfgs"):
        super().__init__(train_feats, train_labels, test_feats)
        self.max_iter = max_iter
        self.C = C
        self.solver = solver

    def fit(self):
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver=self.solver, random_state=0)
        self.model.fit(self.train_feats, self.train_labels)

    def predict(self) -> np.ndarray:
        return self.model.predict(self.test_feats)


class LPClassifier(nn.Module):
    """
    A simple linear classifier.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearProbing(BaseAdaptor):
    """
    A class for performing linear probing on features for classification or regression tasks.
    Linear probing involves training a simple linear model on top of pre-extracted features
    to evaluate their quality for a specific task.
    Attributes:
        train_feats (np.ndarray): The training feature matrix of shape (n_samples, n_features).
        train_labels (np.ndarray): The training labels corresponding to the training features.
        test_feats (np.ndarray): The test feature matrix of shape (n_samples, n_features).
        task_type (str): The type of task, either "classification" or "regression".
        num_epochs (int): The number of epochs for training the linear model. Default is 100.
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.
    Methods:
        fit():
            Trains a linear model on the training features and labels using the specified task type.
        predict() -> np.ndarray:
            Predicts the labels for the test features using the trained model.
    """
    def __init__(self, train_feats, train_labels, test_feats, task_type, num_epochs=100, learning_rate=0.001):
        super().__init__(train_feats, train_labels, test_feats)
        self.task_type = task_type
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self):
        input_dim = self.train_feats.shape[1]
        if self.task_type == "regression":
            self.num_classes = 1
            self.criterion = nn.MSELoss()
        elif self.task_type == "classification":
            self.num_classes = len(np.unique(self.train_labels))
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_feats = torch.tensor(self.train_feats, dtype=torch.float32).to(self.device)
        self.train_labels = torch.tensor(self.train_labels, dtype=torch.long).to(self.device)
        self.test_feats = torch.tensor(self.test_feats, dtype=torch.float32).to(self.device)

        self.model = LPClassifier(input_dim, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.train_feats)
            loss = self.criterion(outputs, self.train_labels)
            loss.backward()
            self.optimizer.step()

    def predict(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.test_feats)
            _, test_preds = torch.max(test_outputs, 1)
        return test_preds.cpu().numpy()


class MLPClassifier(nn.Module):
    """
    A simple MLP classifier with one hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MLP(BaseAdaptor):
    """
    A PyTorch-based Multi-Layer Perceptron (MLP) adaptor for classification and regression tasks.
    Attributes:
        train_feats (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        train_labels (np.ndarray): Training labels corresponding to the training features.
        test_feats (np.ndarray): Test feature matrix of shape (n_samples, n_features).
        task_type (str): Type of task, either "classification" or "regression".
        hidden_dim (int): Number of hidden units in the MLP. Default is 64.
        num_epochs (int): Number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    Methods:
        fit():
            Trains the MLP model using the provided training data.
        predict() -> np.ndarray:
            Generates predictions for the test data using the trained model.
    """
    def __init__(self, train_feats, train_labels, test_feats, task_type, hidden_dim=64, num_epochs=100, learning_rate=0.001):
        super().__init__(train_feats, train_labels, test_feats)
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self):
        input_dim = self.train_feats.shape[1]
        if self.task_type == "regression":
            self.num_classes = 1
            self.criterion = nn.MSELoss()
        elif self.task_type == "classification":
            self.num_classes = len(np.unique(self.train_labels))
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_feats = torch.tensor(self.train_feats, dtype=torch.float32).to(self.device)
        self.train_labels = torch.tensor(self.train_labels, dtype=torch.long).to(self.device)
        self.test_feats = torch.tensor(self.test_feats, dtype=torch.float32).to(self.device)

        self.model = MLPClassifier(input_dim, self.hidden_dim, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.train_feats)
            loss = self.criterion(outputs, self.train_labels)
            loss.backward()
            self.optimizer.step()

    def predict(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.test_feats)
            _, test_preds = torch.max(test_outputs, 1)
        return test_preds.cpu().numpy()