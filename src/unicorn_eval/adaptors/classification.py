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

import logging

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from unicorn_eval.adaptors.base import CaseLevelTaskAdaptor
from unicorn_eval.io import INPUT_DIRECTORY, process, read_inputs


def preprocess_features(
    features: np.ndarray,
    center: bool = True,
    mean: np.ndarray = None,
    normalize_features: bool = True,
) -> np.ndarray:
    """
    Preprocess feature vectors by centering and normalizing, optionally converting to NumPy.

    Args:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        center: Whether to subtract mean of few-shot features
        normalize_features: Whether to apply L2 normalization

    Returns:
        Preprocessed (shot_features, test_features) as torch.Tensor or np.ndarray
    """
    if center:
        features = features - mean

    if normalize_features:
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)

    return features


class KNN(CaseLevelTaskAdaptor):
    """
    A class to perform K-Nearest Neighbors (KNN) probing for classification tasks.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        k (int): Number of neighbors to consider for KNN.
        num_workers (int): Number of parallel jobs for sklearn models. Default is 8.
        center_features (bool): Whether to subtract the mean from features. Default is False.
        normalize_features (bool): Whether to L2 normalize features. Default is False.
    Methods:
        fit():
            Fits the KNN model using the provided few-shot features and labels.
        predict() -> np.ndarray:
            Predicts the labels or values for the provided test features.
    """

    def __init__(
        self,
        k,
        num_workers=8,
        center_features=False,
        normalize_features=False,
        return_probabilities=False,
    ):
        self.k = k
        self.num_workers = num_workers
        self.center_features = center_features
        self.normalize_features = normalize_features
        self.return_probabilities = return_probabilities
        self.model = None

    def fit(self, shot_features, shot_labels, **kwargs):
        self.mean_feature = shot_features.mean(axis=0, keepdims=True)
        processed_shot_features = preprocess_features(
            shot_features,
            center=self.center_features,
            mean=self.mean_feature,
            normalize_features=self.normalize_features,
        )

        self.model = KNeighborsClassifier(n_neighbors=self.k, n_jobs=self.num_workers)
        self.model.fit(processed_shot_features, shot_labels)

    def predict(self, test_cases) -> np.ndarray:
        predictions = []
        for case_name in test_cases:
            test_input = process(
                read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
            )
            test_feature = test_input["embeddings"]
            processed_test_feature = preprocess_features(
                test_feature,
                mean=self.mean_feature,
                center=self.center_features,
                normalize_features=self.normalize_features,
            )

            if self.model is None:
                raise ValueError(
                    "Model has not been fitted yet. Call `fit` before `predict`."
                )

            if self.return_probabilities:
                prediction = self.model.predict_proba(processed_test_feature)
                predictions.append(prediction)
            else:
                prediction = self.model.predict(processed_test_feature)
                predictions.append(prediction)

        return np.array(predictions).squeeze()


class WeightedKNN(CaseLevelTaskAdaptor):
    """
    WeightedKNN is a k-Nearest Neighbors (k-NN) based adaptor that supports weighted similarity
    for classification tasks. It allows customization of
    distance metrics.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        k (int): Number of nearest neighbors to consider.
        metric (str or callable): Similarity metric to use. Options are "cosine", "euclidean", or a callable function.
        center_features (bool): Whether to center the features during preprocessing.
        normalize_features (bool): Whether to normalize the features during preprocessing.
        return_probabilities (bool): Whether to return class probabilities instead of predictions.
    Methods:
        fit():
            Preprocesses the features and sets up the similarity function and class-related attributes
            based on the task type.
        predict() -> np.ndarray | tuple[np.ndarray, np.ndarray]:
            Predicts the output for the test features based on the k-nearest neighbors.
    """

    def __init__(
        self,
        k,
        metric="cosine",
        center_features=False,
        normalize_features=False,
        return_probabilities=False,
    ):
        self.k = k
        self.metric = metric
        self.center_features = center_features
        self.normalize_features = normalize_features
        self.return_probabilities = return_probabilities
        self.similarity_fn = None
        self.unique_classes = None
        self.class_to_idx = None
        self.num_classes = None

    def fit(self, shot_features, shot_labels, **kwargs):
        self.mean_feature = shot_features.mean(axis=0, keepdims=True)
        self.shot_features = preprocess_features(
            shot_features,
            center=self.center_features,
            mean=self.mean_feature,
            normalize_features=self.normalize_features,
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

        self.shot_labels = shot_labels
        self.unique_classes = np.unique(shot_labels)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        self.num_classes = int(shot_labels.max()) + 1

    def predict(self, test_cases) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        predictions, probabilities = [], []
        for case_name in test_cases:
            test_input = process(
                read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
            )
            test_feature = test_input["embeddings"]
            processed_test_feature = preprocess_features(
                test_feature,
                mean=self.mean_feature,
                center=self.center_features,
                normalize_features=self.normalize_features,
            )

            if self.shot_features is None or self.similarity_fn is None:
                raise ValueError(
                    "Model has not been fitted yet. Call `fit` before `predict`."
                )

            similarities = self.similarity_fn(
                processed_test_feature.reshape(1, -1), self.shot_features
            ).flatten()
            k_indices = np.argsort(-similarities)[: self.k]
            k_labels = self.shot_labels[k_indices]
            k_similarities = similarities[k_indices]

            class_weights = np.zeros(self.num_classes)
            for label, similarity in zip(k_labels, k_similarities):
                class_weights[self.class_to_idx[label]] += similarity

            class_probabilities = class_weights / (np.sum(class_weights) + 1e-8)
            probabilities.append(class_probabilities)

            predicted_class = self.unique_classes[np.argmax(class_probabilities)]
            predictions.append(predicted_class)

        if self.return_probabilities:
            return np.array(probabilities).squeeze()
        else:
            return np.array(predictions).squeeze()


class LogisticRegression(CaseLevelTaskAdaptor):
    """
    An adaptor for logistic regression that extends the CaseLevelTaskAdaptor class. This class
    provides functionality to train a logistic regression model and make predictions
    using the provided few-shot and testing features.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        max_iterations (int): The maximum number of iterations for the solver to converge. Default is 1000.
        C (float): Inverse of regularization strength; smaller values specify stronger regularization. Default is 1.0.
        solver (str): The algorithm to use in the optimization problem. Default is "lbfgs".
        return_probabilities (bool): Whether to return class probabilities instead of predictions.
    Methods:
        fit():
            Trains the logistic regression model using the few-shot features and labels.
        predict() -> np.ndarray:
            Predicts the labels for the test features using the trained model.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
        return_probabilities: bool = False,
        seed: int = 0,
    ):
        self.max_iterations = max_iterations
        self.C = C
        self.solver = solver
        self.return_probabilities = return_probabilities
        self.seed = seed

    def fit(self, shot_features, shot_labels, **kwargs):
        self.model = sklearn.linear_model.LogisticRegression(
            C=self.C, max_iter=self.max_iterations, solver=self.solver, random_state=self.seed
        )
        self.model.fit(shot_features, shot_labels)

    def predict(self, test_cases) -> np.ndarray:
        predictions = []
        for case_name in test_cases:
            test_input = process(
                read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
            )
            test_feature = test_input["embeddings"]

            if self.return_probabilities:
                prediction = self.model.predict_proba(test_feature)
                predictions.append(prediction)
            else:
                prediction = self.model.predict(test_feature)
                predictions.append(prediction)

        return np.array(predictions).squeeze()


class LinearClassifier(nn.Module):
    """
    A simple linear classifier.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearProbing(CaseLevelTaskAdaptor):
    """
    A class for performing linear probing on features for classification tasks.
    Linear probing involves training a simple linear model on top of pre-extracted features
    to evaluate their quality for a specific task.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        num_epochs (int): The number of epochs for training the linear model. Default is 100.
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
        shot_extra_labels (np.ndarray): Optional additional labels for training.
        return_probabilities (bool): Whether to return class probabilities instead of predictions.
    Methods:
        fit():
            Trains a linear model on the few-shot features and labels using the specified task type.
        predict() -> np.ndarray:
            Predicts the labels for the test features using the trained model.
    """

    def __init__(
        self,
        num_epochs=100,
        learning_rate=0.001,
        patience=10,
        return_probabilities=False,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.return_probabilities = return_probabilities

    def fit(self, shot_features, shot_labels, **kwargs):
        input_dim = shot_features.shape[1]
        self.num_classes = int(shot_labels.max()) + 1
        self.criterion = nn.CrossEntropyLoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        shot_features = torch.tensor(shot_features, dtype=torch.float32).to(self.device)
        shot_labels = torch.tensor(shot_labels, dtype=torch.long).to(self.device)

        self.model = LinearClassifier(input_dim, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logging.info(
            f"Starting training on {self.device} with {total_params:,} trainable parameters."
        )
        logging.info(str(self.model))

        best_loss = float("inf")
        best_epoch = 0
        best_state = self.model.state_dict()
        for epoch in tqdm.tqdm(
            range(self.num_epochs), desc="Training", unit="epoch", leave=True
        ):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(shot_features)
            loss = self.criterion(logits, shot_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss = loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = self.model.state_dict()
            elif epoch - best_epoch > self.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

            logging.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {loss.item():.4f}")

        self.model.load_state_dict(best_state)
        logging.info(
            f"Restored best model from epoch {best_epoch+1} with loss {best_loss:.4f}"
        )

    def predict(self, test_cases) -> np.ndarray:
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for case_name in test_cases:
                test_input = process(
                    read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
                )

                test_feature = test_input["embeddings"]
                test_features = torch.tensor(test_feature, dtype=torch.float32).to(
                    self.device
                )

                logits = self.model(test_features)
                if self.return_probabilities:
                    prediction = torch.softmax(logits, dim=0)
                    predictions.append(prediction.cpu().numpy())
                else:
                    _, prediction = torch.max(logits, dim=0)
                    predictions.append(prediction.cpu().numpy())

        return np.array(predictions).squeeze()


class MLPClassifier(nn.Module):
    """
    A simple MLP classifier with a tunable number of hidden layers.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiLayerPerceptron(CaseLevelTaskAdaptor):
    """
    A PyTorch-based MLP adaptor for classification tasks.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        hidden_dim (int): Number of hidden units in the model. Default is 256.
        num_layers (int): Number of hidden layers in the MLP. Default is 3.
        num_epochs (int): Number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
        shot_extra_labels (np.ndarray): Optional additional labels for training.
        return_probabilities (bool): Whether to return class probabilities instead of predictions.
    Methods:
        fit():
            Fits the model using the provided few-shot data.
        predict() -> np.ndarray:
            Generates predictions for the test data using the fitted model.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_layers=3,
        num_epochs=100,
        learning_rate=0.001,
        patience=10,
        return_probabilities=False,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.return_probabilities = return_probabilities

    def fit(self, shot_features, shot_labels, **kwargs):
        input_dim = shot_features.shape[1]
        self.num_classes = int(shot_labels.max()) + 1
        self.criterion = nn.CrossEntropyLoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        shot_features = torch.tensor(shot_features, dtype=torch.float32).to(self.device)
        shot_labels = torch.tensor(shot_labels, dtype=torch.long).to(self.device)

        self.model = MLPClassifier(
            input_dim, self.hidden_dim, self.num_classes, self.num_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logging.info(
            f"Starting training on {self.device} with {total_params:,} trainable parameters."
        )
        logging.info(str(self.model))

        best_loss = float("inf")
        best_epoch = 0
        best_state = self.model.state_dict()
        for epoch in tqdm.tqdm(
            range(self.num_epochs), desc="Training", unit="epoch", leave=True
        ):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(shot_features)
            loss = self.criterion(logits, shot_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss = loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = self.model.state_dict()
            elif epoch - best_epoch > self.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

            logging.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}")

        self.model.load_state_dict(best_state)
        logging.info(
            f"Restored best model from epoch {best_epoch+1} with loss {best_loss:.4f}"
        )

    def predict(self, test_cases) -> np.ndarray:
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for case_name in test_cases:
                test_input = process(
                    read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
                )
                test_feature = test_input["embeddings"]
                test_features = torch.tensor(test_feature, dtype=torch.float32).to(
                    self.device
                )
                logits = self.model(test_features)
                if self.return_probabilities:
                    prediction = torch.softmax(logits, dim=0)
                    predictions.append(prediction.cpu().numpy())
                else:
                    _, prediction = torch.max(logits, 0)
                    predictions.append(prediction.cpu().numpy())

        return np.array(predictions).squeeze()
