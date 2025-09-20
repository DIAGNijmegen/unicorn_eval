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
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import KNeighborsRegressor

from unicorn_eval.adaptors.base import CaseLevelTaskAdaptor
from unicorn_eval.adaptors.components import NLLSurvLoss
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


class KNNRegressor(CaseLevelTaskAdaptor):
    """
    A class to perform K-Nearest Neighbors (KNN) probing for regression tasks.
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
    ):
        self.k = k
        self.num_workers = num_workers
        self.center_features = center_features
        self.normalize_features = normalize_features
        self.model = None

    def fit(self, shot_features, shot_labels, **kwargs):
        self.mean_feature = shot_features.mean(axis=0, keepdims=True)
        shot_features, _ = preprocess_features(
            shot_features,
            center=self.center_features,
            mean=self.mean_feature,
            normalize_features=self.normalize_features,
        )

        self.model = KNeighborsRegressor(n_neighbors=self.k, n_jobs=self.num_workers)
        self.model.fit(shot_features, shot_labels)

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

            prediction = self.model.predict(processed_test_feature)
            predictions.append(prediction)

        return np.array(predictions).squeeze()


class WeightedKNNRegressor(CaseLevelTaskAdaptor):
    """
    WeightedKNN is a k-Nearest Neighbors (k-NN) based adaptor that supports weighted similarity
    for regression tasks. It allows customization of distance metrics.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        k (int): Number of nearest neighbors to consider.
        metric (str or callable): Similarity metric to use. Options are "cosine", "euclidean", or a callable function.
        center_features (bool): Whether to center the features during preprocessing.
        normalize_features (bool): Whether to normalize the features during preprocessing.
        class_values (np.ndarray or None): Array of possible class values for regression tasks.
    Methods:
        fit():
            Preprocesses the features and sets up the similarity function and class-related attributes
            based on the task type.
        predict() -> np.ndarray
            Predicts the output for the test features based on the k-nearest neighbors.
    """

    def __init__(
        self,
        k,
        metric="cosine",
        center_features=False,
        normalize_features=False,
        class_values=None,
    ):
        self.k = k
        self.metric = metric
        self.center_features = center_features
        self.normalize_features = normalize_features
        self.class_values = class_values
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

        self.shot_labels = shot_labels

        # define similarity function
        if callable(self.metric):
            self.similarity_fn = self.metric
        elif self.metric == "cosine":
            self.similarity_fn = lambda x, y: cosine_similarity(x, y)
        elif self.metric == "euclidean":
            self.similarity_fn = lambda x, y: 1.0 / (euclidean_distances(x, y) + 1e-8)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

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

            weighted_avg = np.sum(k_labels * k_similarities) / (
                np.sum(k_similarities) + 1e-8
            )
            if self.class_values is not None:
                diffs = np.abs(self.class_values - weighted_avg)
                class_label = self.class_values[np.argmin(diffs)]
                predictions.append(class_label)
            else:
                predictions.append(weighted_avg)

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


class LinearProbingRegressor(CaseLevelTaskAdaptor):
    """
    A class for performing linear probing on features for regression tasks.
    Linear probing involves training a simple linear model on top of pre-extracted features
    to evaluate their quality for a specific task.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        survival (bool): Whether the task is survival analysis. Default is False.
        num_epochs (int): The number of epochs for training the linear model. Default is 100.
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
        shot_extra_labels (np.ndarray): Optional additional labels for training, used in survival analysis.
    Methods:
        fit():
            Trains a linear model on the few-shot features and labels using the specified task type.
        predict() -> np.ndarray:
            Predicts the labels for the test features using the trained model.
    """

    def __init__(
        self,
        survival=False,
        num_epochs=100,
        learning_rate=0.001,
        patience=10,
    ):
        self.survival = survival
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience

    def fit(self, shot_features, shot_labels, shot_extra_labels=None, **kwargs):
        input_dim = shot_features.shape[1]
        if self.survival:
            # discretize survival time into bins
            nbins = 4
            eps = 1e-6
            events = shot_extra_labels["event"]
            uncensored_labels = shot_labels[events == 1]
            if len(uncensored_labels) > 0:
                q_bins = np.quantile(uncensored_labels, q=np.linspace(0, 1, nbins + 1))
            else:
                # if all events are censored, use the entire range of labels
                q_bins = np.quantile(shot_labels, q=np.linspace(0, 1, nbins + 1))
            q_bins[0] = shot_labels.min() - eps
            q_bins[-1] = shot_labels.max() + eps
            shot_labels = np.digitize(shot_labels, bins=q_bins, right=False) - 1
            censoring = 1 - events
            self.num_classes = nbins  # number of bins
            self.criterion = NLLSurvLoss()
        else:
            self.num_classes = 1
            self.criterion = nn.MSELoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        shot_features = torch.tensor(shot_features, dtype=torch.float32).to(self.device)

        if self.survival:
            shot_labels = torch.tensor(shot_labels, dtype=torch.long).to(self.device)
            censoring = torch.tensor(censoring, dtype=torch.long).to(self.device)
        else:
            shot_labels = torch.tensor(shot_labels, dtype=torch.float32).to(self.device)

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
            if self.survival:
                hazards = torch.sigmoid(logits)  # [B, nbins]
                survival = torch.cumprod(1 - hazards, dim=0)  # [B, nbins]
                loss = self.criterion(hazards, survival, shot_labels, censoring)
            else:
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
                if self.survival:
                    hazards = torch.sigmoid(logits)
                    survival = torch.cumprod(1 - hazards, dim=0)
                    risk_scores = -torch.sum(survival, dim=0)
                    prediction = -risk_scores
                    predictions.append(prediction.cpu().numpy())
                else:
                    predictions.append(logits.cpu().numpy())

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


class MultiLayerPerceptronRegressor(CaseLevelTaskAdaptor):
    """
    A PyTorch-based MLP adaptor for regression tasks.
    Attributes:
        shot_features (np.ndarray): Few-shot feature matrix of shape (n_shots, n_features).
        shot_labels (np.ndarray): Few-shot labels.
        test_features (np.ndarray): Test feature matrix of shape (n_test_samples, n_features).
        survival (bool): Whether the task is survival analysis. Default is False.
        hidden_dim (int): Number of hidden units in the model. Default is 256.
        num_layers (int): Number of hidden layers in the MLP. Default is 2.
        num_epochs (int): Number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
        shot_extra_labels (np.ndarray): Optional additional labels for training, used in survival analysis.
    Methods:
        fit():
            Fits the model using the provided few-shot data.
        predict() -> np.ndarray:
            Generates predictions for the test data using the fitted model.
    """

    def __init__(
        self,
        survival=False,
        hidden_dim=256,
        num_layers=3,
        num_epochs=100,
        learning_rate=0.001,
        patience=10,
    ):
        self.survival = survival
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience

    def fit(self, shot_features, shot_labels, shot_extra_labels=None, **kwargs):
        input_dim = shot_features.shape[1]
        if self.survival:
            # discretize survival time into bins
            nbins = 4
            eps = 1e-6
            events = shot_extra_labels["event"]
            uncensored_labels = shot_labels[events == 1]
            if len(uncensored_labels) > 0:
                q_bins = np.quantile(uncensored_labels, q=np.linspace(0, 1, nbins + 1))
            else:
                # if all events are censored, use the entire range of labels
                q_bins = np.quantile(shot_labels, q=np.linspace(0, 1, nbins + 1))
            q_bins[0] = shot_labels.min() - eps
            q_bins[-1] = shot_labels.max() + eps
            shot_labels = np.digitize(shot_labels, bins=q_bins, right=False) - 1
            censoring = 1 - events
            self.num_classes = nbins  # number of bins
            self.criterion = NLLSurvLoss()
        else:
            self.num_classes = 1
            self.criterion = nn.MSELoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        shot_features = torch.tensor(shot_features, dtype=torch.float32).to(self.device)

        if self.survival:
            shot_labels = torch.tensor(shot_labels, dtype=torch.long).to(self.device)
            censoring = torch.tensor(censoring, dtype=torch.long).to(self.device)
        else:
            shot_labels = torch.tensor(shot_labels, dtype=torch.float32).to(self.device)

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
            if self.survival:
                hazards = torch.sigmoid(logits)  # [B, nbins]
                survival = torch.cumprod(1 - hazards, dim=0)  # [B, nbins]
                loss = self.criterion(hazards, survival, shot_labels, censoring)
            else:
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
                if self.survival:
                    hazards = torch.sigmoid(logits)
                    survival = torch.cumprod(1 - hazards, dim=0)
                    risk_scores = -torch.sum(survival, dim=0)
                    prediction = -risk_scores
                    predictions.append(prediction.cpu().numpy())
                else:
                    predictions.append(logits.cpu().numpy())

        return np.array(predictions).squeeze()
