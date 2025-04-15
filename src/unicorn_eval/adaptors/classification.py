from typing import Literal, Union, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


def knn_probing(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    task_type: Literal["classification", "regression"],
    k: int,
    num_workers: int = 8,
    center_feats: bool = False,
    normalize_feats: bool = False,
    event: Optional[np.ndarray] = None,  # only needed for survival
) -> np.ndarray:
    """
    Perform KNN probing for classification, regression, or survival analysis.

    Args:
        train_feats: (N_train, D) training features (np.ndarray)
        train_labels: Labels for classification/regression, or time for survival (np.ndarray)
        test_feats: (N_test, D) test features (np.ndarray)
        task_type: One of "classification", "regression"
        k: Number of neighbors
        num_workers: Number of parallel jobs (for sklearn models)
        center_feats: Subtract mean from features
        normalize_feats: L2 normalize features
        event: Event indicators (1 = event occurred, 0 = censored). Required for survival.

    Returns:
        np.ndarray: Predictions for classification/regression or list of survival functions
    """
    train_feats, test_feats = preprocess_features(
        train_feats, test_feats,
        center=center_feats,
        normalize_feats=normalize_feats,
    )

    if task_type == "classification":
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=num_workers)
        model.fit(train_feats, train_labels)
        return model.predict(test_feats)

    elif task_type == "regression":
        model = KNeighborsRegressor(n_neighbors=k, n_jobs=num_workers)
        model.fit(train_feats, train_labels)
        return model.predict(test_feats)

    else:
        raise ValueError(f"Unknown task: {task_type}")


def weighted_knn_probing(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    k: int,
    task_type: Literal["classification", "ordinal-classification", "regression"],
    center_feats: bool = False,
    normalize_feats: bool = False,
    return_probabilities: bool = False,
    class_values: Optional[np.ndarray] = None,
    metric: Union[Literal["cosine", "euclidean"], Callable[[np.ndarray, np.ndarray], np.ndarray]] = "cosine",
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Predict using weighted kNN with configurable similarity/distance metric.

    Args:
        task_type: "classification", "ordinal-classification" or "regression"
        return_probabilities: return probabilities (only for classification tasks)
        class_values: optional list of valid class values (used in ordinal classification or rounding regression)
        metric: distance/similarity metric ("cosine", "euclidean", or callable)
    Returns:
        predictions or (predictions, probabilities) if classification + return_probabilities
    """
    assert not (task_type == "regression" and return_probabilities), \
        "Cannot return probabilities for regression."

    train_feats, test_feats = preprocess_features(train_feats, test_feats, center_feats, normalize_feats)
    predictions = []

    if task_type in ["classification", "ordinal-classification"]:
        unique_classes = np.unique(train_labels)
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        num_classes = len(unique_classes)
        all_probs = []

    # similarity function
    if callable(metric):
        similarity_fn = metric
    elif metric == "cosine":
        similarity_fn = lambda x, y: cosine_similarity(x, y)
    elif metric == "euclidean":
        similarity_fn = lambda x, y: 1.0 / (euclidean_distances(x, y) + 1e-8)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    for test_point in test_feats:
        sim = similarity_fn(test_point.reshape(1, -1), train_feats).flatten()
        k_indices = np.argsort(-sim)[:k]
        k_labels = train_labels[k_indices]
        k_similarities = sim[k_indices]

        if task_type == "regression":
            weighted_avg = np.sum(k_labels * k_similarities) / (np.sum(k_similarities) + 1e-8)
            if class_values is not None:
                diffs = np.abs(class_values - weighted_avg)
                class_label = class_values[np.argmin(diffs)]
                predictions.append(class_label)
            else:
                predictions.append(weighted_avg)

        elif task_type in ["classification", "ordinal-classification"]:
            class_weights = np.zeros(num_classes)
            for label, sim in zip(k_labels, k_similarities):
                class_weights[class_to_idx[label]] += sim

            class_probs = class_weights / (np.sum(class_weights) + 1e-8)
            all_probs.append(class_probs)

            if task_type == "ordinal-classification":
                expected_val = np.dot(class_probs, unique_classes)
                predicted_class = int(np.round(expected_val))
            else:
                predicted_class = unique_classes[np.argmax(class_probs)]

            predictions.append(predicted_class)

    predictions = np.array(predictions)
    if return_probabilities and task_type in ["classification", "ordinal-classification"]:
        return predictions, np.vstack(all_probs)
    return predictions


def prototype_classification(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    center_feats: bool = True,
    normalize_feats: bool = True,
) -> torch.Tensor:
    """
    Perform prototype-based classification (few-shot learning).

    Args:
        train_feats (torch.Tensor): Training feature vectors.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test feature vectors.
        center_feats (bool): Whether to center features.
        normalize_feats (bool): Whether to normalize features.

    Returns:
        torch.Tensor: Predicted labels for test data.
    """
    train_feats, test_feats = preprocess_features(
        train_feats, test_feats, center_feats, normalize_feats
    )

    unique_labels = torch.unique(train_labels, sorted=True)
    prototypes = torch.stack(
        [train_feats[train_labels == lbl].mean(dim=0) for lbl in unique_labels]
    )

    # Ensure compatibility with device
    prototypes, unique_labels = prototypes.to(test_feats.device), unique_labels.to(
        test_feats.device
    )

    # Compute pairwise distances & get nearest prototype
    distances = (test_feats[:, None] - prototypes[None, :]).norm(dim=-1, p=2)
    return unique_labels[distances.argmin(dim=1)]


def linear_probing(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
    solver: str = "lbfgs",
) -> np.ndarray:
    """
    Perform logistic regression classification.

    Args:
        train_data (np.ndarray): Training data.
        train_labels (np.ndarray): Training labels.
        test_data (np.ndarray): Test data.
        max_iter (int): Maximum iterations.
        C (float): Regularization strength.
        solver (str): Optimization solver.

    Returns:
        np.ndarray: Predicted labels for test data.
    """
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=0)
    model.fit(train_data, train_labels)

    return model.predict(test_data)


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


def mlp(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
) -> torch.Tensor:
    """
    Train an MLP classifier.

    Args:
        train_data (torch.Tensor): Training feature vectors.
        train_labels (torch.Tensor): Training labels.
        test_data (torch.Tensor): Test feature vectors.
        test_labels (torch.Tensor): Test labels.
        input_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
        hidden_dim (int): Hidden layer size.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.

    Returns:
        torch.Tensor: Predicted labels for test data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Move data to device
    train_data, train_labels = train_data.to(device), train_labels.to(device)
    test_data = test_data.to(device)

    # Initialize model
    model = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        _, test_preds = torch.max(test_outputs, 1)

    return test_preds
