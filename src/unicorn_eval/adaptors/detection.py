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

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import filters, gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Sequence
from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor


class DetectionDecoder(nn.Module):
    """MLP that maps vision encoder features to a density map."""

    def __init__(self, input_dim, hidden_dim=512, heatmap_size=16):
        super().__init__()
        self.heatmap_size = heatmap_size  # Store heatmap size
        output_size = heatmap_size * heatmap_size  # Compute output size dynamically

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x).view(-1, self.heatmap_size, self.heatmap_size)


class DetectionDataset(Dataset):
    """Custom dataset to load embeddings and heatmaps."""

    def __init__(self, preprocessed_data, transform=None):
        self.data = preprocessed_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_emb, target_heatmap, patch_coordinates, case = self.data[idx]

        if self.transform:
            patch_emb = self.transform(patch_emb)
            target_heatmap = self.transform(target_heatmap)

        return patch_emb, target_heatmap, patch_coordinates, case


def custom_collate(batch):
    patch_embs, heatmaps, patch_coords, cases = zip(*batch)

    if all(hm is None for hm in heatmaps):
        heatmaps = None
    else:
        heatmaps = default_collate([hm for hm in heatmaps if hm is not None])

    return (
        default_collate(patch_embs),  # Stack patch embeddings
        heatmaps,  # Heatmaps will be None or stacked
        patch_coords,  # Keep as a list
        cases,  # Keep as a list
    )


def heatmap_to_cells_using_maxima(heatmap, neighborhood_size=5, threshold=0.01):
    """
    Detects cell centers in a heatmap using local maxima and thresholding.

    heatmap: 2D array (e.g., 32x32 or 16x16) representing the probability map.
    neighborhood_size: Size of the neighborhood for the maximum filter.
    threshold: Threshold for detecting significant cells based on local maxima.

    Returns:
    x_coords, y_coords: Coordinates of the detected cells' centers.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()  # Convert PyTorch tensor to NumPy array

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}")
    # Apply threshold to heatmap to create a binary map of potential cells
    maxima = heatmap > threshold

    # Use maximum filter to detect local maxima (peaks in heatmap)
    data_max = filters.maximum_filter(heatmap, neighborhood_size)
    maxima = heatmap == data_max  # Only keep true maxima

    # Apply minimum filter to identify significant local differences
    data_min = filters.minimum_filter(heatmap, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0  # Keep only significant maxima

    # Label connected regions (objects) in the binary map
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y = [], []

    # Get the center coordinates of each detected region (cell)
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2  # Center of the x-axis
        y_center = (dy.start + dy.stop - 1) / 2  # Center of the y-axis
        x.append(x_center)
        y.append(y_center)

    return x, y


def train_decoder(decoder, dataloader, heatmap_size=16, num_epochs=200, lr=1e-5):
    """Trains the decoder using the given data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for patch_emb, target_heatmap, _, _ in dataloader:
            patch_emb = patch_emb.to(device)
            target_heatmap = target_heatmap.to(device)
            optimizer.zero_grad()
            pred_heatmap = decoder(patch_emb)
            loss = loss_fn(pred_heatmap, target_heatmap)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    return decoder


def inference(decoder, dataloader, heatmap_size=16, patch_size=224):
    """ "Run inference on the test set."""
    decoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        patch_predictions = []  # List to store the predictions from each patch
        patch_coordinates = []  # List to store the top-left coordinates of each patch
        roi_identifiers = []  # List to store ROI identifiers for each patch

        for patch_emb, _, patch_coordinates_batch, case in dataloader:
            patch_emb = patch_emb.to(device)

            # Make prediction for the patch
            pred_heatmap = decoder(patch_emb)

            # Store the predictions, coordinates, and ROI identifiers
            patch_predictions.append(
                pred_heatmap.cpu().squeeze(0)
            )  # Store predicted heatmap
            patch_coordinates.extend(
                patch_coordinates_batch
            )  # Store coordinates of the patch
            roi_identifiers.extend([case] * len(patch_coordinates_batch))

    case_ids = []  # List to store case identifiers
    test_predictions = []  # List to store points for each case

    for i, (patch_pred, patch_coord, case) in enumerate(
        zip(patch_predictions, patch_coordinates, roi_identifiers)
    ):
        x_local, y_local = heatmap_to_cells_using_maxima(
            patch_pred, neighborhood_size=5
        )
        patch_top_left = patch_coord

        if case not in case_ids:
            case_ids.append(case)
            test_predictions.append([])

        case_index = case_ids.index(case)
        case_points = []
        for x, y in zip(x_local, y_local):
            global_x = patch_top_left[0] + x * (
                patch_size / heatmap_size
            )  # Scaling factor: (ROI size / patch size)
            global_y = patch_top_left[1] + y * (patch_size / heatmap_size)

            case_points.append([global_x, global_y])

        test_predictions[case_index] = np.array(case_points)

    test_predictions = [
        np.array(case_points).tolist() for case_points in test_predictions
    ]
    return test_predictions


def assign_cells_to_patches(cell_data, patch_coordinates, patch_size):
    """Assign ROI cell coordinates to the correct patch."""
    patch_cell_map = {i: [] for i in range(len(patch_coordinates))}

    for x, y, _ in cell_data:
        for i, (x_patch, y_patch) in enumerate(patch_coordinates):
            if (
                x_patch <= x < x_patch + patch_size
                and y_patch <= y < y_patch + patch_size
            ):
                x_local, y_local = x - x_patch, y - y_patch
                patch_cell_map[i].append((x_local, y_local))

    return patch_cell_map


def coordinates_to_heatmap(cell_coords, patch_size=224, heatmap_size=16, sigma=1.0):
    """Convert local cell coordinates into density heatmap."""
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    scale = heatmap_size / patch_size

    for x, y in cell_coords:
        hm_x = int(x * scale)
        hm_y = int(y * scale)
        hm_x, hm_y = np.clip([hm_x, hm_y], 0, heatmap_size - 1)
        heatmap[hm_y, hm_x] += 1.0

    # ensure the output remains float32
    heatmap = gaussian_filter(heatmap, sigma=sigma).astype(np.float32)
    return heatmap


def construct_detection_labels(
    coordinates,
    embeddings,
    names,
    labels=None,
    patch_size=224,
    heatmap_size=16,
    sigma=1.0,
    is_train=True,
):

    processed_data = []

    for case_idx, case_name in enumerate(names):
        patch_coordinates = coordinates[case_idx]
        case_embeddings = embeddings[case_idx]

        if is_train and labels is not None:
            cell_coordinates = labels[case_idx]
            patch_cell_map = assign_cells_to_patches(
                cell_coordinates, patch_coordinates, patch_size
            )

        for i, (x_patch, y_patch) in enumerate(patch_coordinates):
            patch_emb = case_embeddings[i]

            if is_train and labels is not None:
                cell_coordinates = patch_cell_map.get(i, [])
                heatmap = coordinates_to_heatmap(
                    cell_coordinates,
                    patch_size=patch_size,
                    heatmap_size=heatmap_size,
                    sigma=sigma,
                )
            else:
                cell_coordinates = None
                heatmap = None

            processed_data.append(
                (patch_emb, heatmap, (x_patch, y_patch), f"{case_name}")
            )

    return processed_data


class DensityMap(PatchLevelTaskAdaptor):
    def __init__(
        self,
        shot_features,
        shot_labels,
        shot_coordinates,
        shot_names,
        test_features,
        test_coordinates,
        test_names,
        patch_size=224,
        heatmap_size=16,
        num_epochs=200,
        learning_rate=1e-5,
    ):
        super().__init__(
            shot_features,
            shot_labels,
            shot_coordinates,
            test_features,
            test_coordinates,
        )
        self.shot_names = shot_names
        self.test_names = test_names
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decoder = None

    def fit(self):
        input_dim = self.shot_features[0].shape[1]

        shot_data = construct_detection_labels(
            self.shot_coordinates,
            self.shot_features,
            self.shot_names,
            labels=self.shot_labels,
            patch_size=self.patch_size,
            heatmap_size=self.heatmap_size,
        )

        dataset = DetectionDataset(preprocessed_data=shot_data)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=custom_collate
        )

        self.decoder = DetectionDecoder(
            input_dim=input_dim, heatmap_size=self.heatmap_size
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.decoder = train_decoder(
            self.decoder,
            dataloader,
            heatmap_size=self.heatmap_size,
            num_epochs=self.num_epochs,
            lr=self.learning_rate,
        )

    def predict(self) -> list:
        test_data = construct_detection_labels(
            self.test_coordinates,
            self.test_features,
            self.test_names,
            patch_size=self.patch_size,
            heatmap_size=self.heatmap_size,
            is_train=False,
        )
        test_dataset = DetectionDataset(preprocessed_data=test_data)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
        )

        predicted_points = inference(
            self.decoder,
            test_dataloader,
            heatmap_size=self.heatmap_size,
            patch_size=self.patch_size,
        )

        return predicted_points


class TwoLayerPerceptron(nn.Module):
    """2LP used for offline training."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 4)  # dx, dy, dz, logit_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))  # [N,4]


class PatchNoduleRegressor(PatchLevelTaskAdaptor):
    """
    This class implements a lightweight MLP regression head that, for each patch:
      1. Predicts a 4-vector: [dx, dy, dz, logit_p], where (dx, dy, dz) are the predicted
         offsets from the patch center to the true nodule center in patient-space millimetres,
         and logit_p is the raw classification score.
      3. Converts patch indices to world-space coordinates via
         `compute_patch_center_3d`, then reconstructs final nodule centers by adding
         the predicted offsets to the patch center:
      3. Applies a sigmoid to logit_p to obtain a detection probability per patch.

    During inference, `infer_from_patches`:
      - Computes each patch’s world-space center.
      - Runs the MLP to get `[dx, dy, dz, logit_p]`.
      - Adds the offsets to the patch centers to get nodule coordinates.
      - Filters by a probability threshold (e.g., p > 0.9) and outputs an array of
        [x, y, z, p].
    """

    def __init__(
        self,
        shot_features: list[np.ndarray],
        shot_labels: list[list[Sequence[float]]],
        shot_coordinates: list[np.ndarray],
        shot_ids: list[str],
        test_features: list[np.ndarray],
        test_coordinates: list[np.ndarray],
        test_ids: list[str],
        shot_image_origins: dict[str, Sequence[float]],
        shot_image_spacings: dict[str, Sequence[float]],
        shot_image_directions: dict[str, Sequence[float]],
        test_image_origins: dict[str, Sequence[float]],
        test_image_spacings: dict[str, Sequence[float]],
        test_image_directions: dict[str, Sequence[float]],
        hidden_dim: int = 64,
        num_epochs: int = 50,
        lr: float = 1e-3,
        shot_extra_labels: np.ndarray | None = None,
    ):
        super().__init__(
            shot_features,
            shot_labels,
            shot_coordinates,
            test_features,
            test_coordinates,
            shot_extra_labels,
        )
        self.shot_ids = shot_ids
        self.test_ids = test_ids
        self.shot_image_origins = shot_image_origins
        self.shot_image_spacings = shot_image_spacings
        self.shot_image_directions = shot_image_directions
        self.test_image_origins = test_image_origins
        self.test_image_spacings = test_image_spacings
        self.test_image_directions = test_image_directions
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.lr = lr
        input_dim = shot_features[0].shape[1]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 4)

    def compute_patch_center_3d(self, patch_idx, spacing, origin, direction):
        """
        Convert *voxel* index of the patch centre to patient‑space millimetres.
        """
        v_mm = (np.array(patch_idx) + 0.5) * np.array(spacing)  # mm
        R = np.array(direction).reshape(3, 3)
        return np.array(origin) + R.dot(v_mm)

    def train_from_patches(
        self,
        patches: list[dict],
        hidden_dim: int = 64,
        num_epochs: int = 50,
        lr: float = 1e-3,
    ):
        """
        Train a small MLP on a flat list of patch dicts:
          each dict has feature, patch_idx, image_origin, image_spacing,
          image_direction, patch_size, patch_nodules.
        """

        feats = np.stack([p["feature"] for p in patches])  # [N, D]
        idxs = np.stack([p["patch_idx"] for p in patches])  # [N, 3]
        nods = [p["patch_nodules"] for p in patches]  # list of lists
        offsets, cls_labels = [], []
        for p, idx, nod_list in zip(patches, idxs, nods):
            origin = np.array(p["image_origin"])
            spacing = np.array(p["image_spacing"])
            direction = np.array(p["image_direction"]).reshape(3, 3)
            pc = self.compute_patch_center_3d(idx, spacing, origin, direction)
            if nod_list:
                coords = np.array(nod_list)
                nearest = coords[np.argmin(np.linalg.norm(coords - pc, axis=1))]
                delta = nearest - pc
                label = 1.0
            else:
                delta = np.zeros(3, dtype=float)
                label = 0.0
            offsets.append(delta)
            cls_labels.append(label)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        y_off = torch.tensor(offsets, dtype=torch.float32, device=device)
        y_cls = torch.tensor(cls_labels, dtype=torch.float32, device=device)
        self.model = TwoLayerPerceptron(input_dim=x.shape[1], hidden_dim=hidden_dim).to(
            device
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        box_loss = nn.MSELoss()
        cls_loss = nn.BCEWithLogitsLoss()
        for _ in range(num_epochs):
            optimizer.zero_grad()
            out = self.model(x)  # [N,4]
            loss = box_loss(out[:, :3], y_off) + cls_loss(out[:, 3], y_cls)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def infer_from_patches(self, patches: list[dict]) -> np.ndarray:
        """
        Run the trained network on patch dicts and return [x,y,z,p] per row.
        """
        device = next(self.model.parameters()).device
        feats = np.stack([p["feature"] for p in patches])
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        out = self.model(x).cpu().numpy()  # [N,4]
        delta, logits = out[:, :3], out[:, 3]
        centers = np.stack(
            [
                self.compute_patch_center_3d(
                    p["patch_idx"],
                    p["image_spacing"],
                    p["image_origin"],
                    p["image_direction"],
                )
                for p in patches
            ],
            axis=0,
        )
        world_centres = centers + delta
        probs = 1 / (1 + np.exp(-logits))
        return np.concatenate([world_centres, probs[:, None]], axis=1)

    def fit(self):
        # build a *flat* list of per-patch dicts for training
        patch_dicts: list[dict] = []
        for feats_case, idxs_case, nods_case, case_id in zip(
            self.shot_features,
            self.shot_coordinates,
            self.shot_labels,
            self.shot_ids,
        ):
            origin = self.shot_image_origins[case_id]
            spacing = self.shot_image_spacings[case_id]
            direction = self.shot_image_directions[case_id]
            for feat, idx in zip(feats_case, idxs_case):
                patch_dicts.append(
                    {
                        "feature": feat,
                        "patch_idx": idx,
                        "image_origin": origin,
                        "image_spacing": spacing,
                        "image_direction": direction,
                        "patch_nodules": nods_case,
                    }
                )
        self.train_from_patches(
            patches=patch_dicts,
            hidden_dim=self.hidden_dim,
            num_epochs=self.num_epochs,
            lr=self.lr,
        )

    def predict(self) -> np.ndarray:

        test_dicts: list[dict] = []
        case_ids: list[str] = []

        for feats_case, idxs_case, case_id in zip(
            self.test_features,
            self.test_coordinates,
            self.test_ids,
        ):
            origin = self.test_image_origins[case_id]
            spacing = self.test_image_spacings[case_id]
            direction = self.test_image_directions[case_id]
            for feat, idx in zip(feats_case, idxs_case):
                test_dicts.append(
                    {
                        "feature": feat,
                        "patch_idx": idx,
                        "image_origin": origin,
                        "image_spacing": spacing,
                        "image_direction": direction,
                        "patch_nodules": [],  # no GT here
                    }
                )
                case_ids.append(case_id)  # keep alignment with test_dicts

        # raw predictions [x,y,z,p] for every patch
        raw_preds = self.infer_from_patches(test_dicts)
        probs = raw_preds[:, 3]

        # ------- ONLY KEEP p > 0.9 -------- #
        mask = probs > 0.9
        raw_preds = raw_preds[mask]
        case_ids = np.array(case_ids, dtype=object)[mask]

        # prepend test_id to each prediction row
        rows = [[cid, *pred] for cid, pred in zip(case_ids, raw_preds)]
        preds = np.array(rows, dtype=object)

        # Nodule count printout
        n_kept = int(mask.sum())
        print(
            f"[MLPRegressor] Returning {n_kept} nodules (p > 0.9) "
            f"out of {len(mask)} patches"
        )

        return preds
