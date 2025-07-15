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
import torch.nn.functional as F 
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
            patch_pred, neighborhood_size=2
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

        test_predictions[case_index].extend(case_points)

    test_predictions = [
        np.array(case_points).tolist() for case_points in test_predictions
    ]
    return test_predictions


def assign_cells_to_patches(cell_data, patch_coordinates, patch_size):
    """Assign ROI cell coordinates to the correct patch."""
    patch_cell_map = {i: [] for i in range(len(patch_coordinates))}

    for x, y in cell_data:
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

class ScanLevelNoduleDetector(PatchLevelTaskAdaptor):
    """
    Few-shot 3D scan-level nodule detector using light class-token interaction.
    - **Query-based detection**  
  Instead of sliding windows or fully convolutional heads, we allocate a fixed number (`num_queries`) of “query tokens” (via `nn.Embedding`).  Each token learns to attend to one nodule, enabling variable‐count predictions by ranking token confidences.

- **Optional MLP on query tokens**  
  You can toggle an extra two-layer MLP (`use_mlp`) on the class embeddings.  If your base features are highly expressive, turning it off saves compute; if you need more capacity, it adds non-linear transform before matching to the feature volume.

- **Sparse 3D volume construction**  
  `_build_feature_volume` takes an unordered list of patch embeddings + their real-world centers and scatters them into a compact dense tensor of shape `[1, C, D, H, W]`.  It 
  1. normalizes world-space coords to `[0,1]`,  
  2. scales to a user-defined `max_dim`,  
  3. floors to integer grid indices (collapsing zero-range dims),  
  4. and places each patch’s feature vector at its voxel.  
  This lets the decoder head operate on a fixed-shape input regardless of case complexity.

- **Grid ↔ world coordinate mapping**  
  `_grid_to_world_coordinate` inverts the above scatter, converting an integer voxel index back to its approximate real-world center.  This is used both for debug prints and for final world-space outputs.

- **Ground-truth mask generation**  
  `_build_gt_mask` creates a soft 3D “heatmap” of where nodules lie.  Patches closest to any GT center get intensity 1.0, with an exponential fall-off to neighbors.  This mask drives a proximity-weighted focal + Dice + count-regression loss that strongly emphasizes near-GT voxels.

- **Training loop (`fit`)**  
  - **Few-shot per-case episodes**: each “epoch” iterates over one case’s patches.  
  - **Count-regression**: besides voxel-level losses, we predict how many nodules are present (by thresholding each query’s max confidence) and penalize deviation from the true count.  
  - **Plateau LR scheduler**: uses `ReduceLROnPlateau` on the case loss so the learning rate only decays when the model truly stalls.  
  - **Early stopping** after 10 non-improving iterations prevents overfitting on hard cases.

- **Inference (`predict`)**  
  1. Reconstruct feature volume.  
  2. Dot-product each query token with the volume to produce a stack of probability masks.  
  3. Estimate the number of nodules by counting how many queries exceed a confidence threshold.  
  4. Select only the top-`pred_count` queries, find local maxima in each mask, and convert those voxel indices back to real-world coordinates.  
  5. **Hard cap of 7 detections** per scan by ranking all candidates by confidence and discarding the rest.  
  6. Print a final list of `(case_id, x, y, z, confidence)` for downstream evaluation.


    """
    class NoduleClassDecoder(nn.Module):
        def __init__(self, n_queries, feature_size, use_mlp=False):
            super().__init__()  # initialize base nn.Module
            self.use_mlp = use_mlp  # store whether to use MLP on class embeddings
            self.class_embeddings = nn.Embedding(n_queries, feature_size)  # embedding for each query token
            self.image_post_mapping = nn.Identity()  # identity mapping placeholder
            if use_mlp:
                # define a small MLP for class embeddings if requested
                self.mlp = nn.Sequential(
                    nn.Linear(feature_size, feature_size),  # linear layer
                    nn.GELU(),  # nonlinearity
                    nn.Linear(feature_size, feature_size),  # linear layer
                )
        def forward(self, src, class_vector):
            b, c, d, h, w = src.shape  # batch, channels, depth, height, width
            src = self.image_post_mapping(src)  # apply post mapping
            class_embedding = self.class_embeddings(class_vector)  # lookup embeddings
            if self.use_mlp:
                class_embedding = self.mlp(class_embedding)  # optionally transform embeddings
            masks = []  # list to collect per-batch masks
            for i in range(b):
                feat_flat = src[i].view(c, -1)  # flatten spatial dims
                mask = (class_embedding @ feat_flat).view(-1, d, h, w)  # compute dot product then reshape
                masks.append(mask)  # append mask for this batch item
            return torch.stack(masks, dim=0), class_embedding  # return stacked masks and embeddings

    def __init__(
        self,
        shot_features, shot_coordinates, shot_ids, shot_labels,
        test_features, test_coordinates, test_ids,
        test_image_origins, test_image_spacings, test_image_directions,
        shot_image_origins, shot_image_spacings, shot_image_directions,
        patch_size, num_queries=7, use_mlp=True, num_epochs=50, lr=1e-4,
    ):
        super().__init__(  # initialize parent adaptor
            shot_features=shot_features,
            shot_coordinates=shot_coordinates,
            shot_labels=shot_labels,
            test_features=test_features,
            test_coordinates=test_coordinates,
            shot_extra_labels=None,
        )
        self.shot_ids = shot_ids  
        self.test_ids = test_ids  
        self.shot_image_origins = shot_image_origins  
        self.shot_image_spacings = shot_image_spacings  
        self.shot_image_directions = shot_image_directions  
        self.test_image_origins = test_image_origins  
        self.test_image_spacings = test_image_spacings  
        self.test_image_directions = test_image_directions  
        self.patch_size = patch_size  

        # determine feature size from first non-empty shot_features
        feature_size = next(
            (case_feats[0].shape[0] for case_feats in shot_features if len(case_feats) > 0),
            None
        )
        if feature_size is None or feature_size <= 0:
            raise ValueError("Invalid or empty features; cannot determine feature size.")  # error if no features
        self.feature_size = feature_size  # store feature size

        self.num_queries = num_queries  # number of query slots
        self.num_epochs = num_epochs  # training epochs
        self.lr = lr  # learning rate

        # instantiate the decoder model
        self.model = self.NoduleClassDecoder(
            n_queries=self.num_queries,
            feature_size=self.feature_size,
            use_mlp=use_mlp,
        )

    def _build_feature_volume(self, feats, world_coords, max_dim=64):
        if len(feats) == 0 or len(world_coords) == 0:
            raise ValueError("Cannot build feature volume: empty inputs.")
        DEBUG_PATCH_IDX = 3  # <— change this to the patch index you want to inspect
        world_coords = np.array(world_coords)
        feats = np.array(feats)
        if feats.ndim > 2:
            feats = feats.reshape(len(feats), -1)
        if feats.shape[1] != self.feature_size:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_size}, got {feats.shape[1]}")
        mins = world_coords.min(axis=0)
        maxs = world_coords.max(axis=0)
        ranges = maxs - mins
        scale_factor = max_dim / ranges.max() if ranges.max() > 0 else 1.0
        dims = np.ceil(ranges * scale_factor).astype(int)
        dims = np.minimum(np.maximum(dims, 1), max_dim)
        dims[ranges == 0] = 1
        grid = np.zeros((self.feature_size, dims[2], dims[1], dims[0]), dtype=np.float32)
        for i, (feat, wc) in enumerate(zip(feats, world_coords)):
            normalized = np.where(ranges == 0, 0.5, (wc - mins) / ranges)
            coord = np.clip((normalized * (dims - 1)).astype(int), 0, dims - 1)
            x, y, z = coord
            grid[:, z, y, x] = feat
            # ------ DEBUG BLOCK ------
            if i == DEBUG_PATCH_IDX:
                # back-project the voxel center to world coords
                wc_back = self._grid_to_world_coordinate(coord, mins, ranges, dims)
                print(f"[DEBUG] Patch {i} — original wc: {wc}, grid_idx: {coord}, back-projected wc_back: {wc_back}")
            # -------------------------
        return torch.from_numpy(grid).unsqueeze(0), dims, mins, ranges

    @torch.no_grad()
    def predict(self) -> np.ndarray:
        print("\n==== Inference ====")  # inference header
        self.model.eval()  # set eval mode
        predictions = []  # list to collect predictions

        for features_case, coords_case, case_id in zip(
            self.test_features, self.test_coordinates, self.test_ids
        ):
            print(f"\n[TEST] Case ID: {case_id}")  # case ID

            try:
                feat_vol, dims, coord_mins, coord_ranges = self._build_feature_volume(
                    features_case, coords_case
                )
            except ValueError:
                continue  # skip if volume build fails

            masks, _ = self.model(feat_vol, torch.arange(self.num_queries))  # get raw masks
            masks = torch.sigmoid(masks).squeeze(0)  # to probabilities

            # Predict number of nodules by counting queries above confidence threshold
            flat = masks.view(self.num_queries, -1)  # flatten spatial dims
            confidences = flat.max(dim=1).values  # max score per query
            pred_count = int((confidences > 0.5).sum().item())  # threshold at 0.5
            pred_count = max(1, min(pred_count, self.num_queries))  # clamp to [1, num_queries]
            top_queries = confidences.topk(pred_count).indices.tolist()  # select top queries
            print(f"     - Predicted nodule count: {pred_count}, using queries {top_queries}")

            # Detect peaks for each selected query
            for i in top_queries:
                mask = masks[i]
                thr = 0.15
                if mask.max().item() < thr:
                    continue

                from scipy.ndimage import maximum_filter, label as sp_label, center_of_mass
                mask_np = mask.cpu().numpy()
                local_max = (mask_np == maximum_filter(mask_np, size=3)) & (mask_np > thr)
                labels, num = sp_label(local_max)

                for lid in range(1, num + 1):
                    coords = np.where(labels == lid)
                    if not coords[0].size:
                        continue
                    component = (labels == lid)
                    masked = mask_np * component
                    flat_idx = masked.argmax()
                    zc, yc, xc = np.unravel_index(flat_idx, masked.shape)
                    world = self._grid_to_world_coordinate([xc, yc, zc], coord_mins, coord_ranges, dims)

                    # 2) snap to the nearest original patch center
                    orig = np.array(coords_case)                        # shape [N,3]
                    world = np.array(world)                             # convert to numpy array
                    dists = np.linalg.norm(orig - world[None,:], axis=1)
                    best = orig[dists.argmin()]                         # one of your input coords

                    # 3) store that instead
                    conf  = mask[int(zc), int(yc), int(xc)].item()
                    predictions.append([case_id, *best.tolist(), conf])

            # --- limit to top 7 by confidence for this case ---
            case_preds = [p for p in predictions if p[0] == case_id]
            if len(case_preds) > 7:
                # sort by confidence (element 4) descending
                sorted_case = sorted(case_preds, key=lambda x: x[4], reverse=True)
                top7 = set(tuple(x) for x in sorted_case[:7])
                # keep only top7 for this case
                predictions = [p for p in predictions
                               if not (p[0] == case_id and tuple(p) not in top7)]
            # -------------------------------------------------------

            print(
                f"     - Total detections for case {case_id}: "
                f"{len([p for p in predictions if p[0] == case_id])}"
            )  # per-case count

        if predictions:  # summary if any
            case_counts = {}
            for pred in predictions:
                cid = pred[0]
                case_counts[cid] = case_counts.get(cid, 0) + 1
            print(f"\n==== Detection Summary ====")
            print(f"Predictions per case: {case_counts}")
            print(f"Total detections: {sum(case_counts.values())}")
        else:
            print("No predictions made for any case.")

        # print all final predictions
        print("\n==== Final raw prediction coordinates (case_id, x, y, z, conf) ====")
        for row in predictions:
            print(row)

        return np.array(predictions, dtype=object)
    def _build_gt_mask(self, dims, patch_world_coords, nodule_list, coord_mins, coord_ranges, case_id):
        mask = torch.zeros(dims[2], dims[1], dims[0])  # init GT mask
        if not nodule_list:
            return mask  # empty if no nodules
        nloc = np.array(nodule_list)  # array of GT coords
        for pwc in patch_world_coords:  # for each patch
            normalized = np.where(coord_ranges == 0, 0.5, (pwc - coord_mins) / coord_ranges)  # normalize
            gx, gy, gz = np.clip((normalized * (dims - 1)).astype(int), 0, dims - 1)  # grid idx
            dists = np.linalg.norm(nloc - pwc, axis=1)  # distances to nodules
            weight = np.exp(-(dists.min()**2) / (40.0**2))  # falloff weight
            mask[gz, gy, gx] = max(mask[gz, gy, gx], weight)  # assign max weight
        return mask  # return GT mask

    def _grid_to_world_coordinate(self, grid_idx, coord_mins, coord_ranges, dims):
        """
        Convert a grid index [x, y, z] back to world coordinates (mm),
        handling collapsed dimensions where coord_ranges == 0.
        """
        grid_idx = np.array(grid_idx)                            # to array
        normalized = grid_idx / (dims - 1)                        # back to [0,1]
        # if a dimension was collapsed (range==0), just use the original min
        world = np.where(
            coord_ranges == 0,
            coord_mins,
            coord_mins + normalized * coord_ranges
        )
        return world.tolist()

    def fit(self):
        print("==== Few-Shot Training ====")  # training header
        self.model.train()  # set train mode

        # use a smaller initial LR
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr * 0.3)

        # switch to a plateau-based LR scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",        # we want to minimize the loss
            factor=0.5,        # LR ← LR * 0.5 when triggered
            patience=5,        # wait 5 epochs of no improvement
            min_lr=1e-6        # floor on LR
        )

        first = False  # flag for one-time verification print

        for feats_case, coords_case, labels_case, case_id in zip(
            self.shot_features, self.shot_coordinates, self.shot_labels, self.shot_ids
        ):
            print(f"\n[SHOT] Case ID: {case_id}")  # case being trained

            feat_vol, dims, coord_mins, coord_ranges = self._build_feature_volume(feats_case, coords_case)

            if not first and labels_case:
                # verify mapping once at the start
                self._print_patch_coordinate_verification(
                    coords_case, labels_case, case_id, coord_mins, coord_ranges, dims
                )
                first = True

            gt_mask = self._build_gt_mask(dims, coords_case, labels_case, coord_mins, coord_ranges, case_id)
            gt_mask = gt_mask.unsqueeze(0).repeat(self.num_queries, 1, 1, 1)  # repeat for each query

            best_loss = float('inf')
            stale = 0
            true_count = len(labels_case)

            for epoch in range(self.num_epochs):
                optimizer.zero_grad()  # clear grads
                out_logits, _ = self.model(feat_vol, torch.arange(self.num_queries))  # forward pass
                loss, stats = self._improved_loss_function(out_logits.squeeze(0), gt_mask, true_count)
                loss.backward()  # backpropagate
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # clip grads
                optimizer.step()  # optimizer step

                # let the scheduler observe the loss for plateau detection
                scheduler.step(loss)

                # track best loss for early stopping
                current = loss.item()
                if current < best_loss:
                    best_loss = current
                    stale = 0
                else:
                    stale += 1

                # periodic logging with current LR
                if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                    lr = optimizer.param_groups[0]['lr']
                    print(
                        f"     - Case {case_id}, Epoch {epoch+1}/{self.num_epochs}: "
                        f"Loss={current:.6f}, LR={lr:.1e}"
                    )

                if stale >= 10:
                    print(f"     - Early stopping at epoch {epoch+1} (no improvement for 10 epochs)")
                    break

    def _print_patch_coordinate_verification(self, coords_case, labels_case, case_id, coord_mins, coord_ranges, dims):
        """
        Print patch coordinates that contain ground truth nodules for verification.
        This shows the mapping after aggregation step to verify correctness.
        """
        print(f"\n==== Patch Coordinate Verification for Case {case_id} ====")  # header
        print(f"Grid dimensions: {dims}")  # dims
        print(f"Coordinate mins: {coord_mins}")  # mins
        print(f"Coordinate ranges: {coord_ranges}")  # ranges

    def _create_diverse_class_embeddings(self):
        nn.init.xavier_uniform_(self.model.class_embeddings.weight)  # init embeddings
        with torch.no_grad():
            self.model.class_embeddings.weight += torch.randn_like(self.model.class_embeddings.weight) * 0.01  # add noise

    def _improved_loss_function(self, pred_logits, gt_mask, true_count):
        """
        Improved loss function with focal + dice loss and count regularization.
        """
        pred_probs = torch.sigmoid(pred_logits)  # to probabilities
        # boost focus on near-GT voxels even more
        proximity = 1.0 + 4.0 * gt_mask  # up to 5× weight on closest patches
        ce = F.binary_cross_entropy(pred_probs, gt_mask, reduction='none')  # per-voxel CE
        p_t = pred_probs * gt_mask + (1 - pred_probs) * (1 - gt_mask)  # for focal
        alpha_t = 0.25 * gt_mask + 0.75 * (1 - gt_mask)  # class weights
        focal = alpha_t * (1 - p_t) ** 2 * ce  # focal loss
        weighted_focal = (focal * proximity).mean()  # apply proximity

        # dice loss
        inter = (pred_probs * gt_mask * proximity).sum()
        dice = 1 - (2 * inter + 1) / ((pred_probs * proximity).sum() + (gt_mask * proximity).sum() + 1)

        # count regularization: count predicted queries above 0.5
        confidences = pred_probs.view(self.num_queries, -1).max(dim=1).values  # max per query
        pred_count = (confidences > 0.5).float().sum()  # predicted nodule count
        count_loss = F.mse_loss(pred_count, torch.tensor(float(true_count), device=pred_probs.device))  # MSE vs true

        total = weighted_focal + 0.3 * dice + 0.1 * count_loss  # combine with small weight on count

        # compute some stats
        high_acc = ((pred_probs > 0.5) == (gt_mask > 0.5))[gt_mask > 0.8].float().mean() if (gt_mask > 0.8).any() else 0.0

        return total, {
            'focal_loss': weighted_focal.item(),  # focal component
            'dice_loss': dice.item(),  # dice component
            'count_loss': count_loss.item(),  # count reg component
            'pos_ratio': (gt_mask > 0).float().mean().item(),  # fraction positive
            'high_priority_acc': high_acc.item() if isinstance(high_acc, torch.Tensor) else high_acc,  # high-priority accuracy
        }