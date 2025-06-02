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
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.data import DataLoader as dataloader_monai
from monai.data import Dataset as dataset_monai
from monai.losses import DiceLoss
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.segresnet_ds import aniso_kernel, scales_for_resolution
from monai.utils import has_option
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.patch_extraction import extract_patches


def compute_num_upsample_layers(initial_size, target_size):
    if isinstance(target_size, (tuple, list)):
        assert target_size[0] == target_size[1], "Only square output sizes supported"
        target_size = target_size[0]
    return int(math.log2(target_size / initial_size))


def build_deconv_layers(self, in_channels, num_layers):
    layers = []
    current_channels = in_channels

    for _ in range(num_layers - 1):
        out_channels = min(128, current_channels * 2)
        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )
        current_channels = min(
            128, current_channels * 2
        )  # cap the number of channels at 128

    layers.append(
        nn.ConvTranspose2d(
            current_channels,
            self.num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
    )

    return nn.Sequential(*layers)


class SegmentationDecoder(nn.Module):
    def __init__(self, input_dim, patch_size, num_classes):
        super().__init__()
        self.spatial_dims = (32, 8, 8)
        self.output_size = (patch_size, patch_size)
        self.num_classes = num_classes
        num_deconv_layers = compute_num_upsample_layers(
            self.spatial_dims[1], patch_size
        )

        self.fc = nn.Linear(input_dim, np.prod(self.spatial_dims))

        self.deconv_layers = build_deconv_layers(
            self,
            in_channels=self.spatial_dims[0],
            num_layers=num_deconv_layers,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for m in self.deconv_layers:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc(x)  # Expand embedding
        x = x.view(-1, *self.spatial_dims)  # Reshape into spatial format.
        x = self.deconv_layers(x)  # Upsample to (256, 256)
        x = F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )  # Ensure exact size
        return x


def assign_mask_to_patch(mask_data, x_patch, y_patch, patch_size, padding_value=0):
    """Assign ROI mask to the patch."""
    # patch = mask_data[y_patch:y_patch+patch_size, x_patch:x_patch+patch_size]

    x_end = x_patch + patch_size
    y_end = y_patch + patch_size

    pad_x = max(0, -x_patch)
    pad_y = max(0, -y_patch)
    pad_x_end = max(0, x_end - patch_size)
    pad_y_end = max(0, y_end - patch_size)

    padded_mask = np.pad(
        mask_data,
        ((pad_y, pad_y_end), (pad_x, pad_x_end)),
        mode="constant",
        constant_values=padding_value,
    )
    patch = padded_mask[y_patch : y_patch + patch_size, x_patch : x_patch + patch_size]

    return patch


def construct_segmentation_labels(
    coordinates, embeddings, names, labels=None, patch_size=224, is_train=True
):
    processed_data = []

    for case_idx, case_name in enumerate(names):
        patch_coordinates = coordinates[case_idx]
        case_embeddings = embeddings[case_idx]

        if is_train:
            segmentation_mask = labels[case_idx]

        for i, (x_patch, y_patch) in enumerate(patch_coordinates):
            patch_emb = case_embeddings[i]

            if is_train:
                segmentation_mask_patch = assign_mask_to_patch(
                    segmentation_mask, x_patch, y_patch, patch_size
                )
            else:
                segmentation_mask_patch = None

            processed_data.append(
                (patch_emb, segmentation_mask_patch, (x_patch, y_patch), f"{case_name}")
            )

    return processed_data


class SegmentationDataset(Dataset):
    """Custom dataset to load embeddings and heatmaps."""

    def __init__(self, preprocessed_data, transform=None):
        self.data = preprocessed_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_emb, segmentation_mask_patch, patch_coordinates, case = self.data[idx]

        if self.transform:
            patch_emb = self.transform(patch_emb)
            segmentation_mask_patch = self.transform(segmentation_mask_patch)

        return patch_emb, segmentation_mask_patch, patch_coordinates, case


def custom_collate(batch):
    patch_embs, segmentation_masks, patch_coords, cases = zip(*batch)

    if all(segmap is None for segmap in segmentation_masks):
        segmentation_masks = None
    else:
        segmentation_masks = default_collate(
            [segmap for segmap in segmentation_masks if segmap is not None]
        )  # create a tensor from all the non-None segmentation masks in the batch.

    return (
        default_collate(patch_embs),  # Stack patch embeddings
        segmentation_masks,  # segmentation_masks will be None or stacked
        patch_coords,  # Keep as a list
        cases,  # Keep as a list
    )


def train_decoder(decoder, dataloader, num_epochs=200, lr=0.001):
    """Trains the decoder using the given data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )  # targets are class labels (not one-hot)

    for epoch in range(num_epochs):
        total_loss = 0

        for patch_emb, target_mask, _, _ in dataloader:
            patch_emb = patch_emb.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()
            pred_masks = decoder(patch_emb)
            target_mask = (
                target_mask.long()
            )  # Convert to LongTensor for CrossEntropyLoss

            loss = criterion(pred_masks, target_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    return decoder


def inference(decoder, dataloader, patch_size, test_image_sizes=None):
    """Run inference on the test set and reconstruct into a single 2D array."""
    decoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        patch_predictions = []  # List to store the predictions from each patch
        patch_coordinates = []  # List to store the top-left coordinates of each patch
        roi_identifiers = []  # List to store ROI identifiers for each patch

        for (
            patch_emb,
            _,
            patch_coordinates_batch,
            case,
        ) in dataloader:  # patch_emb, segmentation_mask_patch, patch_coordinates, case
            patch_emb = patch_emb.to(device)

            pred_masks = decoder(patch_emb)
            pred_masks = torch.argmax(
                pred_masks, dim=1
            )  # gives a [batch_size, height, width] tensor with class labels

            patch_predictions.append(
                pred_masks.cpu().squeeze(0).numpy()
            )  # Store predicted heatmap (convert to numpy)
            patch_coordinates.extend(
                patch_coordinates_batch
            )  # Store coordinates of the patch
            roi_identifiers.extend(
                [case] * len(patch_coordinates_batch)
            )  # Store the case identifier for each patch

    predicted_masks = {}
    for pred_masks, (x, y), case in zip(
        patch_predictions, patch_coordinates, roi_identifiers
    ):
        case = case[0] if isinstance(case, list) or isinstance(case, tuple) else case
        if case not in predicted_masks:
            case_image_size = test_image_sizes.get(case, None)
            if case_image_size is not None:
                predicted_masks[case] = np.zeros(case_image_size, dtype=np.float32)
            else:
                raise ValueError(f"Image size not found for case {case}")

        max_x = min(x + patch_size, predicted_masks[case].shape[0])
        max_y = min(y + patch_size, predicted_masks[case].shape[1])
        slice_width = max_x - x
        slice_height = max_y - y

        if slice_height > 0 and slice_width > 0:
            pred_masks_resized = pred_masks[:slice_width, :slice_height]
            predicted_masks[case][
                x : x + slice_width, y : y + slice_height
            ] = pred_masks_resized
        else:
            print(
                f"[WARNING] Skipping assignment for case {case} at ({x}, {y}) due to invalid slice size"
            )

    return [v.T for v in predicted_masks.values()]


class SegmentationUpsampling(PatchLevelTaskAdaptor):
    def __init__(
        self,
        shot_features,
        shot_labels,
        shot_coordinates,
        shot_names,
        test_features,
        test_coordinates,
        test_names,
        test_image_sizes,
        patch_size,
        num_epochs=20,
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
        self.test_image_sizes = test_image_sizes
        self.patch_size = patch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decoder = None

    def fit(self):
        input_dim = self.shot_features[0].shape[1]
        num_classes = max([np.max(label) for label in self.shot_labels]) + 1

        shot_data = construct_segmentation_labels(
            self.shot_coordinates,
            self.shot_features,
            self.shot_names,
            labels=self.shot_labels,
            patch_size=self.patch_size,
        )
        dataset = SegmentationDataset(preprocessed_data=shot_data)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=custom_collate
        )

        self.decoder = SegmentationDecoder(
            input_dim=input_dim, patch_size=self.patch_size, num_classes=num_classes
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.decoder = train_decoder(
            self.decoder, dataloader, num_epochs=self.num_epochs, lr=self.learning_rate
        )

    def predict(self) -> list:
        test_data = construct_segmentation_labels(
            self.test_coordinates,
            self.test_features,
            self.test_names,
            patch_size=self.patch_size,
            is_train=False,
        )
        test_dataset = SegmentationDataset(preprocessed_data=test_data)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
        )

        predicted_masks = inference(
            self.decoder,
            test_dataloader,
            patch_size=self.patch_size,
            test_image_sizes=self.test_image_sizes,
        )

        return predicted_masks


def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()
    return {
        "meta": {
            "patch-size": list(patch_size),
            "patch-spacing": list(patch_spacing),
            "image-size": list(image_size),
            "image-origin": list(image_origin),
            "image-spacing": list(image_spacing),
            "image-direction": list(image_direction),
        },
        "patches": list(patch_features),
        "title": title,
    }


class Decoder3D(nn.Module):
    def __init__(self, latent_dim, target_shape, decoder_kwargs):
        super().__init__()
        self.vector_to_tensor = VectorToTensor(latent_dim, target_shape)
        self.decoder = SegResNetDecoderOnly(**decoder_kwargs)

    def forward(self, x):
        x = self.vector_to_tensor(x)
        return self.decoder(x)

def train_decoder3d(decoder, data_loader, device, num_epochs = 5):
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0
        for batch in data_loader:
            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"].to(device)
            optimizer.zero_grad()

            de_output = decoder(patch_emb)
            loss = loss_fn(de_output.squeeze(1), patch_label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {loss / len(data_loader)}")

    return decoder

def world_to_voxel(coord, origin, spacing, inv_direction):
    relative = np.array(coord) - origin
    voxel = inv_direction @ relative
    voxel = voxel / spacing
    return np.round(voxel).astype(int)

def create_grid(decoded_patches):
    grids = {}

    for idx, patches in decoded_patches.items():
        # Pull meta from the first patch
        meta = patches[0]
        image_size = meta["image_size"]
        image_origin = meta["image_origin"]
        image_spacing = meta["image_spacing"]
        direction = np.array(meta["image_direction"]).reshape(3, 3)
        inv_direction = np.linalg.inv(direction)
        patch_size = meta["patch_size"]

        padded_shape = [
            int(np.ceil(image_size[d] / patch_size[d]) * patch_size[d])
            for d in range(3)
        ]
        pX, pY, pZ = patch_size  # SITK order
        patch_size = (pZ, pY, pX)  # NumPy order
        padding = [(padded_shape[d] - image_size[d]) // 2 for d in range(3)]
        padding_mm = np.array(padding) * image_spacing
        adjusted_origin = image_origin - direction @ padding_mm
        # Initialize grid
        pX, pY, pZ = padded_shape  # SITK order
        grid_shape = (pZ, pY, pX)  # NumPy order
        grid = np.zeros(grid_shape, dtype=np.float32)

        for patch in patches:
            i, j, k = world_to_voxel(
                patch["coord"], adjusted_origin, image_spacing, inv_direction
            )
            patch_array = patch["features"].squeeze(0)
            grid[
                k : k + patch_size[0], j : j + patch_size[1], i : i + patch_size[2]
            ] += patch_array

        x_start = padding[0]
        x_end = x_start + image_size[0]
        y_start = padding[1]
        y_end = y_start + image_size[1]
        z_start = padding[2]
        z_end = z_start + image_size[2]
        cropped = grid[z_start:z_end, y_start:y_end, x_start:x_end]

        pred_img = sitk.GetImageFromArray(cropped)
        pred_img.SetOrigin(tuple(image_origin))
        pred_img.SetSpacing(tuple(image_spacing))
        pred_img.SetDirection(tuple(np.array(meta["image_direction"])))

        grids.update({idx: pred_img})
    return grids


def inference3d(decoder, data_loader, device, return_binary,  test_cases, test_label_sizes, test_label_spacing, test_label_origins, test_label_directions):
    decoder.eval()
    with torch.no_grad():
        grouped_predictions = defaultdict(lambda: defaultdict(list))

        for batch in data_loader:
            inputs = batch["patch"].to(device)  # shape: [B, ...]
            coords = batch["coordinates"]  # list of 3 tensors
            image_idxs = batch["case_number"]

            outputs = decoder(inputs)  # shape: [B, ...]
            probs = torch.sigmoid(outputs)
            if return_binary:
                pred_mask = (probs > 0.5).float()
            else:
                pred_mask = probs

            batch["image_origin"] = batch["image_origin"][0]
            batch["image_spacing"] = batch["image_spacing"][0]
            for i in range(len(image_idxs)):
                image_id = int(image_idxs[i])
                coord = tuple(
                    float(c) for c in coords[i]
                )  # convert list to tuple for use as dict key
                grouped_predictions[image_id][coord].append(
                    {
                        "features": pred_mask[i].cpu().numpy(),
                        "patch_size": [
                            int(batch["patch_size"][j][i])
                            for j in range(len(batch["patch_size"]))
                        ],
                        "image_size": [
                            int(batch["image_size"][j][i])
                            for j in range(len(batch["image_size"]))
                        ],
                        "image_origin": [
                            float(batch["image_origin"][j][i])
                            for j in range(len(batch["image_origin"]))
                        ],
                        "image_spacing": [
                            float(batch["image_spacing"][j][i])
                            for j in range(len(batch["image_spacing"]))
                        ],
                        "image_direction": [
                            float(batch["image_direction"][j][i])
                            for j in range(len(batch["image_direction"]))
                        ],
                    }
                )

        averaged_patches = defaultdict(list)

        for image_id, coord_dict in grouped_predictions.items():
            for coord, patches in coord_dict.items():
                all_features = [p["features"] for p in patches]
                stacked = np.stack(all_features, axis=0)
                avg_features = np.mean(stacked, axis=0)

                averaged_patches[image_id].append(
                    {
                        "coord": list(coord),
                        "features": avg_features,
                        "patch_size": patches[0]["patch_size"],
                        "image_size": patches[0]["image_size"],
                        "image_origin": patches[0]["image_origin"],
                        "image_spacing": patches[0]["image_spacing"],
                        "image_direction": patches[0]["image_direction"],
                    }
                )

        grids = create_grid(averaged_patches)

        aligned_preds = {}

        for case_id, pred_msk in grids.items():
            case = test_cases[case_id]
            gt_size = test_label_sizes[case]
            gt_spacing = test_label_spacing[case]
            gt_origin = test_label_origins[case]
            gt_direction = test_label_directions[case]

            pred_on_gt = sitk.Resample(
                pred_msk,
                gt_size,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                gt_origin,
                gt_spacing,
                gt_direction
            )

            aligned_preds[case_id] = sitk.GetArrayFromImage(pred_on_gt)

        return [j for j in aligned_preds.values()]


def construct_data_with_labels(
    coordinates,
    embeddings,
    cases,
    patch_size,
    labels=None,
    image_sizes=None,
    image_origins=None,
    image_spacings=None,
    image_directions=None,
):
    data_array = []

    for case_idx, case in enumerate(cases):
        # patch_spacing = img_feat['meta']['patch-spacing']
        case_embeddings = embeddings[case_idx]
        patch_coordinates = coordinates[case_idx]

        lbl_feat = labels[case_idx] if labels is not None else None

        if len(case_embeddings) != len(patch_coordinates):
            patch_coordinates = np.repeat(
                patch_coordinates, repeats=len(case_embeddings), axis=0
            )

        if lbl_feat is not None:
            if len(case_embeddings) != len(lbl_feat["patches"]):
                lbl_feat["patches"] = np.repeat(
                    lbl_feat["patches"], repeats=len(case_embeddings), axis=0
                )

        for i, patch_img in enumerate(case_embeddings):
            data_dict = {
                "patch": np.array(patch_img, dtype=np.float32),
                "coordinates": patch_coordinates[i],
                "patch_size": patch_size,
                "case_number": case_idx,
            }

            if lbl_feat is not None:
                patch_lbl = lbl_feat["patches"][i]
                assert np.allclose(
                    patch_coordinates[i], patch_lbl["coordinates"]
                ), "Coordinates don't match!"
                data_dict["patch_label"] = np.array(
                    patch_lbl["features"], dtype=np.float32
                )

            if (
                (image_sizes is not None)
                and (image_origins is not None)
                and (image_spacings is not None)
                and (image_directions is not None)
            ):
                image_size = image_sizes[case]
                image_origin = image_origins[case]
                image_spacing = image_spacings[case]
                image_direction = image_directions[case]

                data_dict["image_size"] = image_size
                data_dict["image_origin"] = (image_origin,)
                data_dict["image_spacing"] = (image_spacing,)
                data_dict["image_direction"] = image_direction

            data_array.append(data_dict)

    return data_array


def extract_patch_labels(
    label,
    label_spacing,
    label_origin,
    label_direction,
    image_size,
    image_spacing,
    image_origin,
    image_direction,
    patch_size: list[int] = [16, 256, 256],
    patch_spacing: list[float] | None = None,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image

    Args:
        image: image object
        title (str): Title of the patch-level neural representation
        patch_size (list[int]): Size of the patches to extract
        patch_spacing (list[float] | None): Voxel spacing of the image. If specified, the image will be resampled to this spacing before patch extraction.
    Returns:
        list[dict]: List of dictionaries containing the patch features
        - coordinates (list[tuple]): List of coordinates for each patch, formatted as:
            ((x_start, x_end), (y_start, y_end), (z_start, z_end)).
        - features (list[float]): List of features extracted from the patch
    """
    label = sitk.GetImageFromArray(label)
    label.SetOrigin(label_origin)
    label.SetSpacing(label_spacing)
    label.SetDirection(label_direction)

    label = sitk.Resample(label,
                          image_size,
                          sitk.Transform(),
                          sitk.sitkNearestNeighbor,
                          image_origin,
                          image_spacing,
                          image_direction)

    patch_features = []

    patches, coordinates = extract_patches(
        image=label,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = label.GetSpacing()

    for patch, coordinates in tqdm(
        zip(patches, coordinates), total=len(patches), desc="Extracting features"
    ):
        patch_array = sitk.GetArrayFromImage(patch)
        patch_features.append(
            {
                "coordinates": list(coordinates[0]),  # save the start coordinates
                "features": patch_array,
            }
        )

    patch_labels = make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=label.GetSize(),
        image_origin=label.GetOrigin(),
        image_spacing=label.GetSpacing(),
        image_direction=label.GetDirection(),
        title="patch_labels",
    )

    return patch_labels


def load_patch_data(data_array: np.ndarray, batch_size: int = 40) -> DataLoader:
    train_ds = dataset_monai(data=data_array)
    train_loader = dataloader_monai(train_ds, batch_size=batch_size, shuffle=False)
    return train_loader


class SegmentationUpsampling3D(PatchLevelTaskAdaptor):
    """
    Patch-level adaptor that trains a 3D upsampling decoder for segmentation.

    This adaptor takes precomputed patch-level features from 3D medical images
    and performs segmentation by training a decoder that upsamples the features
    back to voxel space.

    Steps:
    1. Extract patch-level segmentation labels using spatial metadata.
    2. Construct training data from patch features and coordinates.
    3. Train a 3D upsampling decoder to predict voxel-wise segmentation from patch embeddings.
    4. At inference, apply the trained decoder to test patch features and reconstruct full-size predictions.

    Args:
        shot_features : Patch-level feature embeddings of few shots used for for training.
        shot_labels : Full-resolution segmentation labels.
        shot_coordinates : Patch coordinates corresponding to shot_features.
        shot_names : Case identifiers for few shot patches.
        test_features : Patch-level feature embeddings for testing.
        test_coordinates : Patch coordinates corresponding to test_features.
        test_names : Case identifiers for testing patches.
        test_image_sizes, test_image_origins, test_image_spacings, test_image_directions:
            Metadata for reconstructing full-size test predictions.
        shot_image_spacing, shot_image_origins, shot_image_directions:
            Metadata for extracting training labels at patch-level.
        patch_size : Size of each 3D patch.
        return_binary : Whether to threshold predictions to binary masks.
    """

    def __init__(
        self,
        shot_features,
        shot_coordinates,
        shot_names,
        shot_labels,
        shot_image_spacing,
        shot_image_origins,
        shot_image_directions,
        shot_image_sizes,
        shot_label_spacing,
        shot_label_origins,
        shot_label_directions,
        shot_label_sizes,
        test_features,
        test_coordinates,
        test_names,
        test_image_sizes,
        test_image_origins,
        test_image_spacings,
        test_image_directions,
        test_label_sizes,
        test_label_spacing,
        test_label_origins,
        test_label_directions,
        patch_size,
        return_binary=True,
    ):
        label_patch_features = []
        for idx, label in enumerate(shot_labels):
            label_feats = extract_patch_labels(
                label=label,
                label_spacing=shot_label_spacing[shot_names[idx]],
                label_origin=shot_label_origins[shot_names[idx]],
                label_direction=shot_image_sizes[shot_names[idx]],
                image_size=shot_label_directions[shot_names[idx]],
                image_origin=shot_image_origins[shot_names[idx]],
                image_spacing=shot_image_spacing[shot_names[idx]],
                image_direction=shot_image_directions[shot_names[idx]],
                patch_size=patch_size,
            )
            label_patch_features.append(label_feats)
        label_patch_features = np.array(label_patch_features, dtype=object)

        super().__init__(
            shot_features=shot_features,
            shot_labels=label_patch_features,
            shot_coordinates=shot_coordinates,
            test_features=test_features,
            test_coordinates=test_coordinates,
            shot_extra_labels=None,  # not used here
        )

        self.shot_names = shot_names
        self.test_cases = test_names
        self.test_image_sizes = test_image_sizes
        self.test_image_origins = test_image_origins
        self.test_image_spacings = test_image_spacings
        self.test_image_directions = test_image_directions
        self.shot_image_spacing = shot_image_spacing
        self.shot_image_origins = shot_image_origins
        self.shot_image_directions = shot_image_directions
        self.test_label_sizes = test_label_sizes
        self.test_label_spacing = test_label_spacing
        self.test_label_origins = test_label_origins
        self.test_label_directions = test_label_directions
        self.patch_size = patch_size
        self.decoder = None
        self.return_binary = return_binary

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.shot_names,
            patch_size=self.patch_size,
            labels=self.shot_labels,
        )
        train_loader = load_patch_data(train_data)
        latent_dim = len(self.shot_features[0][0])
        target_patch_size = tuple(int(j / 16) for j in self.patch_size)
        target_shape = (
            latent_dim,
            target_patch_size[2],
            target_patch_size[1],
            target_patch_size[0],
        )

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = Decoder3D(
            latent_dim=latent_dim,
            target_shape=target_shape,
            decoder_kwargs={
                "spatial_dims": 3,
                "init_filters": 32,
                "latent_channels": latent_dim,
                "out_channels": 1,
                "blocks_up": (1, 1, 1, 1),
                "dsdepth": 1,
                "upsample_mode": "deconv",
            },
        )

        decoder.to(self.device)
        self.decoder = train_decoder3d(decoder, train_loader, self.device)

    def predict(self) -> np.ndarray:
        # build test data and loader
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            cases=self.test_cases,
            patch_size=self.patch_size,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )

        test_loader = load_patch_data(test_data)
        # run inference using the trained decoder
        return inference3d(self.decoder, test_loader, self.device, self.return_binary, self.test_cases, self.test_label_sizes, self.test_label_spacing, self.test_label_origins, self.test_label_directions)


class SegResNetDecoderOnly(nn.Module):
    """
    A decoder-only variant of monai's SegResNetDS. (https://docs.monai.io/en/stable/networks.html)

    This network accepts a latent feature vector (e.g. [512]) and reshapes it to
    a 5D tensor (for 3D data) as the initial input. It then decodes the representation
    through a series of upsampling blocks to produce an output segmentation (or regression) map.

    Args:
        spatial_dims (int): Number of spatial dimensions. Default is 3.
        init_filters (int): Base number of filters (not used for encoder, only to help define defaults). Default is 32.
        latent_channels (int): The number of channels in the latent vector. For example, 512.
        out_channels (int): Number of output channels. Default is 2.
        act (tuple or str): Activation type/arguments. Default is "relu".
        norm (tuple or str): Normalization type/arguments. Default is "batch".
        blocks_up (tuple): Number of blocks (repeat count) in each upsampling stage.
                           For example, (1, 1, 1) will result in three upsampling stages.
        dsdepth (int): Number of decoder stages to produce deep supervision heads.
                       Only the last `dsdepth` levels will produce an output head.
        upsample_mode (str): Upsampling method. Default is "deconv".
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        latent_channels: int = 512,
        out_channels: int = 2,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_up: tuple = (1, 1, 1),
        dsdepth: int = 1,
        upsample_mode: str = "deconv",
        resolution: tuple | None = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.dsdepth = max(dsdepth, 1)
        self.resolution = resolution

        anisotropic_scales = None
        if resolution:
            anisotropic_scales = scales_for_resolution(
                resolution, n_stages=len(blocks_up) + 1
            )
        self.anisotropic_scales = anisotropic_scales

        # Prepare activation and normalization configurations.
        act = split_args(act)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)

        n_up = len(blocks_up)
        filters = latent_channels

        self.up_layers = nn.ModuleList()
        for i in range(n_up):
            kernel_size, _, stride = (
                aniso_kernel(anisotropic_scales[len(blocks_up) - i - 1])
                if anisotropic_scales
                else (3, 1, 2)
            )

            level = nn.ModuleDict()
            level["upsample"] = UpSample(
                mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=filters,
                out_channels=filters // 2,
                kernel_size=kernel_size,
                scale_factor=stride,
                bias=False,
                align_corners=False,
            )

            lite_blocks = []
            for _ in range(blocks_up[i]):
                lite_blocks.append(
                    nn.Sequential(
                        Conv[Conv.CONV, spatial_dims](
                            in_channels=filters // 2,
                            out_channels=filters // 2,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                        get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=filters // 2),
                        get_act_layer(act)
                    )
                )
            level["blocks"] = nn.Sequential(*lite_blocks)

            if i >= n_up - dsdepth:
                level["head"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters // 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=True,
                )
            else:
                level["head"] = nn.Identity()

            self.up_layers.append(level)
            filters = filters // 2  # Update the number of channels for the next stage.

    def forward(self, out_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            out_flat (torch.Tensor): A 1D latent feature vector with shape [latent_channels].

        Returns:
            torch.Tensor: The decoded output. For deep supervision, the last head output is returned.
        """
        x = out_flat

        outputs = []
        for level in self.up_layers:
            x = level["upsample"](x)
            x = level["blocks"](x)
            # If this level has a head (for deep supervision), get its output.
            if not isinstance(level["head"], nn.Identity):
                outputs.append(level["head"](x))

        # If deep supervision is used, return the output from the last head;
        # otherwise, simply return the final tensor.
        if outputs:
            return outputs[-1]
        return x


class VectorToTensor(nn.Module):
    """
    Projects a 1D latent vector into a 4D/5D tensor with spatial dimensions.

    For a 3D image, this transforms a vector of size `latent_dim` into a tensor
    with shape [batch, out_channels, D, H, W]. In this example, we assume the target
    shape (excluding the batch dimension) is (out_channels, 2, 16, 16).

    Args:
        latent_dim (int): Dimensionality of the latent vector (e.g., 512).
        target_shape (tuple): The target output shape excluding the batch dimension.
                              For example, (64, 2, 16, 16) where 64 is the number of channels.
    """

    def __init__(self, latent_dim: int, target_shape: tuple):
        super().__init__()
        self.target_shape = target_shape
        target_numel = 1
        for dim in target_shape:
            target_numel *= dim
        self.fc = nn.Linear(latent_dim, target_numel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A latent feature vector of shape [latent_dim] or [batch, latent_dim].

        Returns:
            torch.Tensor: A tensor of shape [batch, *target_shape].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)
        x = x.view(x.size(0), *self.target_shape)
        return x