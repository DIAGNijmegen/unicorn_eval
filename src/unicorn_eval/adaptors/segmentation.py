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
import random
from collections import defaultdict
from typing import Iterable, Type

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.data import DataLoader as dataloader_monai
from monai.data import Dataset as dataset_monai
from monai.losses.dice import DiceFocalLoss, DiceLoss
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.segresnet_ds import (aniso_kernel,
                                              scales_for_resolution)
from monai.utils import has_option
from scipy import ndimage as ndi
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.patch_extraction import extract_patches
from unicorn_eval.adaptors.reconstruct_prediction import stitch_patches_fast


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

        tqdm.write(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

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
        patch_spacing,
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
        self.patch_spacing = patch_spacing
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


def train_decoder3d(decoder, data_loader, device, num_epochs: int = 3, iterations_per_epoch: int | None = None, loss_fn=None, optimizer=None, label_mapper=None, verbose: bool = True):
    if loss_fn is None:
        loss_fn = DiceLoss(sigmoid=True)
    if optimizer is None:
        optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0

        iteration_count = 0
        batch_iter = tqdm(data_loader, total=iterations_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not verbose)
        for batch in batch_iter:
            iteration_count += 1

            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"]
            if label_mapper is not None:
                patch_label = label_mapper(patch_label)
            patch_label = patch_label.to(device)

            optimizer.zero_grad()
            de_output = decoder(patch_emb)
            loss = loss_fn(de_output.squeeze(1), patch_label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with current loss and running average
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss / iteration_count:.4f}")

            if iterations_per_epoch is not None and iteration_count >= iterations_per_epoch:
                break

        tqdm.write(f"Epoch {epoch+1}: Avg total loss = {epoch_loss / iteration_count:.4f}")

    return decoder


def world_to_voxel(coord, origin, spacing, inv_direction):
    relative = np.array(coord) - origin
    voxel = inv_direction @ relative
    voxel = voxel / spacing
    return np.round(voxel).astype(int)

def create_grid(decoded_patches):
    grids = {}

    for idx, patches in tqdm(decoded_patches.items(), desc="Creating grids"):
        stitched = stitch_patches_fast(patches)
        grids[idx] = stitched

    if False:
        # deprecated

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


def inference3d(
    *,
    decoder,
    data_loader,
    device,
    return_binary,
    test_cases,
    test_label_sizes,
    test_label_spacing,
    test_label_origins,
    test_label_directions,
    inference_postprocessor=None,
    mask_postprocessor=None,
):
    decoder.eval()
    with torch.no_grad():
        grouped_predictions = defaultdict(lambda: defaultdict(list))

        for batch in tqdm(data_loader, desc="Inference"):
            inputs = batch["patch"].to(device)  # shape: [B, ...]
            coords = batch["coordinates"]  # list of 3 tensors
            image_idxs = batch["case_number"]

            outputs = decoder(inputs)  # shape: [B, ...]
            if inference_postprocessor is not None:
                pred_mask = inference_postprocessor(outputs)
            else:
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
                        "patch_spacing": [
                            float(batch["patch_spacing"][j][i])
                            for j in range(len(batch["patch_spacing"]))
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
                        "patch_spacing": patches[0]["patch_spacing"],
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
            if mask_postprocessor is not None:
                aligned_preds[case_id] = mask_postprocessor(aligned_preds[case_id], pred_on_gt)
        return [j for j in aligned_preds.values()]





def construct_data_with_labels(
    coordinates,
    embeddings,
    cases,
    patch_size,
    patch_spacing,
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
            K = len(case_embeddings) / len(patch_coordinates) 
            patch_coordinates = np.repeat(
                patch_coordinates, repeats=K, axis=0
            )

        if lbl_feat is not None:
            if len(case_embeddings) != len(lbl_feat["patches"]):
                K = len(case_embeddings) / len(lbl_feat["patches"]) 
                lbl_feat["patches"] = np.repeat(
                    lbl_feat["patches"], repeats=K, axis=0
                )

        for i, patch_img in enumerate(case_embeddings):
            data_dict = {
                "patch": np.array(patch_img, dtype=np.float32),
                "coordinates": patch_coordinates[i],
                "patch_size": patch_size,
                "patch_spacing": patch_spacing,
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
    start_coordinates,
    patch_size: list[int] = [16, 256, 256],
    patch_spacing: list[float] | None = None,
) -> dict:
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

    patches = extract_patches(
        image=label,
        coordinates=start_coordinates,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = label.GetSpacing()

    for patch, coordinates in zip(patches, start_coordinates):
        patch_array = sitk.GetArrayFromImage(patch)
        patch_features.append(
            {
                "coordinates": list(coordinates),  # save the start coordinates
                "features": patch_array,
            }
        )

    return make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=label.GetSize(),
        image_origin=label.GetOrigin(),
        image_spacing=label.GetSpacing(),
        image_direction=label.GetDirection(),
        title="patch_labels",
    )


class BalancedSegmentationDataset(Dataset):
    """
    Balanced dataset for segmentation that ensures equal probability of sampling
    positive and negative patches using inverse probability weighting.
    """

    def __init__(self, data, transform=None, random_seed=42):
        self.transform = transform
        self.rng = random.Random(random_seed)
    
        # Separate positive and negative patches
        self.positive_patches = []
        self.negative_patches = []
    
        for data_dict in data:
            patch_label = data_dict["patch_label"]
            if np.any(patch_label != 0):
                self.positive_patches.append(data_dict)
            else:
                self.negative_patches.append(data_dict)

        self.num_positive = len(self.positive_patches)
        self.num_negative = len(self.negative_patches)

        # Total length is twice the minimum class size to ensure balance
        self.total_length = 2 * min(self.num_positive, self.num_negative) if min(self.num_positive, self.num_negative) > 0 else max(self.num_positive, self.num_negative)

        print(f"BalancedSegmentationDataset: {self.num_positive} positive, {self.num_negative} negative patches")
        print(f"Total balanced dataset size: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Use inverse probability weighting: sample positive and negative with equal probability
        if self.num_positive > 0 and self.num_negative > 0:
            # 50% chance of sampling positive or negative
            if self.rng.random() < 0.5:
                # Sample positive patch
                patch_idx = self.rng.randint(0, self.num_positive - 1)
                data_dict = self.positive_patches[patch_idx]
            else:
                # Sample negative patch
                patch_idx = self.rng.randint(0, self.num_negative - 1)
                data_dict = self.negative_patches[patch_idx]
        elif self.num_positive > 0:
            # Only positive patches available
            patch_idx = self.rng.randint(0, self.num_positive - 1)
            data_dict = self.positive_patches[patch_idx]
        else:
            # Only negative patches available
            patch_idx = self.rng.randint(0, self.num_negative - 1)
            data_dict = self.negative_patches[patch_idx]

        # Apply transform if provided
        if self.transform:
            # Apply transform to patch data if needed
            for key, value in data_dict.items():
                if hasattr(value, 'shape'):  # Apply to array-like data
                    data_dict[key] = self.transform(value)

        return data_dict


def load_patch_data(data_array: np.ndarray, batch_size: int = 80, balance_bg: bool = False) -> DataLoader:
    if balance_bg:
        train_ds = BalancedSegmentationDataset(data=data_array)
    else:
        train_ds = dataset_monai(data=data_array)

    return dataloader_monai(train_ds, batch_size=batch_size, shuffle=False)






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
        balance_bg : Whether to balance background and foreground patches using inverse probability weighting.
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
        patch_spacing,
        return_binary=True,
        balance_bg=False,
    ):
        label_patch_features = []
        for idx, label in tqdm(enumerate(shot_labels), desc="Extracting patch labels"):
            label_feats = extract_patch_labels(
                label=label,
                label_spacing=shot_label_spacing[shot_names[idx]],
                label_origin=shot_label_origins[shot_names[idx]],
                label_direction=shot_label_directions[shot_names[idx]],
                image_size=shot_image_sizes[shot_names[idx]],
                image_origin=shot_image_origins[shot_names[idx]],
                image_spacing=shot_image_spacing[shot_names[idx]],
                image_direction=shot_image_directions[shot_names[idx]],
                start_coordinates=shot_coordinates[idx],
                patch_size=patch_size,
                patch_spacing=patch_spacing,
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
        self.patch_spacing = patch_spacing
        self.decoder = None
        self.return_binary = return_binary
        self.balance_bg = balance_bg

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.shot_names,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            labels=self.shot_labels,
        )

        train_loader = load_patch_data(train_data, batch_size=10, balance_bg=self.balance_bg)
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

    def predict(self) -> list:
        # build test data and loader
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            cases=self.test_cases,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )

        test_loader = load_patch_data(test_data, batch_size=10)

        # run inference using the trained decoder
        return inference3d(
            decoder=self.decoder,
            data_loader=test_loader,
            device=self.device,
            return_binary=self.return_binary,
            test_cases=self.test_cases,
            test_label_sizes=self.test_label_sizes,
            test_label_spacing=self.test_label_spacing,
            test_label_origins=self.test_label_origins,
            test_label_directions=self.test_label_directions
        )


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


class ConvDecoder3D(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int],
        target_shape: tuple[int, int, int, int],
        num_classes: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes, self.num_channels, self.spatials = num_classes, target_shape[0], target_shape[1:]
        print(f"ConvDecoder3D: {self.num_classes=}, {self.num_channels=}, {self.spatials=}")
        self.emb_norm = nn.GroupNorm(1, self.num_channels)
        self.emb_activation = nn.GELU()
        self.ctx_stacks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=self.num_channels,
                        out_channels=self.num_channels,
                        kernel_size=3,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.GroupNorm(1, self.num_channels),
                    nn.GELU(),
                )
                for _ in range(2)
            ]
        )
        self.clf_conv = nn.Conv3d(self.num_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        x = x.view(batchsize := x.shape[0], self.num_channels, *self.spatials)
        x = self.emb_norm(x)
        x = self.emb_activation(x)
        # Do all processing in low resolution
        for stack in self.ctx_stacks:
            x = stack(x)
        x = self.clf_conv(x)
        # After processing, convert the patch into full resolution
        x = F.interpolate(x, size=self.patch_size[::-1], mode="trilinear")
        return x


class ConvSegmentation3D(SegmentationUpsampling3D):

    def __init__(self, *args, feature_grid_resolution=None, **kwargs):
        super().__init__(*args, **kwargs)
        # First three components are the original patchsize, next three are the resolution within the patch
        # If no feature grid resolution is given, use (1, 1, 1) to be compatible with sparse models
        self.pack_size = feature_grid_resolution if feature_grid_resolution is not None else (1, 1, 1)
        self.patch_size = self.patch_size[:3]

    @staticmethod
    def instances_from_mask(multiclass_mask: np.ndarray, divider_class: int, divided_class: int, sitk_mask):
        """
        First, each instance of divider_class segments the image into areas.
        Then, the divided class is split into instances using those areas.

        Returns: instance map for divider_class and divided_class
        """
        dim = np.argmax(np.abs(sitk_mask.GetDirection()[::3]))
        assert multiclass_mask.shape[dim] != min(
            multiclass_mask.shape
        ), f"Metadata inconsistency, cannot process instances {sitk_mask.GetSize()=}"

        from skimage.measure import (  # import inline because it is not used for all tasks
            label, regionprops)

        assert multiclass_mask.ndim == 3, f"Expected 3D input, got {multiclass_mask.shape}"
        instance_regions, num_instances = label(multiclass_mask == divider_class, connectivity=1, return_num=True)
        if num_instances == 0:
            print(f"Found no instances of class {divider_class} in the mask.")
            return multiclass_mask
        dividers = [int(np.round(region.centroid[dim])) for region in regionprops(instance_regions)]

        instance_map = np.zeros_like(multiclass_mask)
        for i, threshold in enumerate(dividers):
            min_val = 0 if i == 0 else dividers[i - 1]
            max_val = multiclass_mask.shape[0] if i == len(dividers) - 1 else threshold
            slices = [slice(None)] * multiclass_mask.ndim
            slices[dim] = slice(min_val, max_val)  # Set the slice for the target dimension
            instance = multiclass_mask[tuple(slices)] == divided_class
            instance_map[tuple(slices)] = instance.astype(instance_map.dtype) * (i + 1)  # Start from 1 for instances

        # Add the instances from the instance_regions
        instance_map[instance_regions > 0] += (instance_regions + instance_map.max())[instance_regions > 0]

        # Add all other classes as one instance per class
        mc_classes = (multiclass_mask > 0) & (multiclass_mask != divider_class) & (multiclass_mask != divided_class)
        instance_map[mc_classes] += multiclass_mask[mc_classes] + (instance_map.max() + 1)

        return instance_map

    def gt_to_multiclass(self, gt: torch.Tensor) -> torch.Tensor:
        if self.is_task11:  # Fix Task11 instance segmentation masks using the logic from spider.py
            res = torch.zeros_like(gt)
            res[(gt > 0) & (gt < 100)] = 1
            res[gt == 100] = 2
            res[gt > 200] = 3
            return res[:, None, ...].long()
        else:
            return (gt[:, None, ...] > 0.5).long()

    @torch.no_grad()
    def inference_postprocessor(self, model_outputs):
        if not self.return_binary:  # return raw scores
            assert self.num_classes == 2, f"Scores only implemented for binary segmentation"
            return model_outputs.softmax(dim=1)[:, 1, ...].unsqueeze(1)  # return the positive class scores
        else:  # return the predicted classes
            return torch.argmax(model_outputs, dim=1).unsqueeze(1)  # later code will squeeze second dim

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.shot_names,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            labels=self.shot_labels,
        )
        train_loader = load_patch_data(train_data, batch_size=32, balance_bg=self.balance_bg)

        # Channels are the remaining dimension before the spatial dimensions
        z_dim, num_spatials = len(self.shot_features[0][0]), self.pack_size[0] * self.pack_size[1] * self.pack_size[2]
        assert z_dim % num_spatials == 0, "Latent dimension must be divisible by spatials!"
        # Task11 GT is encoded with instances in 3 classes. This adaptor can only predict the classes, not instances:
        maxlabel = int(max([np.max(patch["features"]) for label in self.shot_labels for patch in label["patches"]]))
        self.is_task11 = maxlabel >= 100
        if self.is_task11:
            self.mask_processor = lambda mask_arr, sitk_mask: ConvSegmentation3D.instances_from_mask(
                mask_arr, 3, 1, sitk_mask
            )
        else:
            self.mask_processor = None
        num_channels, self.num_classes = z_dim // num_spatials, 4 if self.is_task11 else 2
        if self.num_classes != maxlabel + 1:
            print(f"Warning: {self.num_classes=} != {maxlabel + 1=}, will use {self.num_classes} classes for training")
        target_shape = (num_channels, *self.pack_size[::-1])

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = ConvDecoder3D(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            target_shape=target_shape,
        )

        loss = DiceFocalLoss(to_onehot_y=True, softmax=True, alpha=0.25)
        optimizer = optim.AdamW(decoder.parameters(), lr=3e-3)
        decoder.to(self.device)
        self.decoder = train_decoder3d(
            decoder,
            train_loader,
            self.device,
            num_epochs=8,
            loss_fn=loss,
            optimizer=optimizer,
            label_mapper=self.gt_to_multiclass,
        )

    def predict(self):  # Copied from SegmentationUpsampling3D to change activation
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            cases=self.test_cases,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )

        test_loader = load_patch_data(test_data, batch_size=10)
        return inference3d(
            decoder=self.decoder,
            data_loader=test_loader,
            device=self.device,
            return_binary=self.return_binary,
            test_cases=self.test_cases,
            test_label_sizes=self.test_label_sizes,
            test_label_spacing=self.test_label_spacing,
            test_label_origins=self.test_label_origins,
            test_label_directions=self.test_label_directions,
            inference_postprocessor=self.inference_postprocessor,  # overwrite original behaviour of applying sigmoid
            mask_postprocessor=self.mask_processor,
        )


class UpsampleConvSegAdaptor(nn.Module):
    def __init__(self, target_shape=None, in_channels=32, num_classes=2):
        super().__init__()
        self.target_shape = target_shape
        self.in_channels = in_channels
        # Two intermediate conv layers + final prediction layer
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        C = self.in_channels
        B, feat_len = x.shape
        if feat_len % C != 0:
            raise ValueError(
            f"[Adaptor] Embedding length {feat_len} must be divisible by in_channels={C}."
        )

        flat = feat_len // C

        D_ref, H_ref, W_ref = self.target_shape
        ref_ratio = D_ref * H_ref * W_ref

        k = (flat / ref_ratio) ** (1 / 3)
        D = round(D_ref * k)
        H = round(H_ref * k)
        W = round(W_ref * k)
        
        if D * H * W != flat:
            D, H, W = exact_triplet_from_ref(flat, (D_ref, H_ref, W_ref))
        
        x = x.view(B, C, D, H, W)
        x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        x = self.conv_blocks(x)
        return x


class UpsampleConvSegAdaptorLeakyReLU(UpsampleConvSegAdaptor):
    def __init__(self, target_shape=None, in_channels=32, num_classes=2):
        super().__init__(target_shape=target_shape, in_channels=in_channels, num_classes=num_classes)
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


class ConvUpsampleSegAdaptor(nn.Module):
    def __init__(self, target_shape=None, in_channels=32, num_classes=2):
        super().__init__()
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        C = self.in_channels
        B, feat_len = x.shape
        if feat_len % C != 0:
            raise ValueError(
            f"[Adaptor] Embedding length {feat_len} must be divisible by in_channels={C}."
        )

        flat = x.shape[1] // C

        D_ref, H_ref, W_ref = self.target_shape
        ref_ratio = D_ref * H_ref * W_ref

        k = (flat / ref_ratio) ** (1 / 3)

        D = round(D_ref * k)
        H = round(H_ref * k)
        W = round(W_ref * k)

        if D * H * W != flat:
            D, H, W = exact_triplet_from_ref(flat, (D_ref, H_ref, W_ref))

        x = x.view(B, C, D, H, W)
        x = self.conv_blocks(x)
        x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        return x


class LinearUpsampleConv3D(SegmentationUpsampling3D):
    """
    Patch-level adaptor that performs segmentation by linearly upsampling 
    3D patch-level features followed by convolutional refinement.

    This adaptor takes precomputed patch-level features from 3D medical images
    and predicts voxel-wise segmentation by applying a simple decoder that:
    1) linearly upsamples the patch embeddings to the original resolution, and
    2) passes them through 3D convolution layers for spatial refinement.

    Steps:
    1. Extract patch-level segmentation labels using spatial metadata.
    2. Construct training data from patch features and coordinates.
    3. Train a lightweight 3D decoder that linearly upsamples features and refines them with convolution layers.
    4. At inference, apply the decoder to test patch features and reconstruct full-size segmentation predictions.

    Args:
        shot_features : Patch-level feature embeddings of few-shot labeled volumes.
        shot_labels : Full-resolution segmentation labels (used to supervise the decoder).
        shot_coordinates : Patch coordinates corresponding to shot_features.
        shot_names : Case identifiers for few-shot examples.
        test_features : Patch-level feature embeddings for testing.
        test_coordinates : Patch coordinates corresponding to test_features.
        test_names : Case identifiers for testing examples.
        test_image_sizes, test_image_origins, test_image_spacings, test_image_directions:
            Metadata for reconstructing the spatial layout of test predictions.
        shot_image_spacing, shot_image_origins, shot_image_directions:
            Metadata used to align segmentation labels with patch features during training.
        patch_size : Size of each 3D patch.
        return_binary : Whether to threshold predictions into binary segmentation masks.
    """
    def __init__(self, *args, decoder_cls: Type[nn.Module] = UpsampleConvSegAdaptor, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_task11 = False
        self.is_task06 = False
        self.decoder_cls = decoder_cls

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.shot_names,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            labels=self.shot_labels,
        )

        train_loader = load_patch_data(train_data, batch_size=1, balance_bg=self.balance_bg)

        max_class = max_class_label_from_labels(self.shot_labels)
        if max_class >= 100:
            self.is_task11 = True
            num_classes = 4
        elif max_class > 1:
            self.is_task06 = True
            num_classes = 2
            self.return_binary = False  # Do not threshold predictions for task 06
            # TODO: implement this choice more elegantly
        else:
            num_classes = max_class + 1

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = self.decoder_cls(
            target_shape=self.patch_size[::-1],  # (D, H, W)
            num_classes=num_classes,
        )

        print(f"Training decoder with {num_classes} classes")
        decoder.to(self.device)
        self.decoder = train_seg_adaptor3d(decoder, train_loader, self.device, is_task11=self.is_task11, is_task06=self.is_task06)

    def predict(self) -> list:
        # build test data and loader
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            cases=self.test_cases,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )

        test_loader = load_patch_data(test_data, batch_size=1)

        # run inference using the trained decoder
        return inference3d_softmax(
            decoder=self.decoder,
            data_loader=test_loader,
            device=self.device,
            return_binary=self.return_binary,
            test_cases=self.test_cases,
            test_label_sizes=self.test_label_sizes,
            test_label_spacing=self.test_label_spacing,
            test_label_origins=self.test_label_origins,
            test_label_directions=self.test_label_directions,
            is_task11=self.is_task11
        )


class LinearUpsampleConv3D_V2(LinearUpsampleConv3D):
    """
    Adapts LinearUpsampleConv3D:
    - Enable balanced background sampling by default
    - Use a different training strategy
    - Set batch size to 8
    """
    def __init__(self, *args, balance_bg: bool = True, **kwargs):
        super().__init__(*args, balance_bg=balance_bg, **kwargs)

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.shot_names,
            patch_size=self.patch_size,
            patch_spacing=self.patch_spacing,
            labels=self.shot_labels,
        )

        train_loader = load_patch_data(train_data, batch_size=2, balance_bg=self.balance_bg)

        max_class = max_class_label_from_labels(self.shot_labels)
        if max_class >= 100:
            self.is_task11 = True
            num_classes = 4
        elif max_class > 1:
            self.is_task06 = True
            num_classes = 2
            self.return_binary = False  # Do not threshold predictions for task 06
            # TODO: implement this choice more elegantly
        else:
            num_classes = max_class + 1

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = self.decoder_cls(
            target_shape=self.patch_size[::-1],  # (D, H, W)
            num_classes=num_classes,
        )

        print(f"Training decoder with {num_classes} classes")
        decoder.to(self.device)
        self.decoder = train_seg_adaptor3d_v2(decoder, train_loader, self.device, is_task11=self.is_task11, is_task06=self.is_task06)


def expand_instance_labels(y: np.ndarray) -> np.ndarray:
    """
    Reverse-expand class labels to instance labels.

    Input y uses:
      - 0: background
      - 1: class-A (-> instances 1..99)
      - 2: class-B (-> 100)
      - 3: class-C (-> instances 201..)

    Rules:
      - label==1 -> connected components -> 1,2,3,... up to 99 (100th+ capped at 99)
      - label==2 -> 100
      - label==3 -> connected components -> 201,202,...
      - else     -> 0
    """
    y = y.astype(np.int64, copy=False)
    out = np.zeros_like(y, dtype=np.int64)

    # Connectivity: 2D=8, 3D=26
    structure = np.ones((3,) * y.ndim, dtype=np.uint8)

    # --- label==1 ---
    mask1 = (y == 1)
    if np.any(mask1):
        lbl1, n1 = ndi.label(mask1, structure=structure)
        next_lab = 1
        for cid in range(1, n1 + 1):
            blob = (lbl1 == cid)
            if not np.any(blob):
                continue
            assign = next_lab if next_lab <= 99 else 99
            out[blob] = assign
            next_lab += 1

    # --- label==2 ---
    out[y == 2] = 100

    # --- label==3 ---
    mask3 = (y == 3)
    if np.any(mask3):
        lbl3, n3 = ndi.label(mask3, structure=structure)
        base = 201
        for cid in range(1, n3 + 1):
            blob = (lbl3 == cid)
            if not np.any(blob):
                continue
            out[blob] = base
            base += 1

    return out


def inference3d_softmax(*, decoder, data_loader, device, return_binary, test_cases, test_label_sizes, test_label_spacing, test_label_origins, test_label_directions, is_task11=False):
    decoder.eval()
    with torch.no_grad():
        grouped_predictions = defaultdict(lambda: defaultdict(list))

        for batch in tqdm(data_loader, desc="Inference"):
            inputs = batch["patch"].to(device)  # shape: [B, ...]
            coords = batch["coordinates"]  # list of 3 tensors
            image_idxs = batch["case_number"]

            outputs = decoder(inputs)  # shape: [B, ...]
            probs = torch.softmax(outputs, dim=1)

            if return_binary:
                pred_mask = torch.argmax(probs, dim=1, keepdim=True).float()
            else:
                pred_mask = probs[:, 1:]

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
                        "patch_spacing": [
                            float(batch["patch_spacing"][j][i])
                            for j in range(len(batch["patch_spacing"]))
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
                        "patch_spacing": patches[0]["patch_spacing"],
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
            
            if is_task11:
                pred_on_gt_arr = sitk.GetArrayFromImage(pred_on_gt)
                aligned_preds[case_id] = expand_instance_labels(pred_on_gt_arr)
            else:
                aligned_preds[case_id] = sitk.GetArrayFromImage(pred_on_gt)
            
        return [j for j in aligned_preds.values()]


def max_class_label_from_labels(label_patch_features) -> int:
    """
    Find the maximum class label across all patches.
    Returns the maximum label value, or 0 if none found.
    """
    mx = -1
    for case in label_patch_features:
        for p in case.get("patches", []):
            a = np.asarray(p.get("features", ()))
            if a.size == 0:
                continue
            v = np.nanmax(a)
            if np.isfinite(v) and v > mx:
                mx = int(v)
    return mx if mx >= 0 else 0

def remap_task11_labels(label_patch_features):
    """
    Remap feature labels in-place if this is Task 11.

    Input
    -----
    label_patch_features : np.array(dtype=object)
        Array of "cases". Each case is a dict with:
          - 'patches': list of dicts, where each dict has:
              - 'features': np.ndarray (e.g., shape 128x128x16)

    Logic
    -----
    1) Determine Task 11:
       - Scan all feature arrays and compute a global maximum.
       - If global_max >= 100, treat as Task 11 and apply remapping.

    2) Remapping rules (apply only when Task 11):
       - values in (0, 100)  -> 1
       - values == 100       -> 2
       - values > 200        -> 3
       - all other values (e.g., 0, 101–200, 200) remain unchanged.

    Returns
    -------
    dict with keys:
      - 'is_task11': bool
      - 'global_max': float or int or None
      - 'changed_patches': int, number of patches updated
    """
    # --- Step 1: compute global maximum across all features ---
    global_max = None
    for case in label_patch_features:
        for p in case.get("patches", []):
            feats = p.get("features", None)
            if feats is None:
                continue
            arr = np.asarray(feats)
            if arr.size == 0:
                continue
            m = arr.max()
            global_max = m if global_max is None else max(global_max, m)

    is_task11 = (global_max is not None) and (global_max >= 100)
    if not is_task11:
        return {"is_task11": False, "global_max": global_max, "changed_patches": 0}

    # --- Step 2: apply in-place remapping for Task 11 ---
    changed = 0
    for case in label_patch_features:
        for p in case.get("patches", []):
            feats = p.get("features", None)
            if feats is None:
                continue
            arr = np.asarray(feats)
            if arr.size == 0:
                continue

            orig_dtype = arr.dtype
            mapped = arr.copy()

            # Build masks for each rule
            mask1 = (mapped > 0) & (mapped < 100)
            mask2 = (mapped == 100)
            mask3 = (mapped > 200)

            if mask1.any() or mask2.any() or mask3.any():
                mapped[mask1] = 1
                mapped[mask2] = 2
                mapped[mask3] = 3

                # Preserve original dtype and write back
                p["features"] = mapped.astype(orig_dtype, copy=False)
                changed += 1

    return {"is_task11": True, "global_max": global_max, "changed_patches": changed}

def map_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Rules:
      - y == 100      -> 2
      - 1 <= y <= 99  -> 1
      - y >= 201      -> 3
      - else          -> 0
    """

    y_new = torch.zeros_like(y)

    y_new = torch.where(y == 100, 2, y_new)
    y_new = torch.where((y >= 1) & (y <= 99), 1, y_new)
    y_new = torch.where(y >= 201, 3, y_new)

    return y_new


def train_seg_adaptor3d(decoder, data_loader, device, num_epochs = 3, iterations_per_epoch: int | None = None, is_task11=False, is_task06=False, verbose: bool = True):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0

        # batch progress
        batch_iter = tqdm(data_loader, total=iterations_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not verbose)
        iteration_count = 0
    
        for batch in batch_iter:
            iteration_count += 1

            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"].to(device).long()

            if is_task11 or is_task06:
                patch_label = map_labels(patch_label)

            optimizer.zero_grad()
            de_output = decoder(patch_emb) 

            ce = ce_loss(de_output, patch_label) 
            if is_task06:
                loss = ce
            else:
                dice = dice_loss(de_output, patch_label)
                loss = ce + dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with current loss and running average
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss / iteration_count:.4f}")

            if iterations_per_epoch is not None and iteration_count >= iterations_per_epoch:
                break

        tqdm.write(f"Epoch {epoch+1}: Avg total loss = {epoch_loss / iteration_count:.4f}")

    return decoder


def train_seg_adaptor3d_v2(decoder, data_loader, device, num_iterations = 15_000, is_task11=False, is_task06=False, verbose: bool = True):
    # Use weighted CrossEntropyLoss and focal loss components
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)

    decoder.train()

    epoch_loss = 0.0
    iteration_count = 0
    epoch_iterations = 0

    # Create an infinite iterator over the data loader
    data_iter = iter(data_loader)

    # Progress bar for total iterations
    progress_bar = tqdm(total=num_iterations, desc="Training", disable=not verbose)

    # Train decoder
    while iteration_count < num_iterations:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator when data loader is exhausted
            data_iter = iter(data_loader)
            batch = next(data_iter)

        iteration_count += 1
        epoch_iterations += 1

        patch_emb = batch["patch"].to(device)
        patch_label = batch["patch_label"].to(device).long()

        if is_task11 or is_task06:
            patch_label = map_labels(patch_label)

        optimizer.zero_grad()
        de_output = decoder(patch_emb) 

        ce = ce_loss(de_output, patch_label) 
        if is_task06:
            loss = ce
        else:
            dice = dice_loss(de_output, patch_label)
            loss = ce + dice

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        optimizer.step()

        step_loss = loss.item()
        epoch_loss += step_loss

        # Update progress bar with current loss and running average
        progress_bar.set_postfix(loss=f"{step_loss:.5e}", avg=f"{epoch_loss / epoch_iterations:.5e}")
        progress_bar.update(1)

        if iteration_count % 100 == 0:
            avg_loss = epoch_loss / epoch_iterations
            tqdm.write(f"Iteration {iteration_count}: avg_loss={avg_loss:.5e}")

            epoch_loss = 0.0
            epoch_iterations = 0

    progress_bar.close()

    return decoder


def exact_triplet_from_ref(flat: int, ref: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Find integers (D,H,W) with D*H*W == flat, close to the heuristic proportions
    implied by 'ref' (D_ref,H_ref,W_ref).
    """

    D_ref, H_ref, W_ref = ref
    ref_ratio = max(1, D_ref * H_ref * W_ref)
    k = (flat / ref_ratio) ** (1.0 / 3.0)

    # helper: all divisors of n
    def divisors(n: int):
        ds = []
        r = int(math.isqrt(n))
        for d in range(1, r + 1):
            if n % d == 0:
                ds.append(d)
                q = n // d
                if q != d:
                    ds.append(q)
        return sorted(ds)

    # pick W as a divisor of 'flat' closest to W_ref*k
    w_heur = max(1, int(round(W_ref * k)))
    w_candidates = divisors(flat)
    W = min(w_candidates, key=lambda w: abs(w - w_heur))

    base = flat // W  # now need D*H = base

    # pick D as a divisor of 'base' closest to D_ref*k; H follows
    d_heur = max(1, int(round(D_ref * k)))
    d_candidates = divisors(base)
    D = min(d_candidates, key=lambda d: abs(d - d_heur))
    H = base // D

    # ensure non-zero
    D = max(1, D); H = max(1, H); W = max(1, W)
    assert D * H * W == flat, f"factorization failed: {D}*{H}*{W} != {flat}"
    return D, H, W


def dice_loss(pred, target, smooth=1e-5):
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    one_hot_target = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = torch.sum(pred * one_hot_target, dim=(2, 3, 4))
    union = torch.sum(pred + one_hot_target, dim=(2, 3, 4))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
