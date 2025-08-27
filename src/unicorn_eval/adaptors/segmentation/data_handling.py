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

from typing import Iterable

import numpy as np
import SimpleITK as sitk
from monai.data.dataloader import DataLoader as dataloader_monai
from monai.data.dataset import Dataset as dataset_monai
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from unicorn_eval.adaptors.patch_extraction import extract_patches


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


def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] | None = None,
    image_direction: Iterable[float] | None= None,
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


def load_patch_data(data_array: np.ndarray, batch_size: int = 80, balance_bg: bool = False):
    if balance_bg:
        train_ds = BalancedSegmentationDataset(data=data_array)
    else:
        train_ds = dataset_monai(data=data_array)

    return dataloader_monai(train_ds, batch_size=batch_size, shuffle=False)


class BalancedSegmentationDataset:
    """Placeholder for balanced segmentation dataset - implementation would go here"""
    def __init__(self, data):
        self.data = data