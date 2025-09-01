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

import logging
import random
from typing import Callable, Iterable

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

    x_end = x_patch + patch_size    # Calcluate the end x coordinate of the patch
    y_end = y_patch + patch_size    # Calcluate the end y coordinate of the patch

    assert x_patch >= 0 and y_patch >= 0, f"Negative patch coordinates ({x_patch}, {y_patch}) are out of bounds."

     # if x_end exceeds the image width, pad to the right side, if y_end exceeds the image height, pad to the bottom side 
    pad_x_end = max(0, x_end - mask_data.shape[1])
    pad_y_end = max(0, y_end - mask_data.shape[0])

    padded_mask = np.pad(
        mask_data,
        ((0, pad_y_end), (0, pad_x_end)),
        mode="constant",
        constant_values=padding_value,
    )

    patch = padded_mask[y_patch : y_patch + patch_size, x_patch : x_patch + patch_size]
    assert patch.shape == (patch_size, patch_size), f"Patch shape {patch.shape} does not match expected size {(patch_size, patch_size)}"

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
    coordinates: list[np.ndarray],
    embeddings: list[np.ndarray],
    case_names: list[str],
    patch_sizes: dict[str, list[int]],
    patch_spacings: dict[str, list[float]],
    labels: np.ndarray | None = None,  # contains dict[str, any]
    image_sizes: dict[str, list[int]] | None = None,
    image_origins: dict[str, list[float]] | None = None,
    image_spacings: dict[str, list[float]] | None = None,
    image_directions: dict[str, list[float]] | None = None,
    label_mapper: Callable | None = None,
):
    data_array = []

    for case_idx, case_name in enumerate(case_names):
        # patch_spacing = img_feat['meta']['patch-spacing']
        case_embeddings = embeddings[case_idx]
        patch_coordinates = coordinates[case_idx]

        lbl_feat = labels[case_idx] if labels is not None else None

        if len(case_embeddings) != len(patch_coordinates):
            K = len(case_embeddings) / len(patch_coordinates)
            logging.warning(
                f"Number of embeddings ({len(case_embeddings)}) does not match number of coordinates ({len(patch_coordinates)}) for case {case_name}. Repeating coordinates {K} times."
            )
            patch_coordinates = np.repeat(
                patch_coordinates, repeats=K, axis=0
            )

        if lbl_feat is not None:
            if len(case_embeddings) != len(lbl_feat["patches"]):
                K = len(case_embeddings) / len(lbl_feat["patches"])
                logging.warning(
                    f"Number of embeddings ({len(case_embeddings)}) does not match number of label patches ({len(lbl_feat['patches'])}) for case {case_name}. Repeating label patches {K} times."
                )
                lbl_feat["patches"] = np.repeat(
                    lbl_feat["patches"], repeats=K, axis=0
                )

        for i, patch_img in enumerate(case_embeddings):
            data_dict = {
                "patch": np.array(patch_img, dtype=np.float32),
                "coordinates": patch_coordinates[i],
                "patch_size": patch_sizes[case_name],
                "patch_spacing": patch_spacings[case_name],
                "case_number": case_idx,
            }
            if lbl_feat is not None:
                patch_lbl = lbl_feat["patches"][i]
                if label_mapper is not None:
                    patch_lbl["features"] = label_mapper(patch_lbl["features"])

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
                data_dict["image_size"] = image_sizes[case_name]
                data_dict["image_origin"] = image_origins[case_name]
                data_dict["image_spacing"] = image_spacings[case_name]
                data_dict["image_direction"] = image_directions[case_name]

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


class BalancedSegmentationDataset(Dataset):
    """
    Balanced dataset for segmentation that ensures equal probability of sampling
    positive and negative patches using inverse probability weighting.
    """

    def __init__(self, data, transform=None, random_seed=42):
        self.transform = transform
        self.rng = random.Random(random_seed)
    
        # Keep track which patches contain which labels
        self.patches_by_label: dict[int, list[dict]] = {}
    
        for data_dict in data:
            patch_label = data_dict["patch_label"]
            for label_value in np.unique(patch_label):
                if label_value not in self.patches_by_label:
                    self.patches_by_label[label_value] = []

                self.patches_by_label[label_value].append(data_dict)

        self.total_length = len(data)
        self.positive_classes = sorted(set(self.patches_by_label.keys()) - {0})
        if not self.positive_classes:
            raise ValueError("No positive classes found.")

        num_patches_per_label = {k: len(v) for k, v in self.patches_by_label.items()}
        print(f"BalancedSegmentationDataset: Num patches per class: {num_patches_per_label}")
        print(f"Total balanced dataset size: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Sample a patch with foreground 90% of the time, equally distributed across classes
        if self.rng.random() < 0.9:
            class_to_sample = self.rng.choice(self.positive_classes)
        else:
            class_to_sample = 0  # Sample a negative patch

        # Select a random patch from the chosen class
        patch_idx = self.rng.randint(0, len(self.patches_by_label[class_to_sample]) - 1)
        data_dict = self.patches_by_label[class_to_sample][patch_idx]

        # Apply transform if provided
        if self.transform:
            # Apply transform to patch data if needed
            for key, value in data_dict.items():
                if hasattr(value, 'shape'):  # Apply to array-like data
                    data_dict[key] = self.transform(value)

        return data_dict
