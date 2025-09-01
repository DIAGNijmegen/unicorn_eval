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

from collections import defaultdict
from typing import Callable

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from unicorn_eval.adaptors.reconstruct_prediction import stitch_patches_fast


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
    inference_postprocessor: Callable | None = None,
    mask_postprocessor: Callable | None = None,
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

            batch["image_origin"] = batch["image_origin"]
            batch["image_spacing"] = batch["image_spacing"]
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
