from __future__ import annotations

from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, load_patch_data,
    make_patch_level_neural_representation)
from unicorn_eval.adaptors.segmentation.inference import world_to_voxel


class LinearUpsampleConv3D_V1(PatchLevelTaskAdaptor):
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
        global_patch_size,
        global_patch_spacing,
        shot_patch_sizes=None,
        test_patch_sizes=None,
        shot_patch_spacings=None,
        test_patch_spacings=None,
        return_binary=True,
    ):
        label_patch_features = []
        for idx, label in enumerate(shot_labels):
            label_feats = extract_patch_labels_no_resample(
                label=label,
                label_spacing=shot_label_spacing[shot_names[idx]],
                label_origin=shot_label_origins[shot_names[idx]],
                label_direction=shot_label_directions[shot_names[idx]],
                image_size=shot_image_sizes[shot_names[idx]],
                image_origin=shot_image_origins[shot_names[idx]],
                image_spacing=shot_image_spacing[shot_names[idx]],
                image_direction=shot_image_directions[shot_names[idx]],
                patch_size=global_patch_size,
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
        self.patch_size = global_patch_size
        self.patch_spacing = global_patch_spacing
        self.shot_patch_sizes = shot_patch_sizes
        self.test_patch_sizes = test_patch_sizes
        self.shot_patch_spacings = shot_patch_spacings
        self.test_patch_spacings = test_patch_spacings
        self.decoder = None
        self.return_binary = return_binary

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            case_names=self.shot_names,
            patch_sizes=self.shot_patch_sizes,
            patch_spacings=self.shot_patch_spacings,
            labels=self.shot_labels,
        )
        train_loader = load_patch_data(train_data, batch_size=1)

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = LightweightSegAdaptor(
            target_shape=self.patch_size,
        )

        decoder.to(self.device)
        self.decoder = train_seg_adaptor3d(decoder, train_loader, self.device)

    def predict(self) -> np.ndarray:
        # build test data and loader
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            case_names=self.test_cases,
            patch_sizes=self.test_patch_sizes,
            patch_spacings=self.test_patch_spacings,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )

        test_loader = load_patch_data(test_data, batch_size=1)
        # run inference using the trained decoder

        return seg_inference3d(self.decoder, test_loader, self.device, self.return_binary, self.test_cases, self.test_label_sizes, self.test_label_spacing, self.test_label_origins, self.test_label_directions)



def extract_patch_labels_no_resample(
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
    label_array = label.copy()  
    label = sitk.GetImageFromArray(label)
    label.SetOrigin(image_origin)
    label.SetSpacing(image_spacing)
    label.SetDirection(image_direction)

    # a = np.array_equal(label_array, sitk.GetArrayFromImage(label))
    patch_features = []

    D, H, W = label_array.shape  # numpy shape: (z, y, x)
    d, h, w = patch_size

    for z in range(0, D - d + 1, d):
        for y in range(0, H - h + 1, h):
            for x in range(0, W - w + 1, w):
                patch = label_array[z:z + d, y:y + h, x:x + w]
                corner_index = (x, y, z)
                physical_coord = label.TransformIndexToPhysicalPoint(corner_index)

                patch_features.append({
                "coordinates": list(physical_coord),
                "features": patch
            })

    if patch_spacing is None:
        patch_spacing = label.GetSpacing()

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


def seg_inference3d(decoder, data_loader, device, return_binary,  test_cases, test_label_sizes, test_label_spacing, test_label_origins, test_label_directions):
    decoder.eval()
    with torch.no_grad():
        grouped_predictions = defaultdict(lambda: defaultdict(list))

        for batch in data_loader:
            inputs = batch["patch"].to(device)  # shape: [B, ...]
            coords = batch["coordinates"]  # list of 3 tensors
            image_idxs = batch["case_number"]

            outputs = decoder(inputs)  # shape: [B, ...]
            probs = torch.softmax(outputs, dim=1)  # channel dim = 1

            # probs = torch.sigmoid(outputs)
            if return_binary:
                pred_mask = torch.argmax(probs, dim=1).float()
            else:
                pred_mask = probs[:, :1:2]

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

        final_images = []
        for image_id, patch_list in averaged_patches.items():
            image_size = patch_list[0]["image_size"]  # [Z, Y, X]
            patch_size = patch_list[0]["patch_size"]  # [D, H, W]

            origin = patch_list[0]["image_origin"]
            spacing = patch_list[0]["image_spacing"]
            direction = np.array(patch_list[0]["image_direction"]).reshape(3, 3)
            inv_direction = np.linalg.inv(direction)

            # Initialize empty volume
            grid = np.zeros(image_size, dtype=np.float32)
            count = np.zeros(image_size, dtype=np.float32)  # For averaging overlap

            for patch in patch_list:
                coord = patch["coord"]
                patch_data = patch["features"]  # shape: [D, H, W]
                d, h, w = patch_size

                i, j, k = world_to_voxel(coord, origin, spacing, inv_direction)

         
                d, h, w = patch["features"].shape

    
                z_slice = slice(k, k + d)
                y_slice = slice(j, j + h)
                x_slice = slice(i, i + w)

                # Accumulate into grid
                grid[z_slice, y_slice, x_slice] += patch_data
                count[z_slice, y_slice, x_slice] += 1.0

            # Avoid division by zero
            count[count == 0] = 1.0
            final_image = grid / count
            final_images.append(final_image)
        return final_images



class LightweightSegAdaptor(nn.Module):
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
            # nn.Conv3d(mid_channels, num_classes, kernel_size=3, padding=1) 
        )
        
    def forward(self, x):
        C = self.in_channels
        flat_voxel_count = x.shape[1] // C

        D_ref, H_ref, W_ref = self.target_shape
        ref_ratio = D_ref * H_ref * W_ref

        k = (flat_voxel_count / ref_ratio) ** (1 / 3)

        D = round(D_ref * k)
        H = round(H_ref * k)
        W = round(W_ref * k)

        x = x.view(1, C, D, H, W)
        x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        x = self.conv_blocks(x)

        return x


def dice_loss(pred, target, smooth=1e-5):
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    one_hot_target = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = torch.sum(pred * one_hot_target, dim=(2, 3, 4))
    union = torch.sum(pred + one_hot_target, dim=(2, 3, 4))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_seg_adaptor3d(decoder, data_loader, device, num_epochs = 3):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0

        # batch progress
        batch_iter = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        iteration_count = 0

        for batch in batch_iter:
            iteration_count += 1

            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"].to(device).long()

            optimizer.zero_grad()
            de_output = decoder(patch_emb)
            ce = ce_loss(de_output, patch_label)  
            dice = dice_loss(de_output, patch_label)
            loss = ce + dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss / iteration_count:.4f}")

        print(f"Epoch {epoch+1}: Avg total loss = {epoch_loss / iteration_count:.4f}")                               
    return decoder
