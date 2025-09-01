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

import numpy as np
import torch
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2.main import \
    max_class_label_from_labels
from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v2.training import \
    train_decoder3d_v2
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, extract_patch_labels, load_patch_data)
from unicorn_eval.adaptors.segmentation.decoders import Decoder3D
from unicorn_eval.adaptors.segmentation.inference import inference3d


def label_mapper(y: np.ndarray) -> np.ndarray:
    """
    Rules:
      - y == 100      -> 2
      - 1 <= y <= 99  -> 1
      - y >= 201      -> 3
      - else          -> 0
    """

    y_new = np.zeros_like(y)

    y_new = np.where(y == 100, 2, y_new)
    y_new = np.where((y >= 1) & (y <= 99), 1, y_new)
    y_new = np.where(y >= 201, 3, y_new)

    return y_new


class SegmentationUpsampling3D_V2(PatchLevelTaskAdaptor):
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
        global_patch_size,
        global_patch_spacing,
        shot_patch_sizes,
        test_patch_sizes,
        shot_patch_spacings,
        test_patch_spacings,
        return_binary=True,
        balance_bg=True,
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
                patch_size=global_patch_size,
                patch_spacing=global_patch_spacing,
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
        self.shot_image_sizes = shot_image_sizes
        self.shot_image_spacings = shot_image_spacing
        self.shot_image_origins = shot_image_origins
        self.shot_image_directions = shot_image_directions
        self.test_label_sizes = test_label_sizes
        self.test_label_spacing = test_label_spacing
        self.test_label_origins = test_label_origins
        self.test_label_directions = test_label_directions
        self.patch_size = global_patch_size
        self.patch_spacing = global_patch_spacing
        self.shot_patch_sizes = shot_patch_sizes
        self.shot_patch_spacings = shot_patch_spacings
        self.test_patch_sizes = test_patch_sizes
        self.test_patch_spacings = test_patch_spacings
        self.decoder = None
        self.return_binary = return_binary
        self.balance_bg = balance_bg
        self.num_classes = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            case_names=self.shot_names,
            patch_sizes=self.shot_patch_sizes,
            patch_spacings=self.shot_patch_spacings,
            labels=self.shot_labels,
            image_sizes=self.shot_image_sizes,
            image_origins=self.shot_image_origins,
            image_spacings=self.shot_image_spacings,
            image_directions=self.shot_image_directions,
            label_mapper=label_mapper,
        )

        train_loader = load_patch_data(train_data, batch_size=10, balance_bg=self.balance_bg)

        max_class = max_class_label_from_labels(self.shot_labels)
        self.num_classes = max_class + 1

        latent_dim = len(self.shot_features[0][0])
        blocks_up = (1, 1, 1, 1)  # number of upsampling blocks, each upsampling by factor 2
        target_patch_size = tuple(int(j / 2 ** len(blocks_up)) for j in self.patch_size)
        target_shape = (
            latent_dim,
            target_patch_size[2],
            target_patch_size[1],
            target_patch_size[0],
        )

        # set up decoder
        decoder_kwargs={
            "spatial_dims": 3,
            "init_filters": 32,
            "latent_channels": latent_dim,
            "out_channels": self.num_classes,
            "blocks_up": blocks_up,
            "dsdepth": 1,
            "upsample_mode": "deconv",
            "act": "leakyrelu",
        }
        print(f"Setting up decoder with: {latent_dim=}, {target_shape=}, {decoder_kwargs=}")
        decoder = Decoder3D(
            latent_dim=latent_dim,
            target_shape=target_shape,
            decoder_kwargs=decoder_kwargs,
        )

        decoder.to(self.device)
        self.decoder = train_decoder3d_v2(
            decoder=decoder,
            data_loader=train_loader,
            device=self.device,
        )

    def predict(self) -> list:
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
            test_label_directions=self.test_label_directions,
            inference_postprocessor=self.inference_postprocessor,
        )

    def inference_postprocessor(self, mask: torch.Tensor) -> torch.Tensor:
        # Apply post-processing to the predicted mask
        if not self.return_binary:  # return raw scores
            assert self.num_classes == 2, f"Scores only implemented for binary segmentation"
            return mask.softmax(dim=1)[:, 1, ...]  # return the positive class scores
        else:  # return the predicted classes
            return torch.argmax(mask, dim=1)
