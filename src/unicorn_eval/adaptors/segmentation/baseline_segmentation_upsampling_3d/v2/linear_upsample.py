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
import torch.nn as nn

from unicorn_eval.adaptors.segmentation.adaptors import \
    SegmentationUpsampling3D
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2.main import (
    LinearUpsampleConv3D_V2, UpsampleConvSegAdaptor,
    max_class_label_from_labels)
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, load_patch_data)
from unicorn_eval.adaptors.segmentation.decoders import Decoder3D
from unicorn_eval.adaptors.segmentation.inference import inference3d


class SegmentationUpsampling3D_V2(SegmentationUpsampling3D):
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
        latent_dim = len(self.shot_features[0][0])
        blocks_up = (1, 1, 1, 1)  # number of upsampling blocks, each upsampling by factor 2
        target_patch_size = tuple(int(j / 2 ** len(blocks_up)) for j in self.patch_size)
        latent_dim_reduce_factor = (np.prod(target_patch_size) * 16)
        target_shape = (
            latent_dim // latent_dim_reduce_factor,
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
                "latent_channels": latent_dim // latent_dim_reduce_factor,
                "out_channels": 1,
                "blocks_up": blocks_up,
                "dsdepth": 1,
                "upsample_mode": "deconv",
            },
        )

        decoder.to(self.device)
        self.decoder = train_seg_adaptor3d_v2(
            decoder=decoder,
            data_loader=train_loader,
            device=self.device,
            is_task06=False,
            is_task11=True,
        )

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


class UnicornLinearUpsampleConv3D_V1(LinearUpsampleConv3D_V2):
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
