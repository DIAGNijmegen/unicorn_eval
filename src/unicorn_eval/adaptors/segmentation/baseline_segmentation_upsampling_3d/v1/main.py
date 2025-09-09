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
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.segmentation.data_handling import (
    SegmentationDataset, construct_data_with_labels,
    construct_segmentation_labels, custom_collate, extract_patch_labels,
    load_patch_data)
from unicorn_eval.adaptors.segmentation.decoders import (Decoder3D,
                                                         SegmentationDecoder)
from unicorn_eval.adaptors.segmentation.inference import inference, inference3d
from unicorn_eval.adaptors.segmentation.training import (train_decoder,
                                                         train_decoder3d)
from unicorn_eval.io import INPUT_DIRECTORY, process, read_inputs, extract_embeddings


class SegmentationUpsampling(PatchLevelTaskAdaptor):
    def __init__(
        self,
        global_patch_size,
        global_patch_spacing,
        num_epochs=20,
        learning_rate=1e-5,
    ):
        self.patch_size = global_patch_size
        self.patch_spacing = global_patch_spacing
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decoder = None

    def fit(self, shot_features, shot_labels, shot_coordinates, shot_names, **kwargs):
        input_dim = shot_features[0].shape[1]
        num_classes = max([np.max(label) for label in shot_labels]) + 1

        shot_data = construct_segmentation_labels(
            shot_coordinates,
            shot_features,
            shot_names,
            labels=shot_labels,
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

    def predict(self, test_cases) -> list:
        predictions = []
        for case_name in test_cases:
            test_input = process(
                read_inputs(
                    input_dir=INPUT_DIRECTORY, case_names=[case_name]
                )[0]
            )
            case_informations = extract_embeddings(test_input)
            test_data = construct_segmentation_labels(
                [case_informations["coordinates"]],
                [case_informations["embeddings"]],
                [case_name],
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
                test_image_sizes={case_name: case_informations["image_size"]},
            )
            predictions.extend(predicted_masks)

        return predictions


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
        global_patch_size,
        global_patch_spacing,
        return_binary=True,
        balance_bg=False,
    ):
        self.global_patch_size = global_patch_size
        self.global_patch_spacing = global_patch_spacing
        self.decoder = None
        self.return_binary = return_binary
        self.balance_bg = balance_bg

    def fit(self, shot_features, shot_labels, shot_coordinates, shot_names, shot_patch_sizes, shot_patch_spacings, shot_image_sizes, shot_image_origins, shot_image_spacings, shot_image_directions, shot_label_spacings, shot_label_origins, shot_label_directions, **kwargs):
        label_patch_features = []
        for idx, label in tqdm(enumerate(shot_labels), desc="Extracting patch labels"):
            label_feats = extract_patch_labels(
                label=label,
                label_spacing=shot_label_spacings[shot_names[idx]],
                label_origin=shot_label_origins[shot_names[idx]],
                label_direction=shot_label_directions[shot_names[idx]],
                image_size=shot_image_sizes[shot_names[idx]],
                image_origin=shot_image_origins[shot_names[idx]],
                image_spacing=shot_image_spacings[shot_names[idx]],
                image_direction=shot_image_directions[shot_names[idx]],
                start_coordinates=shot_coordinates[idx],
                patch_size=shot_patch_sizes[shot_names[idx]],
                patch_spacing=shot_patch_spacings[shot_names[idx]],
            )
            label_patch_features.append(label_feats)
        label_patch_features = np.array(label_patch_features, dtype=object)

        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=shot_coordinates,
            embeddings=shot_features,
            case_names=shot_names,
            patch_sizes=shot_patch_sizes,
            patch_spacings=shot_patch_spacings,
            labels=label_patch_features,
        )

        train_loader = load_patch_data(train_data, batch_size=10, balance_bg=self.balance_bg)
        latent_dim = len(shot_features[0][0])
        target_patch_size = tuple(int(j / 16) for j in self.global_patch_size)
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
        try:
            self.decoder = train_decoder3d(decoder, train_loader, self.device)
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"Out of memory error occurred while training decoder: {e}")
            if self.device.type == 'cuda':
                logging.info("Retrying using CPU")
                self.device = torch.device("cpu")
                decoder.to(self.device)
                self.decoder = train_decoder3d(decoder, train_loader, self.device)
            else:
                raise

    def predict(self, test_cases) -> list:
        predictions = []
        for case_name in test_cases:
            test_input = process(
                read_inputs(
                    input_dir=INPUT_DIRECTORY, case_names=[case_name]
                )[0]
            )
            case_informations = extract_embeddings(test_input)

            # build test data and loader
            test_data = construct_data_with_labels(
                coordinates=[case_informations["coordinates"]],
                embeddings=[case_informations["embeddings"]],
                case_names=[case_name],
                patch_sizes={case_name: case_informations["patch_size"]},
                patch_spacings={case_name: case_informations["patch_spacing"]},
                image_sizes={case_name: case_informations["image_size"]},
                image_origins={case_name: case_informations["image_origin"]},
                image_spacings={case_name: case_informations["image_spacing"]},
                image_directions={case_name: case_informations["image_direction"]},
            )

            test_loader = load_patch_data(test_data, batch_size=1)

            # run inference using the trained decoder
            prediction = inference3d(
                decoder=self.decoder,
                data_loader=test_loader,
                device=self.device,
                return_binary=self.return_binary,
                test_cases=[case_name],
                test_label_sizes=[case_informations["label_size"]],
                test_label_spacing=[case_informations["label_spacing"]],
                test_label_origins=[case_informations["label_origin"]],
                test_label_directions=[case_informations["label_direction"]]
            )
            predictions.extend(prediction)

        return predictions
