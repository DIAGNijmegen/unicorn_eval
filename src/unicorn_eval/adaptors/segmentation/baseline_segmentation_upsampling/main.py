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
from torch.utils.data import DataLoader

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.segmentation.data_handling import (
    SegmentationDataset, construct_segmentation_labels, custom_collate)
from unicorn_eval.adaptors.segmentation.decoders import SegmentationDecoder
from unicorn_eval.adaptors.segmentation.inference import inference
from unicorn_eval.adaptors.segmentation.training import train_decoder
from unicorn_eval.io import INPUT_DIRECTORY, process, read_inputs


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

    def fit(self, shot_features, shot_labels, shot_coordinates, shot_ids, **kwargs):
        input_dim = shot_features[0].shape[1]
        num_classes = max([np.max(label) for label in shot_labels]) + 1

        shot_data = construct_segmentation_labels(
            shot_coordinates,
            shot_features,
            shot_ids,
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
                read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_name])[0]
            )
            test_data = construct_segmentation_labels(
                [test_input["coordinates"]],
                [test_input["embeddings"]],
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
                test_image_sizes={case_name: test_input["image_size"]},
            )
            predictions.extend(predicted_masks)

        return predictions
