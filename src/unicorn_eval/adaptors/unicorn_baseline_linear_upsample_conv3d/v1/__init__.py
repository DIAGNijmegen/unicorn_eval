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

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset
from tqdm import tqdm

from unicorn_eval.adaptors.aimhi_linear_upsample_conv3d.v1 import dice_loss
from unicorn_eval.adaptors.aimhi_linear_upsample_conv3d.v2 import (
    LinearUpsampleConv3D, map_labels, max_class_label_from_labels)
from unicorn_eval.adaptors.segmentation import (construct_data_with_labels,
                                                load_patch_data)


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


class UnicornLinearUpsampleConv3D_V1(LinearUpsampleConv3D):
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



def train_seg_adaptor3d_v2(decoder, data_loader, device, num_iterations = 5_000, is_task11=False, is_task06=False, verbose: bool = True):
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

