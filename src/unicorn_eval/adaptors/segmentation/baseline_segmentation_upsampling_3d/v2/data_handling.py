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
from torch.utils.data import Dataset


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
