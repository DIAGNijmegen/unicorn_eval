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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor


class SegmentationDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc = nn.Linear(input_dim, 8 * 8 * 32)

        # reshape to (32, 8, 8) and upscale to (256, 256)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                32, 64, kernel_size=4, stride=2, padding=1, output_padding=1
            ),  # 8*8 -> 16*16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 128, kernel_size=4, stride=2, padding=1, output_padding=1
            ),  # 16*16 -> 32*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, output_padding=1
            ),  # 32*32 -> 64*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=1
            ),  # 64*64 -> 128*128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 128*128 -> 256*256
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for m in self.deconv_layers:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc(x)  # Expand embedding
        x = x.view(-1, 32, 8, 8)  # Reshape into spatial format
        x = self.deconv_layers(x)  # Upsample to (256, 256)
        x = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )  # Ensure exact size
        return x


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


def train_decoder(decoder, dataloader, num_epochs=200, lr=0.001):
    """Trains the decoder using the given data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )  # targets are class labels (not one-hot)

    for epoch in range(num_epochs):
        total_loss = 0

        for patch_emb, target_mask, _, _ in dataloader:
            patch_emb = patch_emb.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()
            pred_masks = decoder(patch_emb)
            target_mask = (
                target_mask.long()
            )  # Convert to LongTensor for CrossEntropyLoss

            loss = criterion(pred_masks, target_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    return decoder


def inference(decoder, dataloader, patch_size=224, test_image_sizes=None):
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
                print(f"Image size not found for case {case}")

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
                f"Skipping assignment for case {case} at ({x}, {y}) due to invalid slice size"
            )

    return [v.T for v in predicted_masks.values()]


class SegmentationUpsampling(PatchLevelTaskAdaptor):
    def __init__(
        self,
        shot_features,
        shot_labels,
        shot_coordinates,
        shot_names,
        test_features,
        test_coordinates,
        test_names,
        test_image_sizes,
        patch_size=224,
        num_epochs=20,
        learning_rate=1e-5,
    ):
        super().__init__(shot_features, shot_labels, shot_coordinates, test_features, test_coordinates)
        self.shot_names = shot_names
        self.test_names = test_names
        self.test_image_sizes = test_image_sizes
        self.patch_size = patch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decoder = None

    def fit(self):
        input_dim = self.shot_features[0].shape[1]
        num_classes = max(
            [np.max(label) for label in self.shot_labels]
        ) + 1

        train_data = construct_segmentation_labels(
            self.shot_coordinates,
            self.shot_features,
            self.shot_names,
            labels=self.shot_labels,
            patch_size=self.patch_size,
        )
        dataset = SegmentationDataset(preprocessed_data=train_data)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=custom_collate
        )

        self.decoder = SegmentationDecoder(input_dim=input_dim, num_classes=num_classes).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.decoder = train_decoder(
            self.decoder, dataloader, num_epochs=self.num_epochs, lr=self.learning_rate
        )

    def predict(self) -> list:
        test_data = construct_segmentation_labels(
            self.test_coordinates,
            self.test_features,
            self.test_names,
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
            test_image_sizes=self.test_image_sizes,
        )

        return predicted_masks