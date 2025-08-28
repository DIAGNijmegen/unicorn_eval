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

# TODO: refactor these adaptors in one per folder

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
from monai.losses.dice import DiceFocalLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor
from unicorn_eval.adaptors.segmentation.data_handling import (
    SegmentationDataset, construct_data_with_labels,
    construct_segmentation_labels, custom_collate, extract_patch_labels,
    load_patch_data)
from unicorn_eval.adaptors.segmentation.decoders import (ConvDecoder3D,
                                                         Decoder3D,
                                                         SegmentationDecoder)
from unicorn_eval.adaptors.segmentation.inference import inference, inference3d
from unicorn_eval.adaptors.segmentation.training import (train_decoder,
                                                         train_decoder3d)


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
        patch_size,
        patch_spacing,
        num_epochs=20,
        learning_rate=1e-5,
    ):
        super().__init__(
            shot_features,
            shot_labels,
            shot_coordinates,
            test_features,
            test_coordinates,
        )
        self.shot_names = shot_names
        self.test_names = test_names
        self.test_image_sizes = test_image_sizes
        self.patch_size = patch_size
        self.patch_spacing = patch_spacing
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decoder = None

    def fit(self):
        input_dim = self.shot_features[0].shape[1]
        num_classes = max([np.max(label) for label in self.shot_labels]) + 1

        shot_data = construct_segmentation_labels(
            self.shot_coordinates,
            self.shot_features,
            self.shot_names,
            labels=self.shot_labels,
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
        patch_size,
        patch_spacing,
        return_binary=True,
        balance_bg=False,
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
                patch_size=patch_size,
                patch_spacing=patch_spacing,
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
        self.patch_size = patch_size
        self.patch_spacing = patch_spacing
        self.decoder = None
        self.return_binary = return_binary
        self.balance_bg = balance_bg

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

        train_loader = load_patch_data(train_data, batch_size=10, balance_bg=self.balance_bg)
        latent_dim = len(self.shot_features[0][0])
        target_patch_size = tuple(int(j / 16) for j in self.patch_size)
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
        self.decoder = train_decoder3d(decoder, train_loader, self.device)

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


class ConvSegmentation3D(SegmentationUpsampling3D):

    def __init__(self, *args, feature_grid_resolution=None, **kwargs):
        super().__init__(*args, **kwargs)
        # First three components are the original patchsize, next three are the resolution within the patch
        # If no feature grid resolution is given, use (1, 1, 1) to be compatible with sparse models
        self.pack_size = feature_grid_resolution if feature_grid_resolution is not None else (1, 1, 1)
        self.patch_size = self.patch_size[:3]

    @staticmethod
    def instances_from_mask(multiclass_mask: np.ndarray, divider_class: int, divided_class: int, sitk_mask):
        """
        First, each instance of divider_class segments the image into areas.
        Then, the divided class is split into instances using those areas.

        Returns: instance map for divider_class and divided_class
        """
        dim = np.argmax(np.abs(sitk_mask.GetDirection()[::3]))
        assert multiclass_mask.shape[dim] != min(
            multiclass_mask.shape
        ), f"Metadata inconsistency, cannot process instances {sitk_mask.GetSize()=}"

        from skimage.measure import (  # import inline because it is not used for all tasks
            label, regionprops)

        assert multiclass_mask.ndim == 3, f"Expected 3D input, got {multiclass_mask.shape}"
        instance_regions, num_instances = label(multiclass_mask == divider_class, connectivity=1, return_num=True)
        if num_instances == 0:
            print(f"Found no instances of class {divider_class} in the mask.")
            return multiclass_mask
        dividers = [int(np.round(region.centroid[dim])) for region in regionprops(instance_regions)]

        instance_map = np.zeros_like(multiclass_mask)
        for i, threshold in enumerate(dividers):
            min_val = 0 if i == 0 else dividers[i - 1]
            max_val = multiclass_mask.shape[0] if i == len(dividers) - 1 else threshold
            slices = [slice(None)] * multiclass_mask.ndim
            slices[dim] = slice(min_val, max_val)  # Set the slice for the target dimension
            instance = multiclass_mask[tuple(slices)] == divided_class
            instance_map[tuple(slices)] = instance.astype(instance_map.dtype) * (i + 1)  # Start from 1 for instances

        # Add the instances from the instance_regions
        instance_map[instance_regions > 0] += (instance_regions + instance_map.max())[instance_regions > 0]

        # Add all other classes as one instance per class
        mc_classes = (multiclass_mask > 0) & (multiclass_mask != divider_class) & (multiclass_mask != divided_class)
        instance_map[mc_classes] += multiclass_mask[mc_classes] + (instance_map.max() + 1)

        return instance_map

    def gt_to_multiclass(self, gt: torch.Tensor) -> torch.Tensor:
        if self.is_task11:  # Fix Task11 instance segmentation masks using the logic from spider.py
            res = torch.zeros_like(gt)
            res[(gt > 0) & (gt < 100)] = 1
            res[gt == 100] = 2
            res[gt > 200] = 3
            return res[:, None, ...].long()
        else:
            return (gt[:, None, ...] > 0.5).long()

    @torch.no_grad()
    def inference_postprocessor(self, model_outputs):
        if not self.return_binary:  # return raw scores
            assert self.num_classes == 2, f"Scores only implemented for binary segmentation"
            return model_outputs.softmax(dim=1)[:, 1, ...].unsqueeze(1)  # return the positive class scores
        else:  # return the predicted classes
            return torch.argmax(model_outputs, dim=1).unsqueeze(1)  # later code will squeeze second dim

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
        train_loader = load_patch_data(train_data, batch_size=32, balance_bg=self.balance_bg)

        # Channels are the remaining dimension before the spatial dimensions
        z_dim, num_spatials = len(self.shot_features[0][0]), self.pack_size[0] * self.pack_size[1] * self.pack_size[2]
        assert z_dim % num_spatials == 0, "Latent dimension must be divisible by spatials!"
        # Task11 GT is encoded with instances in 3 classes. This adaptor can only predict the classes, not instances:
        maxlabel = int(max([np.max(patch["features"]) for label in self.shot_labels for patch in label["patches"]]))
        self.is_task11 = maxlabel >= 100
        if self.is_task11:
            self.mask_processor = lambda mask_arr, sitk_mask: ConvSegmentation3D.instances_from_mask(
                mask_arr, 3, 1, sitk_mask
            )
        else:
            self.mask_processor = None
        num_channels, self.num_classes = z_dim // num_spatials, 4 if self.is_task11 else 2
        if self.num_classes != maxlabel + 1:
            print(f"Warning: {self.num_classes=} != {maxlabel + 1=}, will use {self.num_classes} classes for training")
        target_shape = (num_channels, *self.pack_size[::-1])

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = ConvDecoder3D(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            target_shape=target_shape,
        )

        loss = DiceFocalLoss(to_onehot_y=True, softmax=True, alpha=0.25)
        optimizer = optim.AdamW(decoder.parameters(), lr=3e-3)
        decoder.to(self.device)
        self.decoder = train_decoder3d(
            decoder,
            train_loader,
            self.device,
            num_epochs=8,
            loss_fn=loss,
            optimizer=optimizer,
            label_mapper=self.gt_to_multiclass,
        )

    def predict(self):  # Copied from SegmentationUpsampling3D to change activation
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
            inference_postprocessor=self.inference_postprocessor,  # overwrite original behaviour of applying sigmoid
            mask_postprocessor=self.mask_processor,
        )