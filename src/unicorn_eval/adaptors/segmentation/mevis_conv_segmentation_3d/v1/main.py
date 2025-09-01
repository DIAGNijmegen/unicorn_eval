from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.losses.dice import DiceFocalLoss

from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v1 import \
    SegmentationUpsampling3D
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, load_patch_data)
from unicorn_eval.adaptors.segmentation.inference import inference3d
from unicorn_eval.adaptors.segmentation.training import train_decoder3d


class ConvDecoder3D(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int],
        target_shape: tuple[int, int, int, int],
        num_classes: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes, self.num_channels, self.spatials = num_classes, target_shape[0], target_shape[1:]
        print(f"ConvDecoder3D: {self.num_classes=}, {self.num_channels=}, {self.spatials=}")
        self.emb_norm = nn.GroupNorm(1, self.num_channels)
        self.emb_activation = nn.GELU()
        self.ctx_stacks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=self.num_channels,
                        out_channels=self.num_channels,
                        kernel_size=3,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.GroupNorm(1, self.num_channels),
                    nn.GELU(),
                )
                for _ in range(2)
            ]
        )
        self.clf_conv = nn.Conv3d(self.num_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        x = x.view(batchsize := x.shape[0], self.num_channels, *self.spatials)
        x = self.emb_norm(x)
        x = self.emb_activation(x)
        # Do all processing in low resolution
        for stack in self.ctx_stacks:
            x = stack(x)
        x = self.clf_conv(x)
        # After processing, convert the patch into full resolution
        x = F.interpolate(x, size=self.patch_size[::-1], mode="trilinear")
        return x


class ConvSegmentation3D(SegmentationUpsampling3D):

    def __init__(self, *args, feature_grid_resolution=None, **kwargs):
        super().__init__(*args, **kwargs)
        # First three components are the original patchsize, next three are the resolution within the patch
        # If no feature grid resolution is given, use (1, 1, 1) to be compatible with sparse models
        self.pack_size = feature_grid_resolution if feature_grid_resolution is not None else (1, 1, 1)

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
            case_names=self.shot_names,
            patch_sizes=self.shot_patch_sizes,
            patch_spacings=self.shot_patch_spacings,
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
            patch_size=self.global_patch_size,
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
            case_names=self.test_cases,
            patch_sizes=self.test_patch_sizes,
            patch_spacings=self.test_patch_spacings,
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
