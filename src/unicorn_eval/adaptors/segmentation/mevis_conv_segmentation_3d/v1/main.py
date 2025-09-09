from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.losses.dice import DiceFocalLoss
from tqdm import tqdm

from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v1 import \
    SegmentationUpsampling3D
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, extract_patch_labels, load_patch_data)
from unicorn_eval.adaptors.segmentation.inference import inference3d
from unicorn_eval.adaptors.segmentation.training import train_decoder3d
from unicorn_eval.io import INPUT_DIRECTORY, process, read_inputs


class ConvDecoder3D(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int],
        target_shape: tuple[int, int, int, int],
        num_classes: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes, self.num_channels, self.spatials = (
            num_classes,
            target_shape[0],
            target_shape[1:],
        )
        logging.info(
            f"ConvDecoder3D: {self.num_classes=}, {self.num_channels=}, {self.spatials=}"
        )
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
        self.pack_size = (
            feature_grid_resolution
            if feature_grid_resolution is not None
            else (1, 1, 1)
        )

    @staticmethod
    def instances_from_mask(
        multiclass_mask: np.ndarray, divider_class: int, divided_class: int, sitk_mask
    ):
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

        assert (
            multiclass_mask.ndim == 3
        ), f"Expected 3D input, got {multiclass_mask.shape}"
        instance_regions, num_instances = label(
            multiclass_mask == divider_class, connectivity=1, return_num=True
        )
        if num_instances == 0:
            logging.info(f"Found no instances of class {divider_class} in the mask.")
            return multiclass_mask
        dividers = [
            int(np.round(region.centroid[dim]))
            for region in regionprops(instance_regions)
        ]

        instance_map = np.zeros_like(multiclass_mask)
        for i, threshold in enumerate(dividers):
            min_val = 0 if i == 0 else dividers[i - 1]
            max_val = multiclass_mask.shape[0] if i == len(dividers) - 1 else threshold
            slices = [slice(None)] * multiclass_mask.ndim
            slices[dim] = slice(
                min_val, max_val
            )  # Set the slice for the target dimension
            instance = multiclass_mask[tuple(slices)] == divided_class
            instance_map[tuple(slices)] = instance.astype(instance_map.dtype) * (
                i + 1
            )  # Start from 1 for instances

        # Add the instances from the instance_regions
        instance_map[instance_regions > 0] += (instance_regions + instance_map.max())[
            instance_regions > 0
        ]

        # Add all other classes as one instance per class
        mc_classes = (
            (multiclass_mask > 0)
            & (multiclass_mask != divider_class)
            & (multiclass_mask != divided_class)
        )
        instance_map[mc_classes] += multiclass_mask[mc_classes] + (
            instance_map.max() + 1
        )

        return instance_map

    def gt_to_multiclass(self, gt: torch.Tensor) -> torch.Tensor:
        if (
            self.is_task11
        ):  # Fix Task11 instance segmentation masks using the logic from spider.py
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
            assert (
                self.num_classes == 2
            ), f"Scores only implemented for binary segmentation"
            return model_outputs.softmax(dim=1)[:, 1, ...].unsqueeze(
                1
            )  # return the positive class scores
        else:  # return the predicted classes
            return torch.argmax(model_outputs, dim=1).unsqueeze(
                1
            )  # later code will squeeze second dim

    def fit(
        self,
        *,
        shot_features,
        shot_labels,
        shot_coordinates,
        shot_ids,
        shot_patch_sizes,
        shot_patch_spacings,
        shot_image_sizes,
        shot_image_origins,
        shot_image_spacings,
        shot_image_directions,
        shot_label_spacings,
        shot_label_origins,
        shot_label_directions,
        **kwargs,
    ):
        patch_labels = []
        for idx, label in tqdm(enumerate(shot_labels), desc="Extracting patch labels"):
            case_patch_labels = extract_patch_labels(
                label=label,
                label_spacing=shot_label_spacings[shot_ids[idx]],
                label_origin=shot_label_origins[shot_ids[idx]],
                label_direction=shot_label_directions[shot_ids[idx]],
                image_size=shot_image_sizes[shot_ids[idx]],
                image_origin=shot_image_origins[shot_ids[idx]],
                image_spacing=shot_image_spacings[shot_ids[idx]],
                image_direction=shot_image_directions[shot_ids[idx]],
                start_coordinates=shot_coordinates[idx],
                patch_size=self.global_patch_size,
                patch_spacing=self.global_patch_spacing,
            )
            patch_labels.append(case_patch_labels)

        patch_labels = np.array(patch_labels, dtype=object)

        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=shot_coordinates,
            embeddings=shot_features,
            case_ids=shot_ids,
            patch_sizes=shot_patch_sizes,
            patch_spacings=shot_patch_spacings,
            labels=patch_labels,
            image_sizes=shot_image_sizes,
            image_origins=shot_image_origins,
            image_spacings=shot_image_spacings,
            image_directions=shot_image_directions,
        )
        train_loader = load_patch_data(
            train_data, batch_size=32, balance_bg=self.balance_bg
        )

        # Channels are the remaining dimension before the spatial dimensions
        z_dim, num_spatials = (
            len(shot_features[0][0]),
            self.pack_size[0] * self.pack_size[1] * self.pack_size[2],
        )
        assert (
            z_dim % num_spatials == 0
        ), "Latent dimension must be divisible by spatials!"
        # Task11 GT is encoded with instances in 3 classes. This adaptor can only predict the classes, not instances:
        maxlabel = int(
            max(
                [
                    np.max(patch["features"])
                    for label in patch_labels
                    for patch in label["patches"]
                ]
            )
        )
        self.is_task11 = maxlabel >= 100
        if self.is_task11:
            self.mask_processor = (
                lambda mask_arr, sitk_mask: ConvSegmentation3D.instances_from_mask(
                    mask_arr, 3, 1, sitk_mask
                )
            )
        else:
            self.mask_processor = None
        num_channels, self.num_classes = z_dim // num_spatials, (
            4 if self.is_task11 else 2
        )
        if self.num_classes != maxlabel + 1:
            logging.warning(
                f"{self.num_classes=} != {maxlabel + 1=}, will use {self.num_classes} classes for training"
            )
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
        try:
            self.decoder = train_decoder3d(
                decoder,
                train_loader,
                self.device,
                num_epochs=50,
                loss_fn=loss,
                optimizer=optimizer,
                label_mapper=self.gt_to_multiclass,
            )
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"Out of memory error occurred while training decoder: {e}")
            if self.device.type == "cuda":
                logging.info("Retrying using CPU")
                self.device = torch.device("cpu")
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
            else:
                raise

    def predict(self, test_case_ids):
        predictions = []
        for case_id in test_case_ids:
            logging.info(f"Running inference for case {case_id}")
            test_input = process(
                read_inputs(input_dir=INPUT_DIRECTORY, case_names=[case_id])[0]
            )

            test_data = construct_data_with_labels(
                coordinates=[test_input["coordinates"]],
                embeddings=[test_input["embeddings"]],
                case_ids=[case_id],
                patch_sizes={case_id: test_input["patch_size"]},
                patch_spacings={case_id: test_input["patch_spacing"]},
                image_sizes={case_id: test_input["image_size"]},
                image_origins={case_id: test_input["image_origin"]},
                image_spacings={case_id: test_input["image_spacing"]},
                image_directions={case_id: test_input["image_direction"]},
            )

            test_loader = load_patch_data(test_data, batch_size=1)

            # run inference using the trained decoder
            prediction = inference3d(
                decoder=self.decoder,
                data_loader=test_loader,
                device=self.device,
                return_binary=self.return_binary,
                test_cases=[case_id],
                test_label_sizes={case_id: test_input["label_size"]},
                test_label_spacing={case_id: test_input["label_spacing"]},
                test_label_origins={case_id: test_input["label_origin"]},
                test_label_directions={case_id: test_input["label_direction"]},
                inference_postprocessor=self.inference_postprocessor,  # overwrite original behaviour of applying sigmoid
                mask_postprocessor=self.mask_processor,
            )
            assert len(prediction) == 1
            prediction = prediction[0]

            workdir = (
                Path("/opt/app/predictions")
                if Path("/opt/app/predictions").exists()
                else Path("unicorn/workdir")
            )
            path = workdir / f"vision/{case_id}_pred.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, prediction)

            predictions.append(path)

        return predictions
