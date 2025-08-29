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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.segresnet_ds import (aniso_kernel,
                                              scales_for_resolution)
from monai.utils.misc import has_option


def compute_num_upsample_layers(initial_size, target_size):
    if isinstance(target_size, (tuple, list)):
        assert target_size[0] == target_size[1], "Only square output sizes supported"
        target_size = target_size[0]
    return int(math.log2(target_size / initial_size))


def build_deconv_layers(self, in_channels, num_layers):
    layers = []
    current_channels = in_channels

    for _ in range(num_layers - 1):
        out_channels = min(128, current_channels * 2)
        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )
        current_channels = min(
            128, current_channels * 2
        )  # cap the number of channels at 128

    layers.append(
        nn.ConvTranspose2d(
            current_channels,
            self.num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
    )

    return nn.Sequential(*layers)


class SegmentationDecoder(nn.Module):
    def __init__(self, input_dim, patch_size, num_classes):
        super().__init__()
        self.spatial_dims = (32, 8, 8)
        self.output_size = (patch_size, patch_size)
        self.num_classes = num_classes
        num_deconv_layers = compute_num_upsample_layers(
            self.spatial_dims[1], patch_size
        )

        self.fc = nn.Linear(input_dim, np.prod(self.spatial_dims))

        self.deconv_layers = build_deconv_layers(
            self,
            in_channels=self.spatial_dims[0],
            num_layers=num_deconv_layers,
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
        x = x.view(-1, *self.spatial_dims)  # Reshape into spatial format.
        x = self.deconv_layers(x)  # Upsample to (256, 256)
        x = F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )  # Ensure exact size
        return x


class Decoder3D(nn.Module):
    def __init__(self, latent_dim, target_shape, decoder_kwargs):
        super().__init__()
        self.vector_to_tensor = VectorToTensor(latent_dim, target_shape)
        self.decoder = SegResNetDecoderOnly(**decoder_kwargs)

    def forward(self, x):
        x = self.vector_to_tensor(x)
        return self.decoder(x)


class SegResNetDecoderOnly(nn.Module):
    """
    A decoder-only variant of monai's SegResNetDS. (https://docs.monai.io/en/stable/networks.html)

    This network accepts a latent feature vector (e.g. [512]) and reshapes it to
    a 5D tensor (for 3D data) as the initial input. It then decodes the representation
    through a series of upsampling blocks to produce an output segmentation (or regression) map.

    Args:
        spatial_dims (int): Number of spatial dimensions. Default is 3.
        init_filters (int): Base number of filters (not used for encoder, only to help define defaults). Default is 32.
        latent_channels (int): The number of channels in the latent vector. For example, 512.
        out_channels (int): Number of output channels. Default is 2.
        act (tuple or str): Activation type/arguments. Default is "relu".
        norm (tuple or str): Normalization type/arguments. Default is "batch".
        blocks_up (tuple): Number of blocks (repeat count) in each upsampling stage.
                           For example, (1, 1, 1) will result in three upsampling stages.
        dsdepth (int): Number of decoder stages to produce deep supervision heads.
                       Only the last `dsdepth` levels will produce an output head.
        upsample_mode (str): Upsampling method. Default is "deconv".
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        latent_channels: int = 512,
        out_channels: int = 2,
        act: tuple | str = "relu",
        norm: tuple | str = "batch",
        blocks_up: tuple = (1, 1, 1),
        dsdepth: int = 1,
        upsample_mode: str = "deconv",
        resolution: tuple | None = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.dsdepth = max(dsdepth, 1)
        self.resolution = resolution

        anisotropic_scales = None
        if resolution:
            anisotropic_scales = scales_for_resolution(
                resolution, n_stages=len(blocks_up) + 1
            )
        self.anisotropic_scales = anisotropic_scales

        # Prepare activation and normalization configurations.
        act = split_args(act)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)

        n_up = len(blocks_up)
        filters = latent_channels

        self.up_layers = nn.ModuleList()
        for i in range(n_up):
            kernel_size, _, stride = (
                aniso_kernel(anisotropic_scales[len(blocks_up) - i - 1])
                if anisotropic_scales
                else (3, 1, 2)
            )

            level = nn.ModuleDict()
            level["upsample"] = UpSample(
                mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=filters,
                out_channels=filters // 2,
                kernel_size=kernel_size,
                scale_factor=stride,
                bias=False,
                align_corners=False,
            )

            lite_blocks = []
            for _ in range(blocks_up[i]):
                lite_blocks.append(
                    nn.Sequential(
                        Conv[Conv.CONV, spatial_dims](
                            in_channels=filters // 2,
                            out_channels=filters // 2,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                        get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=filters // 2),
                        get_act_layer(act)
                    )
                )
            level["blocks"] = nn.Sequential(*lite_blocks)

            if i >= n_up - dsdepth:
                level["head"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters // 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=True,
                )
            else:
                level["head"] = nn.Identity()

            self.up_layers.append(level)
            filters = filters // 2  # Update the number of channels for the next stage.

    def forward(self, out_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            out_flat (torch.Tensor): A 1D latent feature vector with shape [latent_channels].

        Returns:
            torch.Tensor: The decoded output. For deep supervision, the last head output is returned.
        """
        x = out_flat

        outputs = []
        for level in self.up_layers:
            x = level["upsample"](x)
            x = level["blocks"](x)
            # If this level has a head (for deep supervision), get its output.
            if not isinstance(level["head"], nn.Identity):
                outputs.append(level["head"](x))

        # If deep supervision is used, return the output from the last head;
        # otherwise, simply return the final tensor.
        if outputs:
            return outputs[-1]
        return x


class VectorToTensor(nn.Module):
    """
    Projects a 1D latent vector into a 4D/5D tensor with spatial dimensions.

    For a 3D image, this transforms a vector of size `latent_dim` into a tensor
    with shape [batch, out_channels, D, H, W]. In this example, we assume the target
    shape (excluding the batch dimension) is (out_channels, 2, 16, 16).

    Args:
        latent_dim (int): Dimensionality of the latent vector (e.g., 512).
        target_shape (tuple): The target output shape excluding the batch dimension.
                              For example, (64, 2, 16, 16) where 64 is the number of channels.
    """

    def __init__(self, latent_dim: int, target_shape: tuple):
        super().__init__()
        self.target_shape = target_shape
        target_numel = 1
        for dim in target_shape:
            target_numel *= dim
        self.fc = nn.Linear(latent_dim, target_numel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A latent feature vector of shape [latent_dim] or [batch, latent_dim].

        Returns:
            torch.Tensor: A tensor of shape [batch, *target_shape].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)
        x = x.view(x.size(0), *self.target_shape)
        return x
