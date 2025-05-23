from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.nets.segresnet_ds import SegResEncoder, SegResBlock, aniso_kernel, scales_for_resolution
import copy
from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode, has_option



# Assume that these helper functions and modules (UpSample, SegResBlock, Conv, split_args, has_option, Norm, Act)
# are defined/imported from your project or MONAI.

class SegResNetDecoderOnly(nn.Module):
    """
    A decoder-only variant of SegResNetDS.
    
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
            anisotropic_scales = scales_for_resolution(resolution, n_stages=len(blocks_up)+1)
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
                aniso_kernel(anisotropic_scales[len(blocks_up) - i - 1]) if anisotropic_scales else (3, 1, 2)
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
            # Build a sequential block (repeat SegResBlock as many times as specified)
            blocks = [
                SegResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=filters // 2,
                    kernel_size=kernel_size,
                    norm=norm,
                    act=act,
                )
                for _ in range(blocks_up[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)
            
            # Add a deep supervision head if this level is within the last dsdepth levels.
            if i >= n_up - dsdepth:
                level["head"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters // 2, out_channels=out_channels, kernel_size=1, bias=True
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
