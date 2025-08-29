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

from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v1 import (
    SegmentationUpsampling, SegmentationUpsampling3D)
from unicorn_eval.adaptors.segmentation.data_handling import (
    SegmentationDataset, assign_mask_to_patch, construct_data_with_labels,
    construct_segmentation_labels, custom_collate, extract_patch_labels,
    load_patch_data, make_patch_level_neural_representation)
from unicorn_eval.adaptors.segmentation.decoders import (
    Decoder3D, SegmentationDecoder, SegResNetDecoderOnly, VectorToTensor,
    build_deconv_layers, compute_num_upsample_layers)
from unicorn_eval.adaptors.segmentation.inference import (create_grid,
                                                          inference,
                                                          inference3d,
                                                          world_to_voxel)
from unicorn_eval.adaptors.segmentation.mevis_conv_segmentation_3d.v1.main import (
    ConvDecoder3D, ConvSegmentation3D)
from unicorn_eval.adaptors.segmentation.training import (train_decoder,
                                                         train_decoder3d)

__all__ = [
    # Adaptors
    "SegmentationUpsampling", "SegmentationUpsampling3D", "ConvSegmentation3D",

    # Data handling
    "assign_mask_to_patch", "construct_segmentation_labels", "SegmentationDataset",
    "custom_collate", "construct_data_with_labels", "extract_patch_labels",
    "make_patch_level_neural_representation", "load_patch_data",

    # Decoders
    "compute_num_upsample_layers", "build_deconv_layers", "SegmentationDecoder",
    "Decoder3D", "SegResNetDecoderOnly", "VectorToTensor", "ConvDecoder3D",

    # Inference
    "inference", "world_to_voxel", "create_grid", "inference3d",

    # Training
    "train_decoder", "train_decoder3d",
]