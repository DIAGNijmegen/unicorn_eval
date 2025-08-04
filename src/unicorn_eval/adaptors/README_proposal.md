# AIMHI_LinearUpsampleConv3D

## 📌 Description

`LinearUpsampleConv3D` is a patch-level adaptor for volumetric segmentation tasks.  
It reconstructs full-resolution voxel-wise segmentation maps from patch-level 3D features using a lightweight decoder.

The adaptor performs two main steps:

1. **Linear upsampling** of 3D patch embeddings to match the original image resolution.
2. **3D convolutional refinement** to produce smooth and spatially accurate segmentation outputs.

This method is especially useful in few-shot learning settings, where only limited patch-level labels are available.

## 🧠 Method Details

The adaptor workflow is as follows:

1. **Extract patch-level segmentation labels** using spatial metadata.
2. **Construct training samples** from patch features and physical coordinates.
3. **Train a 3D decoder** that:
   - Linearly upsamples patch features
   - Applies several convolutional layers for refinement
4. **Inference**:
   - The decoder maps patch-level features back to the original resolution.
   - Full-size 3D segmentation is reconstructed from predicted patches.

This adaptor inherits from the base class `PatchLevelTaskAdaptor` and supports end-to-end training and evaluation via the UNICORN framework.

## 🧪 Applicable Tasks
- **Patch-level feature to voxel-level segmentation tasks**


