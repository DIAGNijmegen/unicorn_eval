from typing import Iterable

import SimpleITK as sitk
from picai_prep.preprocessing import resample_img


def safe_region_of_interest(image: sitk.Image, patch_size: Iterable[int], start_index: Iterable[int], pad_value: int = 0) -> sitk.Image:
    size = image.GetSize()

    # Convert to mutable lists
    patch_size = list(patch_size)
    start_index = list(start_index)

    # Calculate how much padding is needed
    lower_pad = [max(0, -s) for s in start_index]
    upper_pad = [max(0, (s + p) - sz) for s, p, sz in zip(start_index, patch_size, size)]

    # Apply padding if needed
    if any(lower_pad) or any(upper_pad):
        image = sitk.ConstantPad(image, lower_pad, upper_pad, pad_value)
        size = image.GetSize()

        # Shift start index into padded image coordinates
        start_index = [s + lp for s, lp in zip(start_index, lower_pad)]

    # Finally extract ROI
    return sitk.RegionOfInterest(image, patch_size, start_index)


def extract_patches(
    image: sitk.Image,
    coordinates: Iterable[tuple[float, float, float]],
    patch_size: Iterable[int],
    spacing: Iterable[float] | None = None,
) -> list[sitk.Image]:
    """
    Extracts uniformly sized patches from a 3D SimpleITK image, optionally resampling it to a specified voxel spacing before extraction.
    
    If `spacing` is provided, the image is first resampled using linear interpolation to achieve the specified spacing. The image is then padded so that its dimensions become exactly divisible by the given patch size. Patches are extracted systematically, covering the entire image volume without overlap or gaps.

    Args:
        image (sitk.Image): Input 3D image from which to extract patches.
        coordinates (Iterable[tuple[float, float, float]]): Start coordinates for each patch in world coordinates. These are used to determine the physical location of each patch in the original image.
        patch_size (list[int]): Patch size as [x, y, z], defining the dimensions of each extracted patch.
        spacing (list[float] | None, optional): Desired voxel spacing as [x, y, z]. If provided, the image will be resampled to this spacing before patch extraction. Defaults to None.

    Returns:
        - patches (list[sitk.Image]): List of extracted image patches.
     """
    if spacing is not None and tuple(spacing) != image.GetSpacing():
        # resample image to specified spacing
        image = resample_img(
            image=image,
            out_spacing=spacing[::-1],
            interpolation=sitk.sitkLinear,
        )

    patches = []
    for coord in coordinates:
        # Convert start coordinate from world space to index space
        start_index = image.TransformPhysicalPointToIndex(coord)

        # Extract ROI
        patch = safe_region_of_interest(image, patch_size, start_index)
        patches.append(patch)

    return patches
