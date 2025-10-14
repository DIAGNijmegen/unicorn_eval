import os
import itertools
import pickle
from pathlib import Path
from typing import Iterable
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import SimpleITK as sitk
with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
    from picai_prep.preprocessing import crop_or_pad, resample_img

from unicorn_eval.adaptors.reconstruct_prediction import stitch_patches_fast
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, extract_patch_labels)
from unicorn_eval.metrics.uls import dice_coefficient


def pad_image(image: sitk.Image, patch_size: Iterable[int]) -> sitk.Image:
    """
    Pads the input image symmetrically so its dimensions become divisible by the specified patch size.

    Padding is performed using the minimum intensity value found in the original image, ensuring padded regions blend naturally with background values.

    Args:
        image (sitk.Image): Input 3D image to be padded.
        patch_size (list[int]): Desired patch size as [x, y, z]. After padding, each dimension of the image will be divisible by the corresponding element in this list.

    Returns:
        sitk.Image: Padded SimpleITK image with dimensions divisible by `patch_size`.
    """
    size = image.GetSize()
    pad = [(p - s % p) % p for s, p in zip(size, patch_size)]
    new_size = tuple(s + pad_dim for s, pad_dim in zip(size, pad))

    image = crop_or_pad(
        image=image,
        size=new_size[::-1],
        pad_only=True,
    )

    return image


def extract_patches(
    image: sitk.Image,
    patch_size: Iterable[int],
    spacing: Iterable[float] | None = None,
) -> tuple[list[sitk.Image], list[tuple]]:
    """
    Extracts uniformly sized patches from a 3D SimpleITK image, optionally resampling it to a specified voxel spacing before extraction.

    If `spacing` is provided, the image is first resampled using linear interpolation to achieve the specified spacing. The image is then padded so that its dimensions become exactly divisible by the given patch size. Patches are extracted systematically, covering the entire image volume without overlap or gaps.

    Args:
        image (sitk.Image): Input 3D image from which to extract patches.
        patch_size (list[int]): Patch size as [x, y, z], defining the dimensions of each extracted patch.
        spacing (list[float] | None, optional): Desired voxel spacing as [x, y, z]. If provided, the image will be resampled to this spacing before patch extraction. Defaults to None.

    Returns:
        - patches (list[sitk.Image]): List of extracted image patches.
        - coordinates (list[tuple]): List of coordinates for each patch, formatted as:
            ((x_start, y_start, z_start), (x_end, y_end, z_end)) in world coordinates.
            Each coordinate pair represents the start and end points of the patch in the original image's physical space.
    """
    if spacing is not None and not np.isclose(image.GetSpacing(), spacing).all():
        # resample image to specified spacing
        image = resample_img(
            image=image,
            out_spacing=spacing[::-1],
            interpolation=sitk.sitkLinear,
        )
        print(f"Resampled image to spacing: {spacing}. Image size: {image.GetSize()}")

    # pad image to fit patch size
    image = pad_image(
        image=image,
        patch_size=patch_size,
    )

    # generate patch coordinates (x, y, z)
    image_size = image.GetSize()
    steps = [range(0, image_size[dim], patch_size[dim]) for dim in range(3)]

    patches = []
    coordinates = []
    for x, y, z in itertools.product(*steps):
        start_coords = (x, y, z)
        patch = sitk.RegionOfInterest(image, patch_size, start_coords)
        patches.append(patch)
        matrix_coordinates = (
            (x, y, z),
            (x + patch_size[0], y + patch_size[1], z + patch_size[2]),
        )
        world_coordinates = tuple(
            image.TransformIndexToPhysicalPoint(coord) for coord in matrix_coordinates
        )
        coordinates.append(world_coordinates)

    return patches, coordinates, image


def test_end_to_end_data_handling(
    task_results_path: Path,
    label_path_pattern: Path,
):
    """
    End-to-end test for data handling pipeline including:
    1. Load task_results from pickle
    2. Extract patch labels using extract_patch_labels
    3. Construct data with labels using construct_data_with_labels  
    4. Stitch patches using stitch_patches_fast
    5. Compare reconstructed labels with original labels
    """
    # Step 1: Load task results from pickle file
    if not task_results_path.exists():
        raise FileNotFoundError(f"Task results file not found: {task_results_path}")

    with open(task_results_path, "rb") as f:
        task_results = pickle.load(f)

    # Extract required data from task_results
    case_ids = task_results["case_ids"]

    # Test on first few cases
    cases_to_test = range(len(case_ids))

    for case_idx in cases_to_test:
        case_name = case_ids[case_idx]

        # unpack metadata
        image_size = task_results["cases_image_sizes"][case_name]
        image_spacing = task_results["cases_image_spacings"][case_name]
        image_origin = task_results["cases_image_origins"][case_name]
        image_direction = task_results["cases_image_directions"][case_name]
        label_spacing = task_results["cases_label_spacings"][case_name]
        label_origin = task_results["cases_label_origins"][case_name]
        label_direction = task_results["cases_label_directions"][case_name]
        patch_coordinates = task_results["cases_coordinates"][case_idx]
        patch_size = task_results["cases_patch_sizes"][case_name]
        patch_spacing = task_results["cases_patch_spacings"][case_name]
        original_label = task_results["case_labels"][case_idx]

        # Step A: convert original label to SimpleITK image
        original_label_image = sitk.GetImageFromArray(original_label)
        original_label_image.SetSpacing(label_spacing)
        original_label_image.SetOrigin(label_origin)
        original_label_image.SetDirection(label_direction)
        output_path = Path(__file__).parent / f"original_label_{case_name}.nii.gz"
        sitk.WriteImage(original_label_image, output_path)

        # Step B: choose coordinates, patch size and spacing based on the original label
        _, coordinates, _ = extract_patches(
            image=original_label_image,
            patch_size=patch_size,
            spacing=patch_spacing,
        )
        # patch_coordinates = [c[0] for c in coordinates]

        # Step 2: Extract patch labels using extract_patch_labels
        patch_labels_dict = extract_patch_labels(
            label=original_label,
            label_spacing=label_spacing,
            label_origin=label_origin,
            label_direction=label_direction,
            image_size=image_size,
            image_spacing=image_spacing,
            image_origin=image_origin,
            image_direction=image_direction,
            start_coordinates=patch_coordinates,
            patch_size=patch_size,
            patch_spacing=patch_spacing,
        )

        # check extracted label
        for patch_data in patch_labels_dict["patches"]:
            # Modify patch_data to match expected format
            patch_data["coord"] = patch_data["coordinates"]
            patch_data["patch_spacing"] = patch_spacing
            patch_data["patch_size"] = patch_size
            patch_data["image_direction"] = image_direction

        reconstructed_label = stitch_patches_fast(patch_labels_dict["patches"])
        sitk.WriteImage(reconstructed_label, Path(__file__).parent / f"extracted_label_{case_name}.nii.gz")

        # Step 3: Construct data with labels using construct_data_with_labels
        data_array = construct_data_with_labels(
            coordinates=[np.asarray(patch_coordinates)],
            embeddings=[np.random.rand(len(patch_coordinates), 32)],
            case_ids=[case_name],
            patch_sizes={case_name: patch_size},
            patch_spacings={case_name: patch_spacing},
            labels=[patch_labels_dict],
            image_sizes=task_results["cases_image_sizes"],
            image_origins=task_results["cases_image_origins"],
            image_spacings=task_results["cases_image_spacings"],
            image_directions=task_results["cases_image_directions"],
        )

        # Prepare case data for stitching in the format expected by stitch_patches_fast for ALL cases
        case_data = {}
        for patch_data in data_array:
            # Modify patch_data to match expected format
            patch_data["features"] = patch_data["patch_label"]
            patch_data["coord"] = patch_data["coordinates"]
            case_number = patch_data["case_number"]
            if case_number not in case_data:
                case_data[case_number] = []
            case_data[case_number].append(patch_data)

        # Check that all cases have positive patches (non-zero labels)
        for case_number, patches in case_data.items():
            print(f"Case {case_name} ({case_idx=}):")

            positive_patch_count = 0
            total_patches = len(patches)

            for patch in patches:
                patch_label = patch["features"]
                if np.any(patch_label > 0):
                    positive_patch_count += 1

            print(f"  Total patches: {total_patches}")
            print(f"  Positive patches: {positive_patch_count}")
            print(f"  Positive patch ratio: {positive_patch_count/total_patches:.2%}")

            if positive_patch_count == 0:
                raise ValueError(f"No positive patches found for case {case_name}")

        # Process each case individually
        for case_number in case_data.keys():
            print(f"Case {case_name} ({case_idx=}):")

            # Step 4: Stitch patches using stitch_patches_fast
            reconstructed_label = stitch_patches_fast(case_data[case_number])

            # Save reconstructed label for manual inspection
            output_path = Path(__file__).parent / f"reconstructed_label_{case_name}.nii.gz"
            sitk.WriteImage(reconstructed_label, str(output_path))
            print(f"  Saved reconstructed label: {output_path}")

            # Step 5: Compare reconstructed label with original label
            reconstructed_array = sitk.GetArrayFromImage(reconstructed_label)

            # Load original label for comparison
            ground_truth_path = Path(label_path_pattern.as_posix().format(case_id=case_name))

            if not ground_truth_path.exists():
                raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

            original_sitk = sitk.ReadImage(str(ground_truth_path))

            # directly resample the original label to the reconstructed label's space
            original_resampled = sitk.Resample(original_sitk, reconstructed_label)
            original_resampled_array = sitk.GetArrayFromImage(original_resampled)

            output_path = Path(__file__).parent / f"original_resampled_label_{case_name}.nii.gz"
            sitk.WriteImage(original_resampled, str(output_path))
            print(f"  Saved original resampled label: {output_path}")

            # Check if shapes match
            print(f"\nCase {case_name} reconstruction comparison:")
            print(f"  Original shape: {original_sitk.GetSize()}")
            print(f"  Reconstructed shape: {reconstructed_label.GetSize()}")
            print(f"  Original resampled shape: {original_resampled.GetSize()}")

            # Compare image properties
            print(f"  Original spacing: {original_sitk.GetSpacing()}")
            print(f"  Reconstructed spacing: {reconstructed_label.GetSpacing()}")
            print(f"  Original resampled spacing: {original_resampled.GetSpacing()}")

            print(f"  Original direction: {original_sitk.GetDirection()}")
            print(f"  Reconstructed direction: {reconstructed_label.GetDirection()}")
            print(f"  Original resampled direction: {original_resampled.GetDirection()}")

            print(f"  Original origin: {original_sitk.GetOrigin()}")
            print(f"  Reconstructed origin: {reconstructed_label.GetOrigin()}")
            print(f"  Original resampled origin: {original_resampled.GetOrigin()}")

            # Check overlap/similarity if shapes match
            if original_resampled_array.shape == reconstructed_array.shape:
                # Calculate Dice coefficient
                dice = dice_coefficient(original_resampled_array, reconstructed_array)
                print(f"  Dice coefficient: {dice:.4f}")

                # For debugging: check if there's an orientation issue
                if dice < 0.8:
                    print("  WARNING: Low Dice coefficient - possible orientation issue")

                    # Try different axis flips to check for orientation issues
                    for axis in range(3):
                        flipped = np.flip(reconstructed_array, axis=axis)
                        intersection_flip = np.sum(original_resampled_array * flipped)
                        dice_flip = 2 * intersection_flip / (np.sum(original_resampled_array) + np.sum(flipped)) if (np.sum(original_resampled_array) + np.sum(flipped)) > 0 else 0
                        if dice_flip > dice:
                            print(f"    Flipping axis {axis} improves Dice to {dice_flip:.4f}")

                    raise ValueError("  ERROR: Low Dice coefficient - possible orientation issue")
            else:
                raise ValueError("  ERROR: Shape mismatch between original and reconstructed labels")

        print("+=+" * 20)


if __name__ == "__main__":
    test_end_to_end_data_handling(
        task_results_path=Path(__file__).parent / "task_results_Task10_segmenting_lesions_within_vois_in_ct.pkl",
        label_path_pattern=Path(__file__).parent.parent / "vision" / "ground_truth-task10-val" / "Task10_segmenting_lesions_within_vois_in_ct" / r"{case_id}" / "images" / "ct-binary-uls" / f"{{case_id}}.mha"
    )

    test_end_to_end_data_handling(
        task_results_path=Path(__file__).parent / "task_results_Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri.pkl",
        label_path_pattern=Path(__file__).parent.parent / "vision" / "ground_truth-task11-val" / "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri" / r"{case_id}" / "images" / "sagittal-spine-mr-segmentation" / f"{{case_id}}.mha"
    )
