import pickle
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from unicorn_eval.adaptors.reconstruct_prediction import stitch_patches_fast
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, extract_patch_labels)
from unicorn_eval.metrics.uls import dice_coefficient


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
    case_labels = task_results["case_labels"]
    case_ids = task_results["case_ids"]
    case_image_sizes = task_results["cases_image_sizes"]
    case_image_spacings = task_results["cases_image_spacings"]
    case_image_origins = task_results["cases_image_origins"]
    case_image_directions = task_results["cases_image_directions"]
    case_label_spacings = task_results["cases_label_spacings"]
    case_label_origins = task_results["cases_label_origins"]
    case_label_directions = task_results["cases_label_directions"]
    shot_coordinates = task_results["shot_coordinates"]
    patch_size = task_results["patch_size"]
    patch_spacing = task_results["patch_spacing"]
    
    if not case_labels or not case_ids:
        pytest.skip("Required case data not found in task_results")
    
    # Test on first few cases
    num_test_cases = min(3, len(case_ids))
    
    for case_idx in range(num_test_cases):
        case_id = case_ids[case_idx]
        original_label = case_labels[case_idx]
        
        # Step 2: Extract patch labels using extract_patch_labels
        patch_labels_dict = extract_patch_labels(
            label=original_label,
            label_spacing=case_label_spacings[case_id],
            label_origin=case_label_origins[case_id],
            label_direction=case_label_directions[case_id],
            image_size=case_image_sizes[case_id],
            image_spacing=case_image_spacings[case_id],
            image_origin=case_image_origins[case_id],
            image_direction=case_image_directions[case_id],
            start_coordinates=shot_coordinates[case_idx],
            patch_size=patch_size,
            patch_spacing=patch_spacing,
        )

        # Step 3: Construct data with labels using construct_data_with_labels
        coords = shot_coordinates[case_idx]
        coords_array = np.array(coords)
        
        data_array = construct_data_with_labels(
            coordinates=[coords_array],
            embeddings=[np.random.rand(len(coords), 512)],
            case_names=[case_id],
            patch_size=patch_size,
            patch_spacing=patch_spacing,
            labels=[patch_labels_dict],
            image_sizes=case_image_sizes,
            image_origins=case_image_origins,
            image_spacings=case_image_spacings,
            image_directions=case_image_directions,
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
        case_id = case_ids[case_number] if case_number < len(case_ids) else f"case_{case_number}"
        print(f"Case {case_id} (case_number={case_number}):")
        
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
            raise ValueError(f"No positive patches found for case {case_id}")
    
    # Process each case individually
    for case_number in case_data.keys():
        case_id = case_ids[case_number] if case_number < len(case_ids) else f"case_{case_number}"
        
        # Step 4: Stitch patches using stitch_patches_fast
        reconstructed_label = stitch_patches_fast(case_data[case_number])
        
        # Step 5: Compare reconstructed label with original label
        reconstructed_array = sitk.GetArrayFromImage(reconstructed_label)
        
        # Load original label for comparison
        ground_truth_path = Path(label_path_pattern.as_posix().format(case_id=case_id))

        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
        original_sitk = sitk.ReadImage(str(ground_truth_path))
        original_array = sitk.GetArrayFromImage(original_sitk)
        
        # directly resample the original label to the reconstructed label's space
        original_resampled = sitk.Resample(original_sitk, reconstructed_label)
        original_resampled_array = sitk.GetArrayFromImage(original_resampled)

        # Check if shapes match
        print(f"\nCase {case_id} reconstruction comparison:")
        print(f"  Original shape: {original_array.shape}")
        print(f"  Reconstructed shape: {reconstructed_array.shape}")
        print(f"  Original resampled shape: {original_resampled_array.shape}")

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
            if dice < 0.5:
                print("  WARNING: Low Dice coefficient - possible orientation issue")

                # Try different axis flips to check for orientation issues
                for axis in range(3):
                    flipped = np.flip(reconstructed_array, axis=axis)
                    intersection_flip = np.sum(original_resampled_array * flipped)
                    dice_flip = 2 * intersection_flip / (np.sum(original_resampled_array) + np.sum(flipped)) if (np.sum(original_resampled_array) + np.sum(flipped)) > 0 else 0
                    if dice_flip > dice:
                        print(f"    Flipping axis {axis} improves Dice to {dice_flip:.4f}")
        else:
            print("  ERROR: Shape mismatch between original and reconstructed labels")

        # Save reconstructed label for manual inspection
        output_path = Path(__file__).parent / f"reconstructed_label_{case_id}.nii.gz"
        sitk.WriteImage(reconstructed_label, str(output_path))
        print(f"  Saved reconstructed label: {output_path}")
        output_path = Path(__file__).parent / f"original_resampled_label_{case_id}.nii.gz"
        sitk.WriteImage(original_resampled, str(output_path))
        print(f"  Saved original resampled label: {output_path}")


if __name__ == "__main__":
    test_end_to_end_data_handling(
        task_results_path=Path(__file__).parent / "task_results_spider.pkl",
        label_path_pattern=Path(__file__).parent.parent / "vision" / "ground_truth-task11-val" / "Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri" / r"{case_id}" / "images" / "sagittal-spine-mr-segmentation" / f"{{case_id}}.mha"
    )
    # test_end_to_end_data_handling(
    #     task_results_path=Path(__file__).parent / "task_results_uls.pkl",
    #     label_path_pattern=Path(__file__).parent.parent / "vision" / "ground_truth-task10-val" / "Task10_segmenting_lesions_within_vois_in_ct" / r"{case_id}" / "images" / "ct-binary-uls" / f"{{case_id}}.mha"
    # )
