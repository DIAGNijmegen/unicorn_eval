import pickle
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from unicorn_eval.adaptors.reconstruct_prediction import stitch_patches_fast
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, extract_patch_labels)


def test_end_to_end_data_handling():
    """
    End-to-end test for data handling pipeline including:
    1. Load task_results from pickle
    2. Extract patch labels using extract_patch_labels
    3. Construct data with labels using construct_data_with_labels  
    4. Stitch patches using stitch_patches_fast
    5. Compare reconstructed labels with original labels
    """
    # Step 1: Load task results from pickle file
    task_results_path = Path(__file__).parent / "task_results.pkl"
    if not task_results_path.exists():
        pytest.skip(f"Task results file not found: {task_results_path}")
    
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
        
        # Prepare case data for stitching in the format expected by stitch_patches_fast
        case_data = {}
        for patch_data in data_array:
            if patch_data["case_number"] == 0:  # Only process first case in this iteration
                # Modify patch_data to match expected format
                patch_data["features"] = patch_data["patch_label"]
                patch_data["coord"] = patch_data["coordinates"]
                case_data[patch_data["case_number"]] = case_data.get(patch_data["case_number"], []) + [patch_data]
        
        if case_data:
            # Step 4: Stitch patches using stitch_patches_fast
            reconstructed_label = stitch_patches_fast(case_data[0])
            
            # Step 5: Compare reconstructed label with original label
            reconstructed_array = sitk.GetArrayFromImage(reconstructed_label)
            
            # Load original label for comparison
            ground_truth_path = Path(__file__).parent.parent / "vision" / "ground_truth-task10-val" / "Task10_segmenting_lesions_within_vois_in_ct" / case_id / "images" / "ct-binary-uls" / f"{case_id}.mha"
            
            if ground_truth_path.exists():
                original_sitk = sitk.ReadImage(str(ground_truth_path))
                original_array = sitk.GetArrayFromImage(original_sitk)
                
                # directly resample the original label to the reconstructed label's space
                original_resampled = sitk.Resample(original_sitk, reconstructed_label)
                original_resampled_array = sitk.GetArrayFromImage(original_resampled)

                # Check if shapes match
                print(f"Case {case_id}:")
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
                    intersection = np.sum(original_resampled_array * reconstructed_array)
                    dice = 2 * intersection / (np.sum(original_resampled_array) + np.sum(reconstructed_array))
                    print(f"  Dice coefficient: {dice:.4f}")
                    
                    # For debugging: check if there's an orientation issue
                    if dice < 0.5:
                        print("  WARNING: Low Dice coefficient - possible orientation issue")
                        
                        # Try different axis flips to check for orientation issues
                        for axis in range(3):
                            flipped = np.flip(reconstructed_array, axis=axis)
                            intersection_flip = np.sum(original_array * flipped)
                            dice_flip = 2 * intersection_flip / (np.sum(original_array) + np.sum(flipped))
                            if dice_flip > dice:
                                print(f"    Flipping axis {axis} improves Dice to {dice_flip:.4f}")
                else:
                    print("  ERROR: Shape mismatch between original and reconstructed labels")
            else:
                print(f"  Ground truth file not found: {ground_truth_path}")
                
            # Save reconstructed label for manual inspection
            output_path = Path(__file__).parent / f"reconstructed_label_{case_id}.nii.gz"
            sitk.WriteImage(reconstructed_label, str(output_path))
            print(f"  Saved reconstructed label: {output_path}")
        else:
            raise ValueError(f"No case data found for case number 0 in case {case_id}")


if __name__ == "__main__":
    test_end_to_end_data_handling()