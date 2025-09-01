from __future__ import annotations

import math
from collections import defaultdict
from typing import Type

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage as ndi
from tqdm import tqdm

from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v1.main import \
    dice_loss
from unicorn_eval.adaptors.segmentation.baseline_segmentation_upsampling_3d.v1 import \
    SegmentationUpsampling3D
from unicorn_eval.adaptors.segmentation.data_handling import (
    construct_data_with_labels, load_patch_data)
from unicorn_eval.adaptors.segmentation.inference import create_grid


class UpsampleConvSegAdaptor(nn.Module):
    def __init__(self, target_shape=None, in_channels=32, num_classes=2):
        super().__init__()
        self.target_shape = target_shape
        self.in_channels = in_channels
        # Two intermediate conv layers + final prediction layer
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        C = self.in_channels
        B, feat_len = x.shape
        if feat_len % C != 0:
            raise ValueError(
            f"[Adaptor] Embedding length {feat_len} must be divisible by in_channels={C}."
        )

        flat = feat_len // C

        if self.target_shape is None:
            raise ValueError("[Adaptor] target_shape must be specified.")

        D_ref, H_ref, W_ref = self.target_shape
        ref_ratio = D_ref * H_ref * W_ref

        k = (flat / ref_ratio) ** (1 / 3)
        D = round(D_ref * k)
        H = round(H_ref * k)
        W = round(W_ref * k)
        
        if D * H * W != flat:
            D, H, W = exact_triplet_from_ref(flat, (D_ref, H_ref, W_ref))
        
        x = x.view(B, C, D, H, W)
        x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        x = self.conv_blocks(x)
        return x


class ConvUpsampleSegAdaptor(nn.Module):
    def __init__(self, target_shape=None, in_channels=32, num_classes=2):
        super().__init__()
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        C = self.in_channels
        B, feat_len = x.shape
        if feat_len % C != 0:
            raise ValueError(
            f"[Adaptor] Embedding length {feat_len} must be divisible by in_channels={C}."
        )

        flat = x.shape[1] // C

        if self.target_shape is None:
            raise ValueError("[Adaptor] target_shape must be specified.")

        D_ref, H_ref, W_ref = self.target_shape
        ref_ratio = D_ref * H_ref * W_ref

        k = (flat / ref_ratio) ** (1 / 3)

        D = round(D_ref * k)
        H = round(H_ref * k)
        W = round(W_ref * k)

        if D * H * W != flat:
            D, H, W = exact_triplet_from_ref(flat, (D_ref, H_ref, W_ref))

        x = x.view(B, C, D, H, W)
        x = self.conv_blocks(x)
        x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        return x


class LinearUpsampleConv3D_V2(SegmentationUpsampling3D):
    """
    Patch-level adaptor that performs segmentation by linearly upsampling 
    3D patch-level features followed by convolutional refinement.

    This adaptor takes precomputed patch-level features from 3D medical images
    and predicts voxel-wise segmentation by applying a simple decoder that:
    1) linearly upsamples the patch embeddings to the original resolution, and
    2) passes them through 3D convolution layers for spatial refinement.

    Steps:
    1. Extract patch-level segmentation labels using spatial metadata.
    2. Construct training data from patch features and coordinates.
    3. Train a lightweight 3D decoder that linearly upsamples features and refines them with convolution layers.
    4. At inference, apply the decoder to test patch features and reconstruct full-size segmentation predictions.

    Args:
        shot_features : Patch-level feature embeddings of few-shot labeled volumes.
        shot_labels : Full-resolution segmentation labels (used to supervise the decoder).
        shot_coordinates : Patch coordinates corresponding to shot_features.
        shot_names : Case identifiers for few-shot examples.
        test_features : Patch-level feature embeddings for testing.
        test_coordinates : Patch coordinates corresponding to test_features.
        test_names : Case identifiers for testing examples.
        test_image_sizes, test_image_origins, test_image_spacings, test_image_directions:
            Metadata for reconstructing the spatial layout of test predictions.
        shot_image_spacing, shot_image_origins, shot_image_directions:
            Metadata used to align segmentation labels with patch features during training.
        patch_size : Size of each 3D patch.
        return_binary : Whether to threshold predictions into binary segmentation masks.
    """
    def __init__(self, *args, decoder_cls: Type[nn.Module] = UpsampleConvSegAdaptor, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_task11 = False
        self.is_task06 = False
        self.decoder_cls = decoder_cls

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

        train_loader = load_patch_data(train_data, batch_size=1, balance_bg=self.balance_bg)

        max_class = max_class_label_from_labels(self.shot_labels)
        if max_class >= 100:
            self.is_task11 = True
            num_classes = 4
        elif max_class > 1:
            self.is_task06 = True
            num_classes = 2
        else:
            num_classes = max_class + 1

        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = self.decoder_cls(
            target_shape=self.global_patch_size[::-1],  # (D, H, W)
            num_classes=num_classes,
        )

        print(f"Training decoder with {num_classes} classes")
        decoder.to(self.device)
        self.decoder = train_seg_adaptor3d(decoder, train_loader, self.device, is_task11=self.is_task11, is_task06=self.is_task06)

    def predict(self) -> list:
        # build test data and loader
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

        # wrong patch spacing
        for data in test_data:
            data['patch_spacing'] = data['image_spacing']

        test_loader = load_patch_data(test_data, batch_size=1)

        # run inference using the trained decoder
        return inference3d_softmax(
            decoder=self.decoder,
            data_loader=test_loader,
            device=self.device,
            return_binary=self.return_binary,
            test_cases=self.test_cases,
            test_label_sizes=self.test_label_sizes,
            test_label_spacing=self.test_label_spacing,
            test_label_origins=self.test_label_origins,
            test_label_directions=self.test_label_directions,
            is_task11=self.is_task11
        )


def expand_instance_labels(y: np.ndarray) -> np.ndarray:
    """
    Reverse-expand class labels to instance labels.

    Input y uses:
      - 0: background
      - 1: class-A (-> instances 1..99)
      - 2: class-B (-> 100)
      - 3: class-C (-> instances 201..)

    Rules:
      - label==1 -> connected components -> 1,2,3,... up to 99 (100th+ capped at 99)
      - label==2 -> 100
      - label==3 -> connected components -> 201,202,...
      - else     -> 0
    """
    y = y.astype(np.int64, copy=False)
    out = np.zeros_like(y, dtype=np.int64)

    # Connectivity: 2D=8, 3D=26
    structure = np.ones((3,) * y.ndim, dtype=np.uint8)

    # --- label==1 ---
    mask1 = (y == 1)
    if np.any(mask1):
        lbl1, n1 = ndi.label(mask1, structure=structure)
        next_lab = 1
        for cid in range(1, n1 + 1):
            blob = (lbl1 == cid)
            if not np.any(blob):
                continue
            assign = next_lab if next_lab <= 99 else 99
            out[blob] = assign
            next_lab += 1

    # --- label==2 ---
    out[y == 2] = 100

    # --- label==3 ---
    mask3 = (y == 3)
    if np.any(mask3):
        lbl3, n3 = ndi.label(mask3, structure=structure)
        base = 201
        for cid in range(1, n3 + 1):
            blob = (lbl3 == cid)
            if not np.any(blob):
                continue
            out[blob] = base
            base += 1

    return out


def inference3d_softmax(*, decoder, data_loader, device, return_binary, test_cases, test_label_sizes, test_label_spacing, test_label_origins, test_label_directions, is_task11=False):
    decoder.eval()
    with torch.no_grad():
        grouped_predictions = defaultdict(lambda: defaultdict(list))

        for batch in tqdm(data_loader, desc="Inference"):
            inputs = batch["patch"].to(device)  # shape: [B, ...]
            coords = batch["coordinates"]  # list of 3 tensors
            image_idxs = batch["case_number"]

            outputs = decoder(inputs)  # shape: [B, ...]
            probs = torch.softmax(outputs, dim=1)

            if return_binary:
                pred_mask = torch.argmax(probs, dim=1, keepdim=True).float()
            else:
                pred_mask = probs[:, 1:]

            batch["image_origin"] = batch["image_origin"]
            batch["image_spacing"] = batch["image_spacing"]
            for i in range(len(image_idxs)):
                image_id = int(image_idxs[i])
                coord = tuple(
                    float(c) for c in coords[i]
                )  # convert list to tuple for use as dict key
                grouped_predictions[image_id][coord].append(
                    {
                        "features": pred_mask[i].cpu().numpy(),
                        "patch_size": [
                            int(batch["patch_size"][j][i])
                            for j in range(len(batch["patch_size"]))
                        ],
                        "patch_spacing": [
                            float(batch["patch_spacing"][j][i])
                            for j in range(len(batch["patch_spacing"]))
                        ],
                        "image_size": [
                            int(batch["image_size"][j][i])
                            for j in range(len(batch["image_size"]))
                        ],
                        "image_origin": [
                            float(batch["image_origin"][j][i])
                            for j in range(len(batch["image_origin"]))
                        ],
                        "image_spacing": [
                            float(batch["image_spacing"][j][i])
                            for j in range(len(batch["image_spacing"]))
                        ],
                        "image_direction": [
                            float(batch["image_direction"][j][i])
                            for j in range(len(batch["image_direction"]))
                        ],
                    }
                )

        averaged_patches = defaultdict(list)

        for image_id, coord_dict in grouped_predictions.items():
            for coord, patches in coord_dict.items():
                all_features = [p["features"] for p in patches]
                stacked = np.stack(all_features, axis=0)
                avg_features = np.mean(stacked, axis=0)

                averaged_patches[image_id].append(
                    {
                        "coord": list(coord),
                        "features": avg_features,
                        "patch_size": patches[0]["patch_size"],
                        "patch_spacing": patches[0]["patch_spacing"],
                        "image_size": patches[0]["image_size"],
                        "image_origin": patches[0]["image_origin"],
                        "image_spacing": patches[0]["image_spacing"],
                        "image_direction": patches[0]["image_direction"],
                    }
                )
        
        grids = create_grid(averaged_patches)

        aligned_preds = {}

        for case_id, pred_msk in grids.items():
            case = test_cases[case_id]
            gt_size = test_label_sizes[case]
            gt_spacing = test_label_spacing[case]
            gt_origin = test_label_origins[case]
            gt_direction = test_label_directions[case]

            pred_on_gt = sitk.Resample(
                pred_msk,
                gt_size,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                gt_origin,
                gt_spacing,
                gt_direction
            )
            
            if is_task11:
                pred_on_gt_arr = sitk.GetArrayFromImage(pred_on_gt)
                aligned_preds[case_id] = expand_instance_labels(pred_on_gt_arr)
            else:
                aligned_preds[case_id] = sitk.GetArrayFromImage(pred_on_gt)
            
        return [j for j in aligned_preds.values()]


def max_class_label_from_labels(label_patch_features) -> int:
    """
    Find the maximum class label across all patches.
    Returns the maximum label value, or 0 if none found.
    """
    mx = -1
    for case in label_patch_features:
        for p in case.get("patches", []):
            a = np.asarray(p.get("features", ()))
            if a.size == 0:
                continue
            v = np.nanmax(a)
            if np.isfinite(v) and v > mx:
                mx = int(v)
    return mx if mx >= 0 else 0


def remap_task11_labels(label_patch_features):
    """
    Remap feature labels in-place if this is Task 11.

    Input
    -----
    label_patch_features : np.array(dtype=object)
        Array of "cases". Each case is a dict with:
          - 'patches': list of dicts, where each dict has:
              - 'features': np.ndarray (e.g., shape 128x128x16)

    Logic
    -----
    1) Determine Task 11:
       - Scan all feature arrays and compute a global maximum.
       - If global_max >= 100, treat as Task 11 and apply remapping.

    2) Remapping rules (apply only when Task 11):
       - values in (0, 100)  -> 1
       - values == 100       -> 2
       - values > 200        -> 3
       - all other values (e.g., 0, 101–200, 200) remain unchanged.

    Returns
    -------
    dict with keys:
      - 'is_task11': bool
      - 'global_max': float or int or None
      - 'changed_patches': int, number of patches updated
    """
    # --- Step 1: compute global maximum across all features ---
    global_max = None
    for case in label_patch_features:
        for p in case.get("patches", []):
            feats = p.get("features", None)
            if feats is None:
                continue
            arr = np.asarray(feats)
            if arr.size == 0:
                continue
            m = arr.max()
            global_max = m if global_max is None else max(global_max, m)

    is_task11 = (global_max is not None) and (global_max >= 100)
    if not is_task11:
        return {"is_task11": False, "global_max": global_max, "changed_patches": 0}

    # --- Step 2: apply in-place remapping for Task 11 ---
    changed = 0
    for case in label_patch_features:
        for p in case.get("patches", []):
            feats = p.get("features", None)
            if feats is None:
                continue
            arr = np.asarray(feats)
            if arr.size == 0:
                continue

            orig_dtype = arr.dtype
            mapped = arr.copy()

            # Build masks for each rule
            mask1 = (mapped > 0) & (mapped < 100)
            mask2 = (mapped == 100)
            mask3 = (mapped > 200)

            if mask1.any() or mask2.any() or mask3.any():
                mapped[mask1] = 1
                mapped[mask2] = 2
                mapped[mask3] = 3

                # Preserve original dtype and write back
                p["features"] = mapped.astype(orig_dtype, copy=False)
                changed += 1

    return {"is_task11": True, "global_max": global_max, "changed_patches": changed}


def map_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Rules:
      - y == 100      -> 2
      - 1 <= y <= 99  -> 1
      - y >= 201      -> 3
      - else          -> 0
    """

    y_new = torch.zeros_like(y)

    y_new = torch.where(y == 100, 2, y_new)
    y_new = torch.where((y >= 1) & (y <= 99), 1, y_new)
    y_new = torch.where(y >= 201, 3, y_new)

    return y_new


def train_seg_adaptor3d(decoder, data_loader, device, num_epochs = 3, iterations_per_epoch: int | None = None, is_task11=False, is_task06=False, verbose: bool = True):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0

        # batch progress
        batch_iter = tqdm(data_loader, total=iterations_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not verbose)
        iteration_count = 0
    
        for batch in batch_iter:
            iteration_count += 1

            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"].to(device).long()

            if is_task11 or is_task06:
                patch_label = map_labels(patch_label)

            optimizer.zero_grad()
            de_output = decoder(patch_emb) 

            ce = ce_loss(de_output, patch_label) 
            if is_task06:
                loss = ce
            else:
                dice = dice_loss(de_output, patch_label)
                loss = ce + dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with current loss and running average
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss / iteration_count:.4f}")

            if iterations_per_epoch is not None and iteration_count >= iterations_per_epoch:
                break

        tqdm.write(f"Epoch {epoch+1}: Avg total loss = {epoch_loss / iteration_count:.4f}")

    return decoder


def exact_triplet_from_ref(flat: int, ref: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Find integers (D,H,W) with D*H*W == flat, close to the heuristic proportions
    implied by 'ref' (D_ref,H_ref,W_ref).
    """

    D_ref, H_ref, W_ref = ref
    ref_ratio = max(1, D_ref * H_ref * W_ref)
    k = (flat / ref_ratio) ** (1.0 / 3.0)

    # helper: all divisors of n
    def divisors(n: int):
        ds = []
        r = int(math.isqrt(n))
        for d in range(1, r + 1):
            if n % d == 0:
                ds.append(d)
                q = n // d
                if q != d:
                    ds.append(q)
        return sorted(ds)

    # pick W as a divisor of 'flat' closest to W_ref*k
    w_heur = max(1, int(round(W_ref * k)))
    w_candidates = divisors(flat)
    W = min(w_candidates, key=lambda w: abs(w - w_heur))

    base = flat // W  # now need D*H = base

    # pick D as a divisor of 'base' closest to D_ref*k; H follows
    d_heur = max(1, int(round(D_ref * k)))
    d_candidates = divisors(base)
    D = min(d_candidates, key=lambda d: abs(d - d_heur))
    H = base // D

    # ensure non-zero
    D = max(1, D); H = max(1, H); W = max(1, W)
    assert D * H * W == flat, f"factorization failed: {D}*{H}*{W} != {flat}"
    return D, H, W
