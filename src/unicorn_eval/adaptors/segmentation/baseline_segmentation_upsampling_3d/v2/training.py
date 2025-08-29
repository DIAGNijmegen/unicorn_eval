from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v1.main import \
    dice_loss
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2.main import \
    map_labels


def train_seg_adaptor3d_v2(decoder, data_loader, device, num_iterations = 5_000, is_task11=False, is_task06=False, verbose: bool = True):
    # Use weighted CrossEntropyLoss and focal loss components
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)

    decoder.train()

    epoch_loss = 0.0
    iteration_count = 0
    epoch_iterations = 0

    # Create an infinite iterator over the data loader
    data_iter = iter(data_loader)

    # Progress bar for total iterations
    progress_bar = tqdm(total=num_iterations, desc="Training", disable=not verbose)

    # Train decoder
    while iteration_count < num_iterations:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator when data loader is exhausted
            data_iter = iter(data_loader)
            batch = next(data_iter)

        iteration_count += 1
        epoch_iterations += 1

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

        # Gradient clipping to prevent exploding gradients
        clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        optimizer.step()

        step_loss = loss.item()
        epoch_loss += step_loss

        # Update progress bar with current loss and running average
        progress_bar.set_postfix(loss=f"{step_loss:.5e}", avg=f"{epoch_loss / epoch_iterations:.5e}")
        progress_bar.update(1)

        if iteration_count % 100 == 0:
            avg_loss = epoch_loss / epoch_iterations
            tqdm.write(f"Iteration {iteration_count}: avg_loss={avg_loss:.5e}")

            epoch_loss = 0.0
            epoch_iterations = 0

    progress_bar.close()

    return decoder

