from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from monai.data.dataloader import DataLoader
from monai.losses.dice import DiceCELoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v1.main import \
    dice_loss
from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2.main import \
    map_labels


def train_decoder3d_v2(
    decoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
    iterations_per_epoch: int | None = None,
    loss_fn=None,
    optimizer=None,
    label_mapper=None,
    verbose: bool = True
):
    if loss_fn is None:
        loss_fn = DiceCELoss(sigmoid=True)
    if optimizer is None:
        optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    # Train decoder
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0

        iteration_count = 0
        batch_iter = tqdm(data_loader, total=iterations_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not verbose)
        for batch in batch_iter:
            iteration_count += 1

            patch_emb = batch["patch"].to(device)
            patch_label = batch["patch_label"]
            if label_mapper is not None:
                patch_label = label_mapper(patch_label)
            patch_label = patch_label.to(device)

            optimizer.zero_grad()
            de_output = decoder(patch_emb)
            loss = loss_fn(de_output.squeeze(1), patch_label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with current loss and running average
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss / iteration_count:.4f}")

            if iterations_per_epoch is not None and iteration_count >= iterations_per_epoch:
                break

        tqdm.write(f"Epoch {epoch+1}: Avg total loss = {epoch_loss / iteration_count:.4f}")

    return decoder


def train_seg_adaptor3d_v2(
    decoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_iterations: int = 5_000,
    is_task11: bool = False,
    is_task06: bool = False,
    verbose: bool = True
):
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

