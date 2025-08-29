from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from monai.data.dataloader import DataLoader
from monai.losses.dice import DiceCELoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from unicorn_eval.adaptors.segmentation.aimhi_linear_upsample_conv3d.v2.main import \
    map_labels


def train_decoder3d_v2(
    decoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_iterations: int = 5_000,
    loss_fn: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    label_mapper: Callable | None = None,
    is_task11: bool = False,
    is_task06: bool = False,
    verbose: bool = True,
):
    if loss_fn is None:
        if is_task06:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = DiceCELoss(sigmoid=True)

    if optimizer is None:
        optimizer = optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)

    if label_mapper is None and (is_task11 or is_task06):
        label_mapper = map_labels

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
        patch_label = batch["patch_label"]

        if label_mapper is not None:
            patch_label = label_mapper(patch_label)

        patch_label = patch_label.to(device).long()

        optimizer.zero_grad()
        de_output = decoder(patch_emb) 

        loss = loss_fn(de_output.squeeze(1), patch_label)

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
