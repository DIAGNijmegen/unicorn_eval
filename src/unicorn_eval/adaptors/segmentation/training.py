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
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from monai.losses.dice import DiceLoss
from tqdm import tqdm


def train_decoder(decoder, dataloader, num_epochs=200, lr=0.001):
    """Trains the decoder using the given data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )  # targets are class labels (not one-hot)

    for epoch in range(num_epochs):
        total_loss = 0

        for patch_emb, target_mask, _, _ in dataloader:
            patch_emb = patch_emb.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()
            pred_masks = decoder(patch_emb)
            target_mask = (
                target_mask.long()
            )  # Convert to LongTensor for CrossEntropyLoss

            loss = criterion(pred_masks, target_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        tqdm.write(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    return decoder


def train_decoder3d(decoder, data_loader, device, num_epochs: int = 3, iterations_per_epoch: int | None = None, loss_fn=None, optimizer=None, label_mapper=None, verbose: bool = True):
    if loss_fn is None:
        loss_fn = DiceLoss(sigmoid=True)
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
