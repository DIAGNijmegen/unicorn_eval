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
import torch
import torch.optim as optim
from monai.losses import DiceLoss
from unicorn_eval.adaptors.segmentation_decoder import SegResNetDecoderOnly, VectorToTensor
import torch.nn as nn
import numpy as np
from unicorn_eval.adaptors.patch_extraction import extract_patches
from collections import defaultdict
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from typing import Any, Iterable
from unicorn_eval.adaptors.base import PatchLevelTaskAdaptor

def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()
    return {
        "meta": {
            "patch-size": list(patch_size),
            "patch-spacing": list(patch_spacing),
            "image-size": list(image_size),
            "image-origin": list(image_origin),
            "image-spacing": list(image_spacing),
            "image-direction": list(image_direction),
        },
        "patches": list(patch_features),
        "title": title,
    }

class Decoder(nn.Module):
    def __init__(self, latent_dim, target_shape, decoder_kwargs):
        super().__init__()
        self.vector_to_tensor = VectorToTensor(latent_dim, target_shape)
        self.decoder = SegResNetDecoderOnly(**decoder_kwargs)
    
    def forward(self, x):
        x = self.vector_to_tensor(x)
        return self.decoder(x)


def train_decoder(decoder, data_loader, device):  
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    # Train decoder
    num_epochs = 5
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0
        for idx, batch in enumerate(data_loader):
            patch_emb = batch['patch'].to(device)
            patch_label = batch['patch_label'].to(device)   
            optimizer.zero_grad()
            de_output = decoder(patch_emb)

            loss = loss_fn(de_output.squeeze(1), patch_label)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
    return decoder 

def world_to_voxel(coord, origin, spacing, inv_direction):
    relative = np.array(coord) - origin
    voxel = inv_direction @ relative
    voxel = voxel / spacing
    return np.round(voxel).astype(int)

def create_grid(decoded_patches):
    grids = {}

    for idx, patches in decoded_patches.items():
        # Pull meta from the first patch
        meta = patches[0]
        image_size = meta['image_size']
        image_origin = meta['image_origin']
        image_spacing = meta['image_spacing']
        direction = np.array(meta['image_direction']).reshape(3, 3)
        inv_direction = np.linalg.inv(direction)
        patch_size = meta['patch_size']

        padded_shape = [
                int(np.ceil(image_size[d] / patch_size[d]) * patch_size[d])
                for d in range(3)
            ]        
        pX, pY, pZ = patch_size       # SITK order
        patch_size = (pZ, pY, pX)  # NumPy order
        padding = [(padded_shape[d] - image_size[d]) // 2 for d in range(3)]
        padding_mm = np.array(padding) * image_spacing
        adjusted_origin = image_origin - inv_direction @ padding_mm
        # Initialize grid
        pX, pY, pZ = padded_shape       # SITK order
        grid_shape = (pZ, pY, pX)  # NumPy order
        grid = np.zeros(grid_shape, dtype=np.float32)

        for patch in patches:
            i, j, k = world_to_voxel(patch['coord'], adjusted_origin, image_spacing, inv_direction)
            patch_array = patch['features'].squeeze(0)
            grid[k:k+patch_size[0], j:j+patch_size[1], i:i+patch_size[2]] += patch_array

        x_start = padding[0]
        x_end = x_start + image_size[0]
        y_start = padding[1]
        y_end = y_start + image_size[1]
        z_start = padding[2]
        z_end = z_start + image_size[2]
        cropped = grid[z_start:z_end, y_start:y_end, x_start:x_end]

        grids.update({idx : cropped})
    return grids

def inference(decoder, data_loader, device, return_binary):
    decoder.eval()
    with torch.no_grad():  
        decoded_patches = defaultdict(list)

        for batch in data_loader:
            inputs = batch['patch'].to(device)  # shape: [B, ...]
            coords = batch['coordinates']  # list of 3 tensors
            image_idxs = batch['case_number']

            outputs = decoder(inputs)  # shape: [B, ...]
            probs = torch.sigmoid(outputs)
            if return_binary:
                pred_mask = (probs > 0.5).float()
            else: 
                pred_mask = probs

            batch["image_origin"] = batch["image_origin"][0]
            batch["image_spacing"] = batch["image_spacing"][0]
            for i in range(0, len(image_idxs)):
                image_id = int(image_idxs[i])
                decoded_patches[image_id].append({
                        'coord': [float(c) for c in coords[i]],
                        'features': pred_mask[i].cpu().numpy(),
                        'patch_size': [int(batch['patch_size'][j][i]) for j in range(0, len(batch['patch_size']))],
                        'image_size': [int(batch['image_size'][j][i]) for j in range(0, len(batch['image_size']))],
                        'image_origin':[float(batch['image_origin'][j][i]) for j in range(0, len(batch['image_origin']))],
                        'image_spacing': [float(batch['image_spacing'][j][i]) for j in range(0, len(batch['image_spacing']))],
                        'image_direction': [float(batch['image_direction'][j][i]) for j in range(0, len(batch['image_direction']))],
                    })
    grids = create_grid(decoded_patches)
    return [j for j in grids.values()]

def construct_data_with_labels(
    coordinates,
    embeddings,
    cases,
    patch_size,
    labels=None, 
    image_sizes=None,
    image_origins=None,
    image_spacings=None,
    image_directions=None):
    data_array = []

    for case_idx, case in enumerate(cases):
        #patch_spacing = img_feat['meta']['patch-spacing']
        case_embeddings = embeddings[case_idx]
        patch_coordinates = coordinates[case_idx]

        lbl_feat = labels[case_idx] if labels is not None else None

        if len(case_embeddings) != len(patch_coordinates): 
            patch_coordinates = np.repeat(patch_coordinates, repeats=len(case_embeddings), axis=0)
   
        if lbl_feat is not None:
            if len(case_embeddings) != len(lbl_feat['patches']): 
                lbl_feat['patches'] = np.repeat(lbl_feat['patches'], repeats=len(case_embeddings), axis=0)

        for i, patch_img in enumerate(case_embeddings):
            data_dict = {
                'patch': np.array(patch_img, dtype=np.float32),
                'coordinates': patch_coordinates[i], 
                'patch_size': patch_size, 
                'case_number': case_idx
            }

            if lbl_feat is not None:
                patch_lbl = lbl_feat['patches'][i]
                assert np.allclose(patch_coordinates[i], patch_lbl['coordinates']), "Coordinates don't match!"
                data_dict['patch_label'] = np.array(patch_lbl['features'], dtype=np.float32) 
            
            if(image_sizes is not None) and (image_origins is not None) and (image_spacings is not None) and (image_directions is not None):
                image_size = image_sizes[case]
                image_origin = image_origins[case]
                image_spacing = image_spacings[case]
                image_direction = image_directions[case]
                
                data_dict['image_size'] = image_size
                data_dict['image_origin'] = image_origin, 
                data_dict['image_spacing'] = image_spacing, 
                data_dict['image_direction'] = image_direction

            data_array.append(data_dict)

    return data_array


def extract_patch_labels(
        image,
        image_spacing,
        image_origin, 
        image_direction,
        patch_size: list[int] = [16, 256, 256],
        patch_spacing: list[float] | None = None,
    ) -> list[dict]:
    """
    Generate a list of patch features from a radiology image

    Args:
        image_path (Path): Path to the image file
        title (str): Title of the patch-level neural representation
        patch_size (list[int]): Size of the patches to extract
        patch_spacing (list[float] | None): Voxel spacing of the image. If specified, the image will be resampled to this spacing before patch extraction.
    Returns:
        list[dict]: List of dictionaries containing the patch features
        - coordinates (list[tuple]): List of coordinates for each patch, formatted as:
            ((x_start, x_end), (y_start, y_end), (z_start, z_end)).
        - features (list[float]): List of features extracted from the patch
    """
    image = sitk.GetImageFromArray(image)
    image.SetOrigin(image_origin)
    image.SetSpacing(image_spacing)
    image.SetDirection(image_direction)

    patch_features = []

    patches, coordinates = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    for patch, coordinates in tqdm(zip(patches, coordinates), total=len(patches), desc="Extracting features"):
        patch_array = sitk.GetArrayFromImage(patch)
        patch_features.append({
            "coordinates": list(coordinates[0]),  # save the start coordinates
            "features": patch_array,
        })

    patch_labels = make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=image.GetSize(),
        image_origin=image.GetOrigin(),
        image_spacing=image.GetSpacing(),
        image_direction=image.GetDirection(),
        title='patch_labels',
    )

    return patch_labels

def load_patch_data(data_array: np.ndarray, batch_size: int = 4) -> DataLoader:
    train_ds = Dataset(data=data_array)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    return train_loader

class SegmentationUpsampling3D(PatchLevelTaskAdaptor):
    """
    Patch-level adaptor that trains a 3D upsampling decoder for segmentation.
    """

    def __init__(
        self,
        train_feats,
        train_coords,
        train_cases,
        train_labels,
        test_feats,
        test_coords,
        test_cases,
        test_image_sizes, 
        test_image_origins,
        test_image_spacings,
        test_image_directions,
        train_image_spacing,
        train_image_origins,
        train_image_directions,
        patch_size,
        return_binary = True,
    ):
        label_patch_features = []
        for idx, patch in enumerate(train_labels):
            label_feats = extract_patch_labels(
                patch,
                train_image_spacing[train_cases[idx]],
                train_image_origins[train_cases[idx]],
                train_image_directions[train_cases[idx]],
                patch_size
            )
            label_patch_features.append(label_feats)
        label_patch_features = np.array(label_patch_features, dtype=object)

        super().__init__(
            shot_features=train_feats,
            shot_labels=label_patch_features,
            shot_coordinates=train_coords,
            test_features=test_feats,
            test_coordinates=test_coords,
            shot_extra_labels=None,  # not used here
        )

        self.train_cases = train_cases
        self.test_cases = test_cases
        self.test_image_sizes = test_image_sizes
        self.test_image_origins = test_image_origins
        self.test_image_spacings = test_image_spacings
        self.test_image_directions = test_image_directions
        self.train_image_spacing = train_image_spacing
        self.train_image_origins = train_image_origins
        self.train_image_directions = train_image_directions
        self.patch_size = patch_size
        self.return_binary = return_binary

    def fit(self):
        # build training data and loader
        train_data = construct_data_with_labels(
            coordinates=self.shot_coordinates,
            embeddings=self.shot_features,
            cases=self.train_cases,
            patch_size=self.patch_size,
            labels=self.shot_labels,
        )
        train_loader = load_patch_data(train_data)

        target_patch_size = tuple(int(j/16) for j in self.patch_size)
        target_shape = (512, target_patch_size[2], target_patch_size[1], target_patch_size[0])
        
        # set up device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = Decoder(
            latent_dim=512,
            target_shape=target_shape,
            decoder_kwargs={
                "spatial_dims": 3,
                "init_filters": 32,
                "latent_channels": 512,
                "out_channels": 1,
                "blocks_up": (1, 1, 1, 1),
                "dsdepth": 1,
                "upsample_mode": "deconv",
            },
        ).to(self.device)
        self.decoder = train_decoder(self.decoder, train_loader, self.device)

    def predict(self) -> np.ndarray:
        # build test data and loader
        test_data = construct_data_with_labels(
            coordinates=self.test_coordinates,
            embeddings=self.test_features,
            cases=self.test_cases,
            patch_size=self.patch_size,
            image_sizes=self.test_image_sizes,
            image_origins=self.test_image_origins,
            image_spacings=self.test_image_spacings,
            image_directions=self.test_image_directions,
        )
        test_loader = load_patch_data(test_data)

        # run inference using the trained decoder
        return inference(self.decoder, test_loader, self.device, self.return_binary)

        





