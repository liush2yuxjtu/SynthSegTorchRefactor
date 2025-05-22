"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader


class SpatialTransformer(nn.Module):
    """
    Spatial transformer module for applying spatial transformations to 3D volumes.
    """
    def __init__(self, size, mode='bilinear'):
        """
        Initialize the spatial transformer.
        
        Args:
            size: Output size (d, h, w)
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        """
        super(SpatialTransformer, self).__init__()
        self.size = size
        self.mode = mode
        
    def forward(self, src, flow):
        """
        Apply spatial transformation to source volume using flow field.
        
        Args:
            src: Source volume [B, C, D, H, W]
            flow: Flow field [B, 3, D, H, W]
            
        Returns:
            Transformed volume
        """
        # Create sampling grid
        grid = self._create_sampling_grid(flow.shape[2:])
        grid = grid.to(flow.device)
        
        # Add flow to grid
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        
        # Normalize grid to [-1, 1]
        for i in range(3):
            grid[..., i] = 2 * (grid[..., i] / (self.size[i] - 1) - 0.5)
        
        # Apply transformation
        return F.grid_sample(src, grid, mode=self.mode, align_corners=True)
    
    def _create_sampling_grid(self, size):
        """Create a base sampling grid."""
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, dim=-1)
        grid = grid.unsqueeze(0)  # Add batch dimension
        return grid


class RandomSpatialDeformation(nn.Module):
    """
    Apply random spatial deformations to input volumes.
    """
    def __init__(
        self,
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=4.0,
        nonlin_scale=0.04,
        inter_method='nearest'
    ):
        """
        Initialize random spatial deformation module.
        
        Args:
            scaling_bounds: Bounds for random scaling
            rotation_bounds: Bounds for random rotation (degrees)
            shearing_bounds: Bounds for random shearing
            translation_bounds: Bounds for random translation
            nonlin_std: Standard deviation for nonlinear deformation
            nonlin_scale: Scale factor for nonlinear deformation
            inter_method: Interpolation method ('nearest', 'bilinear')
        """
        super(RandomSpatialDeformation, self).__init__()
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.inter_method = inter_method
        
    def forward(self, x):
        """
        Apply random spatial deformation to input volume.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Deformed volume
        """
        batch_size = x.shape[0]
        shape = x.shape[2:]
        
        # Create affine transformation matrices
        affine_matrices = self._get_random_affine_matrices(batch_size)
        
        # Create nonlinear deformation fields if needed
        if self.nonlin_std > 0:
            nonlinear_field = self._get_nonlinear_field(batch_size, shape)
        else:
            nonlinear_field = torch.zeros(batch_size, 3, *shape, device=x.device)
        
        # Create affine deformation field
        affine_field = self._affine_to_flow(affine_matrices, shape)
        
        # Combine deformation fields
        flow = affine_field + nonlinear_field
        
        # Apply transformation
        transformer = SpatialTransformer(shape, mode=self.inter_method)
        return transformer(x, flow)
    
    def _get_random_affine_matrices(self, batch_size):
        """Generate random affine transformation matrices."""
        device = next(self.parameters()).device
        matrices = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply random scaling
        if self.scaling_bounds:
            if isinstance(self.scaling_bounds, (int, float)):
                scale_factor = torch.rand(batch_size, 3, device=device) * 2 * self.scaling_bounds + (1 - self.scaling_bounds)
            else:
                scale_factor = torch.rand(batch_size, 3, device=device) * 2 * torch.tensor(self.scaling_bounds, device=device) + (1 - torch.tensor(self.scaling_bounds, device=device))
            
            for i in range(batch_size):
                matrices[i, :3, :3] = torch.diag(scale_factor[i]) @ matrices[i, :3, :3]
        
        # Apply random rotations
        if self.rotation_bounds:
            for axis in range(3):
                if isinstance(self.rotation_bounds, (int, float)):
                    angle = (torch.rand(batch_size, device=device) * 2 - 1) * self.rotation_bounds * (np.pi / 180)
                else:
                    angle = (torch.rand(batch_size, device=device) * 2 - 1) * torch.tensor(self.rotation_bounds, device=device) * (np.pi / 180)
                
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                
                for i in range(batch_size):
                    rot_matrix = torch.eye(3, device=device)
                    idx1, idx2 = [j for j in range(3) if j != axis]
                    rot_matrix[idx1, idx1] = cos_a[i]
                    rot_matrix[idx1, idx2] = -sin_a[i]
                    rot_matrix[idx2, idx1] = sin_a[i]
                    rot_matrix[idx2, idx2] = cos_a[i]
                    
                    matrices[i, :3, :3] = rot_matrix @ matrices[i, :3, :3]
        
        # Apply random shearing
        if self.shearing_bounds:
            if isinstance(self.shearing_bounds, (int, float)):
                shear_factor = (torch.rand(batch_size, 6, device=device) * 2 - 1) * self.shearing_bounds
            else:
                shear_factor = (torch.rand(batch_size, 6, device=device) * 2 - 1) * torch.tensor(self.shearing_bounds, device=device)
            
            for i in range(batch_size):
                shear_matrix = torch.eye(3, device=device)
                count = 0
                for axis in range(3):
                    for axis2 in range(3):
                        if axis != axis2:
                            shear_matrix[axis, axis2] = shear_factor[i, count]
                            count += 1
                
                matrices[i, :3, :3] = shear_matrix @ matrices[i, :3, :3]
        
        # Apply random translations
        if self.translation_bounds:
            if isinstance(self.translation_bounds, (int, float)):
                trans_factor = (torch.rand(batch_size, 3, device=device) * 2 - 1) * self.translation_bounds
            else:
                trans_factor = (torch.rand(batch_size, 3, device=device) * 2 - 1) * torch.tensor(self.translation_bounds, device=device)
            
            for i in range(batch_size):
                matrices[i, :3, 3] = trans_factor[i]
        
        return matrices
    
    def _get_nonlinear_field(self, batch_size, shape):
        """Generate random nonlinear deformation field."""
        device = next(self.parameters()).device
        
        # Calculate small field size
        small_shape = [int(s * self.nonlin_scale) for s in shape]
        small_shape = [max(s, 2) for s in small_shape]  # Ensure minimum size of 2
        
        # Generate random small field
        small_field = torch.randn(batch_size, 3, *small_shape, device=device) * self.nonlin_std
        
        # Upsample to full size
        field = F.interpolate(small_field, size=shape, mode='trilinear', align_corners=True)
        
        return field
    
    def _affine_to_flow(self, matrices, shape):
        """Convert affine matrices to flow fields."""
        device = matrices.device
        batch_size = matrices.shape[0]
        
        # Create base grid
        vectors = [torch.arange(0, s, device=device) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, dim=-1)
        
        # Add homogeneous coordinate
        ones = torch.ones(*shape, 1, device=device)
        grid_homo = torch.cat([grid, ones], dim=-1)
        grid_homo = grid_homo.view(-1, 4, 1)
        
        # Apply transformations
        flow = torch.zeros(batch_size, 3, *shape, device=device)
        for i in range(batch_size):
            # Apply inverse transformation
            inv_matrix = torch.inverse(matrices[i])
            transformed = inv_matrix @ grid_homo
            transformed = transformed.view(*shape, 4)[:, :, :, :3]
            
            # Calculate displacement
            displacement = transformed - grid
            
            # Reshape to flow field
            for j in range(3):
                flow[i, j] = displacement[..., j]
        
        return flow


class RandomCrop(nn.Module):
    """
    Randomly crop a volume to a specified size.
    """
    def __init__(self, crop_shape):
        """
        Initialize random crop module.
        
        Args:
            crop_shape: Target shape after cropping
        """
        super(RandomCrop, self).__init__()
        self.crop_shape = crop_shape
        
    def forward(self, x):
        """
        Apply random crop to input volume.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Cropped volume
        """
        input_shape = x.shape[2:]
        batch_size = x.shape[0]
        
        # Calculate valid crop ranges
        crop_ranges = []
        for i in range(len(input_shape)):
            if input_shape[i] > self.crop_shape[i]:
                crop_ranges.append(input_shape[i] - self.crop_shape[i])
            else:
                crop_ranges.append(0)
        
        # Generate random crop positions
        crop_starts = [torch.randint(0, r + 1, (batch_size,), device=x.device) for r in crop_ranges]
        
        # Apply crop to each sample in batch
        crops = []
        for b in range(batch_size):
            slices = [slice(None), slice(None)]  # Batch and channel dimensions
            for i, start in enumerate(crop_starts):
                if crop_ranges[i] > 0:
                    slices.append(slice(start[b], start[b] + self.crop_shape[i]))
                else:
                    slices.append(slice(None))
            crops.append(x[tuple(slices)])
        
        return torch.stack(crops)


class RandomFlip(nn.Module):
    """
    Randomly flip a volume along a specified axis.
    """
    def __init__(self, flip_axis, flip_prob, generation_labels, n_neutral_labels):
        """
        Initialize random flip module.
        
        Args:
            flip_axis: Axis along which to flip (0, 1, or 2)
            flip_prob: Probability of applying flip
            generation_labels: Labels used for generation
            n_neutral_labels: Number of neutral labels
        """
        super(RandomFlip, self).__init__()
        self.flip_axis = flip_axis
        self.flip_prob = flip_prob
        self.generation_labels = generation_labels
        self.n_neutral_labels = n_neutral_labels
        
    def forward(self, x):
        """
        Apply random flip to input volume.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Flipped volume
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Determine which samples to flip
        flip_mask = torch.rand(batch_size, device=device) < self.flip_prob
        
        if not flip_mask.any():
            return x
        
        # Create output tensor
        output = x.clone()
        
        # Apply flip to selected samples
        for b in range(batch_size):
            if flip_mask[b]:
                # Flip the volume
                slices = [b, slice(None)]
                for i in range(3):
                    if i == self.flip_axis:
                        slices.append(slice(None, None, -1))
                    else:
                        slices.append(slice(None))
                
                output[b] = x[tuple(slices)]
                
                # Swap lateralized labels if this is a segmentation
                if x.shape[1] == 1:  # Segmentation has 1 channel
                    # Create mapping for swapping labels
                    label_mapping = torch.arange(self.generation_labels.max() + 1, device=device)
                    
                    # Map lateralized labels
                    for i in range(self.n_neutral_labels, len(self.generation_labels) // 2 + self.n_neutral_labels):
                        j = len(self.generation_labels) - (i - self.n_neutral_labels) - 1 + self.n_neutral_labels
                        label_mapping[self.generation_labels[i]] = self.generation_labels[j]
                        label_mapping[self.generation_labels[j]] = self.generation_labels[i]
                    
                    # Apply mapping
                    output_flat = output[b, 0].reshape(-1)
                    output_flat = label_mapping[output_flat.long()]
                    output[b, 0] = output_flat.reshape(output[b, 0].shape)
        
        return output


class SampleConditionalGMM(nn.Module):
    """
    Sample from a conditional Gaussian Mixture Model based on label maps.
    """
    def __init__(self, generation_labels):
        """
        Initialize conditional GMM sampling module.
        
        Args:
            generation_labels: Labels used for generation
        """
        super(SampleConditionalGMM, self).__init__()
        self.generation_labels = generation_labels
        
    def forward(self, inputs):
        """
        Sample from conditional GMM.
        
        Args:
            inputs: List containing [labels, means, stds]
                labels: Label maps [B, 1, D, H, W]
                means: Mean values for each label [B, L, C]
                stds: Standard deviation values for each label [B, L, C]
            
        Returns:
            Sampled image
        """
        labels, means, stds = inputs
        batch_size, _, *spatial_dims = labels.shape
        n_channels = means.shape[-1]
        
        # Create output tensor
        output = torch.zeros(batch_size, n_channels, *spatial_dims, device=labels.device)
        
        # For each batch
        for b in range(batch_size):
            # For each channel
            for c in range(n_channels):
                # Get means and stds for this channel
                channel_means = means[b, :, c]
                channel_stds = stds[b, :, c]
                
                # Create noise field
                noise = torch.randn(*spatial_dims, device=labels.device)
                
                # For each label
                for i, label in enumerate(self.generation_labels):
                    # Create mask for this label
                    mask = (labels[b, 0] == label)
                    
                    if mask.any():
                        # Sample from Gaussian for this label
                        mean = channel_means[i]
                        std = channel_stds[i]
                        
                        # Apply to output
                        output[b, c][mask] = mean + std * noise[mask]
        
        return output


class BiasFieldCorruption(nn.Module):
    """
    Apply bias field corruption to images.
    """
    def __init__(self, bias_field_std=0.7, bias_scale=0.025, same_bias_for_all_channels=True):
        """
        Initialize bias field corruption module.
        
        Args:
            bias_field_std: Standard deviation of the bias field
            bias_scale: Scale factor for the bias field
            same_bias_for_all_channels: Whether to use the same bias field for all channels
        """
        super(BiasFieldCorruption, self).__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.same_bias_for_all_channels = same_bias_for_all_channels
        
    def forward(self, x):
        """
        Apply bias field corruption.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Corrupted volume
        """
        if self.bias_field_std <= 0:
            return x
        
        batch_size, n_channels, *spatial_dims = x.shape
        device = x.device
        
        # Calculate small field size
        small_shape = [int(s * self.bias_scale) for s in spatial_dims]
        small_shape = [max(s, 2) for s in small_shape]  # Ensure minimum size of 2
        
        # Generate bias fields
        if self.same_bias_for_all_channels:
            # Generate one bias field per batch
            small_bias = torch.randn(batch_size, 1, *small_shape, device=device) * self.bias_field_std
            bias = F.interpolate(small_bias, size=spatial_dims, mode='trilinear', align_corners=True)
            bias = bias.repeat(1, n_channels, 1, 1, 1)
        else:
            # Generate different bias field for each channel
            small_bias = torch.randn(batch_size, n_channels, *small_shape, device=device) * self.bias_field_std
            bias = F.interpolate(small_bias, size=spatial_dims, mode='trilinear', align_corners=True)
        
        # Apply bias field (multiplicative)
        return x * torch.exp(bias)


class IntensityAugmentation(nn.Module):
    """
    Apply intensity augmentation to images.
    """
    def __init__(self, clip=300, normalise=True, gamma_std=0.5, separate_channels=True):
        """
        Initialize intensity augmentation module.
        
        Args:
            clip: Clip values above this threshold
            normalise: Whether to normalize to [0, 1]
            gamma_std: Standard deviation for gamma augmentation
            separate_channels: Whether to apply augmentation separately to each channel
        """
        super(IntensityAugmentation, self).__init__()
        self.clip = clip
        self.normalise = normalise
        self.gamma_std = gamma_std
        self.separate_channels = separate_channels
        
    def forward(self, x):
        """
        Apply intensity augmentation.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Augmented volume
        """
        batch_size, n_channels = x.shape[:2]
        device = x.device
        
        # Clip values
        if self.clip:
            x = torch.clamp(x, max=self.clip)
        
        # Apply gamma augmentation
        if self.gamma_std > 0:
            if self.separate_channels:
                # Apply separately to each channel
                for b in range(batch_size):
                    for c in range(n_channels):
                        # Normalize to [0, 1] for gamma
                        x_min = x[b, c].min()
                        x_max = x[b, c].max()
                        if x_max > x_min:
                            x_norm = (x[b, c] - x_min) / (x_max - x_min)
                            
                            # Apply random gamma
                            gamma = torch.exp(torch.randn(1, device=device) * self.gamma_std)
                            x[b, c] = x_min + (x_max - x_min) * x_norm ** gamma
            else:
                # Apply same gamma to all channels
                for b in range(batch_size):
                    # Normalize to [0, 1] for gamma
                    x_min = x[b].min()
                    x_max = x[b].max()
                    if x_max > x_min:
                        x_norm = (x[b] - x_min) / (x_max - x_min)
                        
                        # Apply random gamma
                        gamma = torch.exp(torch.randn(1, device=device) * self.gamma_std)
                        x[b] = x_min + (x_max - x_min) * x_norm ** gamma
        
        # Normalize to [0, 1]
        if self.normalise:
            if self.separate_channels:
                # Normalize each channel separately
                for b in range(batch_size):
                    for c in range(n_channels):
                        x_min = x[b, c].min()
                        x_max = x[b, c].max()
                        if x_max > x_min:
                            x[b, c] = (x[b, c] - x_min) / (x_max - x_min)
            else:
                # Normalize all channels together
                for b in range(batch_size):
                    x_min = x[b].min()
                    x_max = x[b].max()
                    if x_max > x_min:
                        x[b] = (x[b] - x_min) / (x_max - x_min)
        
        return x


class GaussianBlur(nn.Module):
    """
    Apply Gaussian blur to images.
    """
    def __init__(self, sigma, truncate=4.0):
        """
        Initialize Gaussian blur module.
        
        Args:
            sigma: Standard deviation for Gaussian kernel
            truncate: Truncate the filter at this many standard deviations
        """
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.truncate = truncate
        
    def forward(self, x):
        """
        Apply Gaussian blur.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Blurred volume
        """
        if isinstance(self.sigma, (list, tuple, np.ndarray)):
            sigma = torch.tensor(self.sigma, device=x.device)
        else:
            sigma = torch.tensor([self.sigma] * 3, device=x.device)
        
        # Skip if sigma is too small
        if torch.all(sigma < 0.01):
            return x
        
        # Create 1D kernels
        kernels = []
        for i in range(3):
            if sigma[i] > 0.01:
                kernel_size = int(self.truncate * sigma[i]) * 2 + 1
                kernel_size = max(3, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Create 1D Gaussian kernel
                kernel_1d = torch.exp(-0.5 * (torch.arange(kernel_size, device=x.device) - kernel_size // 2) ** 2 / sigma[i] ** 2)
                kernel_1d = kernel_1d / kernel_1d.sum()
                
                kernels.append((i, kernel_1d, kernel_size))
        
        # Apply blur along each dimension
        output = x
        for dim, kernel_1d, kernel_size in kernels:
            # Reshape kernel for this dimension
            if dim == 0:  # D dimension
                kernel = kernel_1d.view(1, 1, kernel_size, 1, 1)
                padding = (0, 0, 0, 0, kernel_size // 2, kernel_size // 2)
            elif dim == 1:  # H dimension
                kernel = kernel_1d.view(1, 1, 1, kernel_size, 1)
                padding = (0, 0, kernel_size // 2, kernel_size // 2, 0, 0)
            else:  # W dimension
                kernel = kernel_1d.view(1, 1, 1, 1, kernel_size)
                padding = (kernel_size // 2, kernel_size // 2, 0, 0, 0, 0)
            
            # Apply convolution
            output = F.conv3d(
                F.pad(output, padding, mode='replicate'),
                kernel.repeat(output.shape[1], 1, 1, 1, 1),
                groups=output.shape[1]
            )
        
        return output


class DynamicGaussianBlur(nn.Module):
    """
    Apply Gaussian blur with dynamic sigma values.
    """
    def __init__(self, max_sigma, truncate=4.0):
        """
        Initialize dynamic Gaussian blur module.
        
        Args:
            max_sigma: Maximum sigma values
            truncate: Truncate the filter at this many standard deviations
        """
        super(DynamicGaussianBlur, self).__init__()
        self.max_sigma = max_sigma
        self.truncate = truncate
        
    def forward(self, inputs):
        """
        Apply dynamic Gaussian blur.
        
        Args:
            inputs: List containing [x, sigma]
                x: Input volume [B, C, D, H, W]
                sigma: Sigma values [B, 3]
            
        Returns:
            Blurred volume
        """
        x, sigma = inputs
        batch_size = x.shape[0]
        device = x.device
        
        # Process each batch separately
        outputs = []
        for b in range(batch_size):
            # Get sigma for this batch
            batch_sigma = sigma[b]
            
            # Skip if sigma is too small
            if torch.all(batch_sigma < 0.01):
                outputs.append(x[b:b+1])
                continue
            
            # Create 1D kernels
            kernels = []
            for i in range(3):
                if batch_sigma[i] > 0.01:
                    kernel_size = int(self.truncate * batch_sigma[i]) * 2 + 1
                    kernel_size = max(3, kernel_size)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    # Create 1D Gaussian kernel
                    kernel_1d = torch.exp(-0.5 * (torch.arange(kernel_size, device=device) - kernel_size // 2) ** 2 / batch_sigma[i] ** 2)
                    kernel_1d = kernel_1d / kernel_1d.sum()
                    
                    kernels.append((i, kernel_1d, kernel_size))
            
            # Apply blur along each dimension
            output = x[b:b+1]
            for dim, kernel_1d, kernel_size in kernels:
                # Reshape kernel for this dimension
                if dim == 0:  # D dimension
                    kernel = kernel_1d.view(1, 1, kernel_size, 1, 1)
                    padding = (0, 0, 0, 0, kernel_size // 2, kernel_size // 2)
                elif dim == 1:  # H dimension
                    kernel = kernel_1d.view(1, 1, 1, kernel_size, 1)
                    padding = (0, 0, kernel_size // 2, kernel_size // 2, 0, 0)
                else:  # W dimension
                    kernel = kernel_1d.view(1, 1, 1, 1, kernel_size)
                    padding = (kernel_size // 2, kernel_size // 2, 0, 0, 0, 0)
                
                # Apply convolution
                output = F.conv3d(
                    F.pad(output, padding, mode='replicate'),
                    kernel.repeat(output.shape[1], 1, 1, 1, 1),
                    groups=output.shape[1]
                )
            
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)


class MimicAcquisition(nn.Module):
    """
    Mimic acquisition at a specific resolution.
    """
    def __init__(self, atlas_res, data_res, output_shape=None, downsample=False):
        """
        Initialize acquisition mimicking module.
        
        Args:
            atlas_res: Resolution of the input volume
            data_res: Target acquisition resolution
            output_shape: Target output shape
            downsample: Whether to downsample the volume
        """
        super(MimicAcquisition, self).__init__()
        self.atlas_res = atlas_res
        self.data_res = data_res
        self.output_shape = output_shape
        self.downsample = downsample
        
    def forward(self, inputs):
        """
        Mimic acquisition process.
        
        Args:
            inputs: List containing [x, resolution]
                x: Input volume [B, C, D, H, W]
                resolution: Target resolution [B, 3] or [3]
            
        Returns:
            Processed volume
        """
        x, resolution = inputs
        batch_size = x.shape[0]
        device = x.device
        
        # Get atlas resolution
        if isinstance(self.atlas_res, (list, tuple, np.ndarray)):
            atlas_res = torch.tensor(self.atlas_res, device=device)
        else:
            atlas_res = torch.tensor([self.atlas_res] * 3, device=device)
        
        # Process each batch separately
        outputs = []
        for b in range(batch_size):
            # Get resolution for this batch
            if resolution.dim() > 1:
                batch_res = resolution[b]
            else:
                batch_res = resolution
            
            # Skip if resolution is the same
            if torch.all(torch.abs(batch_res - atlas_res) < 0.01):
                outputs.append(x[b:b+1])
                continue
            
            # Calculate downsampling factors
            factors = atlas_res / batch_res
            
            if self.downsample:
                # Downsample to low resolution
                lr_shape = [int(x.shape[2+i] / factors[i]) for i in range(3)]
                lr_shape = [max(s, 1) for s in lr_shape]  # Ensure minimum size of 1
                
                # Downsample
                x_lr = F.interpolate(x[b:b+1], size=lr_shape, mode='trilinear', align_corners=True)
                
                # Upsample back to original or target shape
                if self.output_shape is not None:
                    output = F.interpolate(x_lr, size=self.output_shape, mode='trilinear', align_corners=True)
                else:
                    output = F.interpolate(x_lr, size=x.shape[2:], mode='trilinear', align_corners=True)
            else:
                # Just keep the original volume
                output = x[b:b+1]
                
                # Resize to target shape if needed
                if self.output_shape is not None and self.output_shape != x.shape[2:]:
                    output = F.interpolate(output, size=self.output_shape, mode='trilinear', align_corners=True)
            
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)


class ConvertLabels(nn.Module):
    """
    Convert label values from source to destination values.
    """
    def __init__(self, source_values, dest_values=None):
        """
        Initialize label conversion module.
        
        Args:
            source_values: Source label values
            dest_values: Destination label values (if None, same as source)
        """
        super(ConvertLabels, self).__init__()
        self.source_values = source_values
        self.dest_values = dest_values if dest_values is not None else source_values
        
    def forward(self, x):
        """
        Convert label values.
        
        Args:
            x: Input volume with labels [B, 1, D, H, W]
            
        Returns:
            Volume with converted labels
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Create mapping
        max_label = int(self.source_values.max().item())
        mapping = torch.zeros(max_label + 1, dtype=torch.long, device=device)
        
        for i, src_val in enumerate(self.source_values):
            mapping[src_val] = self.dest_values[i]
        
        # Apply mapping
        output = x.clone()
        for b in range(batch_size):
            # Flatten for efficient mapping
            flat_labels = output[b, 0].reshape(-1).long()
            flat_labels = mapping[flat_labels]
            output[b, 0] = flat_labels.reshape(output[b, 0].shape)
        
        return output


class ImageGradients(nn.Module):
    """
    Compute image gradients using various filters.
    """
    def __init__(self, filter_type='sobel', return_magnitude=True):
        """
        Initialize image gradients module.
        
        Args:
            filter_type: Type of filter ('sobel', 'prewitt', etc.)
            return_magnitude: Whether to return gradient magnitude
        """
        super(ImageGradients, self).__init__()
        self.filter_type = filter_type
        self.return_magnitude = return_magnitude
        
    def forward(self, x):
        """
        Compute image gradients.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Gradient volume
        """
        batch_size, n_channels = x.shape[:2]
        device = x.device
        
        # Create Sobel filters
        if self.filter_type == 'sobel':
            # 3D Sobel filters
            filters = []
            for axis in range(3):
                kernel = torch.zeros(3, 3, 3, device=device)
                
                # Fill the kernel
                if axis == 0:  # D dimension
                    kernel[0, :, :] = -1
                    kernel[2, :, :] = 1
                    kernel[1, 1, 1] = 0
                elif axis == 1:  # H dimension
                    kernel[:, 0, :] = -1
                    kernel[:, 2, :] = 1
                    kernel[1, 1, 1] = 0
                else:  # W dimension
                    kernel[:, :, 0] = -1
                    kernel[:, :, 2] = 1
                    kernel[1, 1, 1] = 0
                
                filters.append(kernel)
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")
        
        # Apply filters to each channel
        gradients = []
        for b in range(batch_size):
            for c in range(n_channels):
                # Compute gradients along each axis
                grads = []
                for kernel in filters:
                    # Reshape kernel for 3D convolution
                    kernel = kernel.unsqueeze(0).unsqueeze(0)
                    
                    # Apply convolution
                    grad = F.conv3d(
                        F.pad(x[b:b+1, c:c+1], (1, 1, 1, 1, 1, 1), mode='replicate'),
                        kernel
                    )
                    grads.append(grad)
                
                # Combine gradients
                if self.return_magnitude:
                    # Compute magnitude
                    grad_magnitude = torch.sqrt(sum(g**2 for g in grads))
                    gradients.append(grad_magnitude)
                else:
                    # Return all gradients
                    gradients.extend(grads)
        
        # Combine results
        if self.return_magnitude:
            return torch.cat(gradients, dim=1)
        else:
            return torch.cat(gradients, dim=1)


class SampleResolution(nn.Module):
    """
    Sample random resolution for acquisition simulation.
    """
    def __init__(self, atlas_res, max_res_iso, max_res_aniso):
        """
        Initialize resolution sampling module.
        
        Args:
            atlas_res: Resolution of the input volume
            max_res_iso: Maximum isotropic resolution
            max_res_aniso: Maximum anisotropic resolution
        """
        super(SampleResolution, self).__init__()
        self.atlas_res = atlas_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        
    def forward(self, x):
        """
        Sample random resolution.
        
        Args:
            x: Dummy input (not used)
            
        Returns:
            Tuple of (resolution, blur_resolution)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get atlas resolution
        if isinstance(self.atlas_res, (list, tuple, np.ndarray)):
            atlas_res = torch.tensor(self.atlas_res, device=device)
        else:
            atlas_res = torch.tensor([self.atlas_res] * 3, device=device)
        
        # Get maximum resolutions
        if isinstance(self.max_res_iso, (list, tuple, np.ndarray)):
            max_res_iso = torch.tensor(self.max_res_iso, device=device)
        else:
            max_res_iso = torch.tensor([self.max_res_iso] * 3, device=device)
            
        if isinstance(self.max_res_aniso, (list, tuple, np.ndarray)):
            max_res_aniso = torch.tensor(self.max_res_aniso, device=device)
        else:
            max_res_aniso = torch.tensor([self.max_res_aniso] * 3, device=device)
        
        # Sample resolutions for each batch
        resolutions = []
        blur_resolutions = []
        
        for b in range(batch_size):
            # Randomly choose between isotropic and anisotropic
            if torch.rand(1, device=device) > 0.5:
                # Isotropic resolution
                res_factor = torch.rand(1, device=device)
                resolution = atlas_res + res_factor * (max_res_iso - atlas_res)
                blur_resolution = resolution.clone()
            else:
                # Anisotropic resolution (high resolution in all but one dimension)
                resolution = atlas_res.clone()
                blur_resolution = atlas_res.clone()
                
                # Choose random axis for anisotropy
                axis = torch.randint(0, 3, (1,), device=device).item()
                
                # Sample resolution for that axis
                res_factor = torch.rand(1, device=device)
                resolution[axis] = atlas_res[axis] + res_factor * (max_res_aniso[axis] - atlas_res[axis])
                blur_resolution[axis] = resolution[axis]
            
            resolutions.append(resolution)
            blur_resolutions.append(blur_resolution)
        
        return torch.stack(resolutions), torch.stack(blur_resolutions)


def blurring_sigma_for_downsampling(current_res, target_res, thickness=None):
    """
    Calculate sigma for Gaussian blurring when downsampling.
    
    Args:
        current_res: Current resolution
        target_res: Target resolution
        thickness: Slice thickness (if None, same as target_res)
        
    Returns:
        Sigma values for blurring
    """
    if isinstance(current_res, (list, tuple, np.ndarray)):
        current_res = torch.tensor(current_res)
    else:
        current_res = torch.tensor([current_res] * 3)
        
    if isinstance(target_res, (list, tuple, np.ndarray)):
        target_res = torch.tensor(target_res)
    else:
        target_res = torch.tensor([target_res] * 3)
        
    if thickness is None:
        thickness = target_res.clone()
    elif isinstance(thickness, (list, tuple, np.ndarray)):
        thickness = torch.tensor(thickness)
    else:
        thickness = torch.tensor([thickness] * 3)
    
    # Calculate sigma based on resolution difference
    sigma = torch.sqrt((thickness**2 - current_res**2) / (8 * torch.log(torch.tensor(2.0))))
    
    # Set sigma to 0 where target resolution is smaller than current
    sigma = torch.where(target_res <= current_res, torch.tensor(0.0), sigma)
    
    return sigma


class BrainGeneratorDataset(Dataset):
    """
    Dataset for brain generation.
    """
    def __init__(
        self,
        labels_dir,
        generation_labels=None,
        n_neutral_labels=None,
        output_labels=None,
        subjects_prob=None,
        n_channels=1,
        generation_classes=None,
        prior_distributions='uniform',
        prior_means=None,
        prior_stds=None,
        use_specific_stats_for_channel=False,
        mix_prior_and_random=False
    ):
        """
        Initialize brain generator dataset.
        
        Args:
            labels_dir: Directory containing label maps
            generation_labels: Labels used for generation
            n_neutral_labels: Number of neutral labels
            output_labels: Labels to output
            subjects_prob: Probability of selecting each subject
            n_channels: Number of channels to generate
            generation_classes: Classes for generation
            prior_distributions: Type of prior distributions
            prior_means: Prior means for GMM
            prior_stds: Prior standard deviations for GMM
            use_specific_stats_for_channel: Whether to use specific stats for each channel
            mix_prior_and_random: Whether to mix prior and random stats
        """
        self.labels_paths = self._list_files(labels_dir)
        
        # Set up subject probabilities
        if subjects_prob is not None:
            self.subjects_prob = np.array(self._load_if_path(subjects_prob), dtype='float32')
            assert len(self.subjects_prob) == len(self.labels_paths), \
                'subjects_prob should have the same length as labels_path'
            self.subjects_prob /= np.sum(self.subjects_prob)
        else:
            self.subjects_prob = np.ones(len(self.labels_paths), dtype='float32') / len(self.labels_paths)
        
        # Get label information
        self.n_channels = n_channels
        
        # Load first label map to get shape and other info
        first_label = self._load_volume(self.labels_paths[0])
        self.labels_shape = first_label.shape
        
        # Set up generation labels
        if generation_labels is not None:
            self.generation_labels = self._load_if_path(generation_labels)
        else:
            self.generation_labels = self._get_unique_labels(labels_dir)
        
        # Set up output labels
        if output_labels is not None:
            self.output_labels = self._load_if_path(output_labels)
        else:
            self.output_labels = self.generation_labels
        
        # Set up neutral labels
        if n_neutral_labels is not None:
            self.n_neutral_labels = n_neutral_labels
        else:
            self.n_neutral_labels = len(self.generation_labels)
        
        # Set up generation classes
        if generation_classes is not None:
            self.generation_classes = self._load_if_path(generation_classes)
        else:
            self.generation_classes = np.arange(len(self.generation_labels))
        
        # Set up GMM parameters
        self.prior_distributions = prior_distributions
        
        if prior_means is not None:
            self.prior_means = self._load_if_path(prior_means)
        else:
            self.prior_means = np.array([25, 225])
        
        if prior_stds is not None:
            self.prior_stds = self._load_if_path(prior_stds)
        else:
            self.prior_stds = np.array([5, 25])
        
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        self.mix_prior_and_random = mix_prior_and_random
    
    def __len__(self):
        """Return the number of label maps."""
        return len(self.labels_paths)
    
    def __getitem__(self, idx):
        """
        Get a random label map and generate GMM parameters.
        
        Args:
            idx: Index (not used, random sampling is used instead)
            
        Returns:
            Dictionary with label map and GMM parameters
        """
        # Randomly select a label map based on probabilities
        idx = np.random.choice(len(self.labels_paths), p=self.subjects_prob)
        
        # Load label map
        label_map = self._load_volume(self.labels_paths[idx])
        
        # Special case for cerebral segmentation
        if (np.random.uniform() > 0.7) and ('seg_cerebral' in self.labels_paths[idx]):
            label_map[label_map == 24] = 0
        
        # Add batch and channel dimensions
        label_map = np.expand_dims(np.expand_dims(label_map, axis=0), axis=0)
        
        # Generate means and standard deviations
        means = np.empty((1, len(self.generation_labels), 0))
        stds = np.empty((1, len(self.generation_labels), 0))
        
        for channel in range(self.n_channels):
            # Get channel-specific priors if needed
            tmp_prior_means = self._get_channel_prior(self.prior_means, channel)
            if self.mix_prior_and_random and (np.random.uniform() > 0.5):
                tmp_prior_means = None
                
            tmp_prior_stds = self._get_channel_prior(self.prior_stds, channel)
            if self.mix_prior_and_random and (np.random.uniform() > 0.5):
                tmp_prior_stds = None
            
            # Draw means and stds from priors
            n_classes = len(np.unique(self.generation_classes))
            tmp_classes_means = self._draw_from_distribution(tmp_prior_means, n_classes, 125., 125.)
            tmp_classes_stds = self._draw_from_distribution(tmp_prior_stds, n_classes, 15., 15.)
            
            # Special handling for background
            random_coef = np.random.uniform()
            if random_coef > 0.95:  # Reset background to 0 in 5% of cases
                tmp_classes_means[0] = 0
                tmp_classes_stds[0] = 0
            elif random_coef > 0.7:  # Reset background to low Gaussian in 25% of cases
                tmp_classes_means[0] = np.random.uniform(0, 15)
                tmp_classes_stds[0] = np.random.uniform(0, 5)
            
            # Map class values to label values
            tmp_means = np.expand_dims(np.expand_dims(tmp_classes_means[self.generation_classes], axis=0), axis=-1)
            tmp_stds = np.expand_dims(np.expand_dims(tmp_classes_stds[self.generation_classes], axis=0), axis=-1)
            
            means = np.concatenate([means, tmp_means], axis=-1)
            stds = np.concatenate([stds, tmp_stds], axis=-1)
        
        return {
            'label_map': label_map,
            'means': means,
            'stds': stds
        }
    
    def _list_files(self, directory):
        """List all files in a directory."""
        if os.path.isfile(directory):
            return [directory]
        
        files = []
        for ext in ['*.nii.gz', '*.nii', '*.mgz', '*.npz']:
            files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        
        return sorted(files)
    
    def _load_volume(self, path):
        """Load a volume from file."""
        # This is a simplified version - in a real implementation,
        # you would use nibabel or similar to load medical images
        if path.endswith('.npz'):
            data = np.load(path)
            return data['vol']
        else:
            # Placeholder for loading nifti/mgz files
            # In a real implementation, use nibabel:
            # import nibabel as nib
            # return nib.load(path).get_fdata()
            raise NotImplementedError("Please implement proper loading for your file format")
    
    def _load_if_path(self, obj):
        """Load object if it's a file path."""
        if isinstance(obj, str) and os.path.isfile(obj):
            return np.load(obj)
        return obj
    
    def _get_unique_labels(self, labels_dir):
        """Get unique labels from all label maps."""
        all_labels = set()
        
        for path in self.labels_paths:
            labels = np.unique(self._load_volume(path))
            all_labels.update(labels)
        
        return np.array(sorted(list(all_labels)))
    
    def _get_channel_prior(self, prior, channel):
        """Get channel-specific prior."""
        if prior is None:
            return None
            
        if isinstance(prior, np.ndarray):
            if (prior.shape[0] > 2) and self.use_specific_stats_for_channel:
                if prior.shape[0] // 2 != self.n_channels:
                    raise ValueError("Number of blocks in prior does not match n_channels")
                return prior[2 * channel:2 * channel + 2, :]
            else:
                return prior
        else:
            return prior
    
    def _draw_from_distribution(self, prior, size, default_mean, default_std):
        """Draw values from distribution based on prior."""
        if prior is None:
            # Use default values
            if self.prior_distributions == 'uniform':
                return np.random.uniform(
                    default_mean - default_std,
                    default_mean + default_std,
                    size=size
                )
            else:  # normal
                return np.random.normal(default_mean, default_std, size=size)
        
        if len(np.array(prior).shape) == 1 and len(prior) == 2:
            # Single distribution for all classes
            if self.prior_distributions == 'uniform':
                return np.random.uniform(prior[0], prior[1], size=size)
            else:  # normal
                return np.random.normal(prior[0], prior[1], size=size)
        else:
            # Different distribution for each class
            result = np.zeros(size)
            for k in range(size):
                if k < prior.shape[1]:
                    if self.prior_distributions == 'uniform':
                        result[k] = np.random.uniform(prior[0, k], prior[1, k])
                    else:  # normal
                        result[k] = np.random.normal(prior[0, k], prior[1, k])
                else:
                    # Default for classes without specific priors
                    if self.prior_distributions == 'uniform':
                        result[k] = np.random.uniform(default_mean - default_std, default_mean + default_std)
                    else:  # normal
                        result[k] = np.random.normal(default_mean, default_std)
            
            return result


class BrainGeneratorModel(nn.Module):
    """
    Model for generating synthetic brain images from label maps.
    """
    def __init__(
        self,
        labels_shape,
        n_channels,
        generation_labels,
        output_labels,
        n_neutral_labels,
        atlas_res,
        target_res=None,
        output_shape=None,
        output_div_by_n=None,
        flipping=True,
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=4.0,
        nonlin_scale=0.04,
        randomise_res=False,
        max_res_iso=4.0,
        max_res_aniso=8.0,
        data_res=None,
        thickness=None,
        bias_field_std=0.7,
        bias_scale=0.025,
        return_gradients=False
    ):
        """
        Initialize brain generator model.
        
        Args:
            labels_shape: Shape of input label maps
            n_channels: Number of channels to generate
            generation_labels: Labels used for generation
            output_labels: Labels to output
            n_neutral_labels: Number of neutral labels
            atlas_res: Resolution of input label maps
            target_res: Target resolution
            output_shape: Target output shape
            output_div_by_n: Ensure output shape is divisible by this value
            flipping: Whether to apply random flipping
            scaling_bounds: Bounds for random scaling
            rotation_bounds: Bounds for random rotation
            shearing_bounds: Bounds for random shearing
            translation_bounds: Bounds for random translation
            nonlin_std: Standard deviation for nonlinear deformation
            nonlin_scale: Scale factor for nonlinear deformation
            randomise_res: Whether to randomize resolution
            max_res_iso: Maximum isotropic resolution
            max_res_aniso: Maximum anisotropic resolution
            data_res: Specific data resolution to mimic
            thickness: Slice thickness
            bias_field_std: Standard deviation for bias field
            bias_scale: Scale factor for bias field
            return_gradients: Whether to return image gradients
        """
        super(BrainGeneratorModel, self).__init__()
        
        # Store parameters
        self.labels_shape = labels_shape if isinstance(labels_shape, (list, tuple)) else [labels_shape] * 3
        self.n_channels = n_channels
        self.generation_labels = generation_labels
        self.output_labels = output_labels
        self.n_neutral_labels = n_neutral_labels
        self.atlas_res = atlas_res if isinstance(atlas_res, (list, tuple, np.ndarray)) else [atlas_res] * 3
        self.target_res = target_res if target_res is not None else self.atlas_res
        self.target_res = self.target_res if isinstance(self.target_res, (list, tuple, np.ndarray)) else [self.target_res] * 3
        self.flipping = flipping
        self.return_gradients = return_gradients
        
        # Calculate shapes
        self.crop_shape, self.output_shape = self._get_shapes(output_shape, output_div_by_n)
        
        # Set up modules
        self.spatial_deformation = RandomSpatialDeformation(
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            nonlin_scale=nonlin_scale,
            inter_method='nearest'
        )
        
        if self.crop_shape != self.labels_shape:
            self.random_crop = RandomCrop(self.crop_shape)
        else:
            self.random_crop = nn.Identity()
        
        if flipping:
            self.random_flip = RandomFlip(0, 0.5, generation_labels, n_neutral_labels)
        else:
            self.random_flip = nn.Identity()
        
        self.sample_gmm = SampleConditionalGMM(generation_labels)
        
        if bias_field_std > 0:
            self.bias_field = BiasFieldCorruption(bias_field_std, bias_scale, True)
        else:
            self.bias_field = nn.Identity()
        
        self.intensity_augmentation = IntensityAugmentation(clip=300, normalise=True, gamma_std=0.5, separate_channels=True)
        
        # Resolution-related modules
        self.randomise_res = randomise_res
        if randomise_res:
            self.sample_resolution = SampleResolution(self.atlas_res, max_res_iso, max_res_aniso)
            self.dynamic_blur = DynamicGaussianBlur(max(max_res_iso, max_res_aniso) / np.array(self.atlas_res), 1.03)
            self.mimic_acquisition = MimicAcquisition(self.atlas_res, self.atlas_res, self.output_shape, False)
        else:
            data_res = self.atlas_res if data_res is None else data_res
            data_res = data_res if isinstance(data_res, (list, tuple, np.ndarray)) else [data_res] * 3
            
            thickness = data_res if thickness is None else thickness
            thickness = thickness if isinstance(thickness, (list, tuple, np.ndarray)) else [thickness] * 3
            
            sigma = blurring_sigma_for_downsampling(self.atlas_res, data_res, thickness)
            self.gaussian_blur = GaussianBlur(sigma, 1.03)
            self.mimic_acquisition = MimicAcquisition(self.atlas_res, data_res, self.output_shape)
        
        if return_gradients:
            self.image_gradients = ImageGradients('sobel', True)
            self.gradient_augmentation = IntensityAugmentation(clip=10, normalise=True)
        
        if self.crop_shape != self.output_shape:
            self.resample = lambda x: F.interpolate(x, size=self.output_shape, mode='nearest')
        else:
            self.resample = nn.Identity()
        
        self.convert_labels = ConvertLabels(generation_labels, output_labels)
    
    def forward(self, inputs):
        """
        Generate synthetic brain image from label map.
        
        Args:
            inputs: Dictionary containing:
                label_map: Input label map [B, 1, D, H, W]
                means: GMM means [B, L, C]
                stds: GMM standard deviations [B, L, C]
            
        Returns:
            Tuple of (image, labels)
        """
        label_map = inputs['label_map']
        means = inputs['means']
        stds = inputs['stds']
        
        # Apply spatial deformation
        labels = self.spatial_deformation(label_map)
        
        # Apply random crop if needed
        if self.crop_shape != self.labels_shape:
            labels = self.random_crop(labels)
        
        # Apply random flip if enabled
        if self.flipping:
            labels = self.random_flip(labels)
        
        # Sample from GMM
        image = self.sample_gmm([labels, means, stds])
        
        # Apply bias field
        image = self.bias_field(image)
        
        # Apply intensity augmentation
        image = self.intensity_augmentation(image)
        
        # Process each channel for resolution
        if self.randomise_res:
            # Sample random resolution
            resolution, blur_res = self.sample_resolution(means)
            
            # Calculate sigma for blurring
            sigma = torch.stack([
                blurring_sigma_for_downsampling(
                    torch.tensor(self.atlas_res, device=resolution.device),
                    resolution[i],
                    blur_res[i]
                )
                for i in range(resolution.shape[0])
            ])
            
            # Apply blurring and mimic acquisition
            image = self.dynamic_blur([image, sigma])
            image = self.mimic_acquisition([image, resolution])
        else:
            # Apply fixed blurring and acquisition
            image = self.gaussian_blur(image)
            resolution = torch.tensor(self.atlas_res, device=image.device)
            image = self.mimic_acquisition([image, resolution])
        
        # Compute image gradients if requested
        if self.return_gradients:
            image = self.image_gradients(image)
            image = self.gradient_augmentation(image)
        
        # Resample labels to output shape if needed
        if self.crop_shape != self.output_shape:
            labels = self.resample(labels)
        
        # Convert labels to output values
        labels = self.convert_labels(labels)
        
        return image, labels
    
    def _get_shapes(self, output_shape, output_div_by_n):
        """
        Calculate crop shape and output shape.
        
        Args:
            output_shape: Requested output shape
            output_div_by_n: Ensure output shape is divisible by this value
            
        Returns:
            Tuple of (crop_shape, output_shape)
        """
        # Default is to use the input shape
        crop_shape = list(self.labels_shape)
        
        # Calculate resampling factor if resolutions differ
        if self.atlas_res != self.target_res:
            resample_factor = [self.atlas_res[i] / self.target_res[i] for i in range(len(self.atlas_res))]
        else:
            resample_factor = None
        
        # Process output shape if specified
        if output_shape is not None:
            output_shape = output_shape if isinstance(output_shape, (list, tuple)) else [output_shape] * len(self.labels_shape)
            
            # Make sure output shape is smaller or equal to label shape
            if resample_factor is not None:
                output_shape = [min(int(self.labels_shape[i] * resample_factor[i]), output_shape[i]) for i in range(len(self.labels_shape))]
            else:
                output_shape = [min(self.labels_shape[i], output_shape[i]) for i in range(len(self.labels_shape))]
            
            # Make output shape divisible by n if requested
            if output_div_by_n is not None:
                output_div_by_n = output_div_by_n if isinstance(output_div_by_n, (list, tuple)) else [output_div_by_n] * len(self.labels_shape)
                output_shape = [int(output_shape[i] // output_div_by_n[i] * output_div_by_n[i]) for i in range(len(self.labels_shape))]
            
            # Calculate crop shape based on output shape and resampling factor
            if resample_factor is not None:
                crop_shape = [int(output_shape[i] / resample_factor[i]) for i in range(len(self.labels_shape))]
            else:
                crop_shape = output_shape
        else:
            # If no output shape specified, use resampled input shape
            if resample_factor is not None:
                output_shape = [int(self.labels_shape[i] * resample_factor[i]) for i in range(len(self.labels_shape))]
            else:
                output_shape = self.labels_shape
            
            # Make output shape divisible by n if requested
            if output_div_by_n is not None:
                output_div_by_n = output_div_by_n if isinstance(output_div_by_n, (list, tuple)) else [output_div_by_n] * len(self.labels_shape)
                output_shape = [int(output_shape[i] // output_div_by_n[i] * output_div_by_n[i]) for i in range(len(self.labels_shape))]
                
                # Adjust crop shape if needed
                if resample_factor is not None:
                    crop_shape = [int(output_shape[i] / resample_factor[i]) for i in range(len(self.labels_shape))]
        
        return crop_shape, output_shape


class BrainGenerator:
    """
    Main class for generating synthetic brain images from label maps.
    """
    def __init__(
        self,
        labels_dir,
        generation_labels=None,
        n_neutral_labels=None,
        output_labels=None,
        subjects_prob=None,
        batchsize=1,
        n_channels=1,
        target_res=None,
        output_shape=None,
        output_div_by_n=None,
        prior_distributions='uniform',
        generation_classes=None,
        prior_means=None,
        prior_stds=None,
        use_specific_stats_for_channel=False,
        mix_prior_and_random=False,
        flipping=True,
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=4.0,
        nonlin_scale=0.04,
        randomise_res=True,
        max_res_iso=4.0,
        max_res_aniso=8.0,
        data_res=None,
        thickness=None,
        bias_field_std=0.7,
        bias_scale=0.025,
        return_gradients=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize brain generator.
        
        Args:
            labels_dir: Directory containing label maps
            generation_labels: Labels used for generation
            n_neutral_labels: Number of neutral labels
            output_labels: Labels to output
            subjects_prob: Probability of selecting each subject
            batchsize: Batch size for generation
            n_channels: Number of channels to generate
            target_res: Target resolution
            output_shape: Target output shape
            output_div_by_n: Ensure output shape is divisible by this value
            prior_distributions: Type of prior distributions
            generation_classes: Classes for generation
            prior_means: Prior means for GMM
            prior_stds: Prior standard deviations for GMM
            use_specific_stats_for_channel: Whether to use specific stats for each channel
            mix_prior_and_random: Whether to mix prior and random stats
            flipping: Whether to apply random flipping
            scaling_bounds: Bounds for random scaling
            rotation_bounds: Bounds for random rotation
            shearing_bounds: Bounds for random shearing
            translation_bounds: Bounds for random translation
            nonlin_std: Standard deviation for nonlinear deformation
            nonlin_scale: Scale factor for nonlinear deformation
            randomise_res: Whether to randomize resolution
            max_res_iso: Maximum isotropic resolution
            max_res_aniso: Maximum anisotropic resolution
            data_res: Specific data resolution to mimic
            thickness: Slice thickness
            bias_field_std: Standard deviation for bias field
            bias_scale: Scale factor for bias field
            return_gradients: Whether to return image gradients
            device: Device to use for computation
        """
        self.device = device
        self.batchsize = batchsize
        
        # Create dataset
        self.dataset = BrainGeneratorDataset(
            labels_dir=labels_dir,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            output_labels=output_labels,
            subjects_prob=subjects_prob,
            n_channels=n_channels,
            generation_classes=generation_classes,
            prior_distributions=prior_distributions,
            prior_means=prior_means,
            prior_stds=prior_stds,
            use_specific_stats_for_channel=use_specific_stats_for_channel,
            mix_prior_and_random=mix_prior_and_random
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=0
        )
        
        # Create model
        self.model = BrainGeneratorModel(
            labels_shape=self.dataset.labels_shape,
            n_channels=n_channels,
            generation_labels=self.dataset.generation_labels,
            output_labels=self.dataset.output_labels,
            n_neutral_labels=self.dataset.n_neutral_labels,
            atlas_res=self.dataset._load_volume(self.dataset.labels_paths[0]).shape,  # This is a simplification
            target_res=target_res,
            output_shape=output_shape,
            output_div_by_n=output_div_by_n,
            flipping=flipping,
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            nonlin_scale=nonlin_scale,
            randomise_res=randomise_res,
            max_res_iso=max_res_iso,
            max_res_aniso=max_res_aniso,
            data_res=data_res,
            thickness=thickness,
            bias_field_std=bias_field_std,
            bias_scale=bias_scale,
            return_gradients=return_gradients
        ).to(device)
        
        # Get affine matrix from first label map (simplified)
        self.aff = np.eye(4)
        
        # Get dimensionality
        self.n_dims = len(self.dataset.labels_shape)
    
    def generate_brain(self):
        """
        Generate a synthetic brain image.
        
        Returns:
            Tuple of (image, labels)
        """
        # Get next batch of inputs
        batch = next(iter(self.dataloader))
        
        # Move to device
        batch = {k: torch.tensor(v, device=self.device) for k, v in batch.items()}
        
        # Generate image and labels
        with torch.no_grad():
            image, labels = self.model(batch)
        
        # Convert to numpy
        image = image.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Align to reference space
        aligned_images = []
        aligned_labels = []
        
        for i in range(self.batchsize):
            # In a real implementation, you would use proper alignment
            # This is a simplified version that just returns the data as is
            aligned_images.append(image[i])
            aligned_labels.append(labels[i])
        
        # Stack and squeeze
        image = np.squeeze(np.stack(aligned_images, axis=0))
        labels = np.squeeze(np.stack(aligned_labels, axis=0))
        
        return image, labels