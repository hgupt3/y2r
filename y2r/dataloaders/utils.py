import numpy as np
from PIL import Image
from einops import repeat
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml
from pathlib import Path

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_core import CropRandomizer


# ==============================================================================
# NORMALIZATION UTILITIES
# ==============================================================================

class NormalizationStats:
    """Load and apply normalization from stats file."""
    
    def __init__(self, stats_path):
        """
        Load normalization statistics from YAML file.
        
        Args:
            stats_path: Path to normalization_stats.yaml
        """
        with open(stats_path, 'r') as f:
            self.stats = yaml.safe_load(f)
        
        # Extract stats - no defaults, must be present in stats file
        self.imagenet_mean = np.array(self.stats['imagenet_mean'])
        self.imagenet_std = np.array(self.stats['imagenet_std'])
        self.track_type = self.stats['track_type']
        self.track_dim = self.stats['track_dim']
        
        self.disp_mean = np.array(self.stats['displacement_mean'])
        self.disp_std = np.array(self.stats['displacement_std'])
        
        # 3D-specific stats (only required for 3D)
        if self.track_type == '3d':
            self.depth_mean = self.stats['depth_mean']
            self.depth_std = self.stats['depth_std']
            self.pose_mean = np.array(self.stats['pose_mean'])
            self.pose_std = np.array(self.stats['pose_std'])
    
    # Depth normalization
    def normalize_depth(self, d):
        """Normalize depth values."""
        return (d - self.depth_mean) / (self.depth_std + 1e-8)
    
    def denormalize_depth(self, d):
        """Denormalize depth values."""
        return d * self.depth_std + self.depth_mean
    
    # Displacement normalization
    def normalize_displacement(self, disp):
        """Normalize displacement values."""
        mean = self.disp_mean.astype(np.float32)
        std = self.disp_std.astype(np.float32) + 1e-8
        return (disp - mean) / std
    
    def denormalize_displacement(self, disp):
        """Denormalize displacement values."""
        mean = self.disp_mean.astype(np.float32)
        std = self.disp_std.astype(np.float32)
        return disp * std + mean
    
    # Pose normalization
    def normalize_pose(self, pose):
        """Normalize 9D pose values."""
        mean = self.pose_mean.astype(np.float32)
        std = self.pose_std.astype(np.float32) + 1e-8
        return (pose - mean) / std
    
    def denormalize_pose(self, pose):
        """Denormalize 9D pose values."""
        mean = self.pose_mean.astype(np.float32)
        std = self.pose_std.astype(np.float32)
        return pose * std + mean
    
    # Torch versions for use in forward pass
    def normalize_depth_torch(self, d):
        """Normalize depth values (torch tensor)."""
        return (d - self.depth_mean) / (self.depth_std + 1e-8)
    
    def normalize_displacement_torch(self, disp):
        """Normalize displacement values (torch tensor)."""
        mean = torch.tensor(self.disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.disp_std, dtype=disp.dtype, device=disp.device) + 1e-8
        return (disp - mean) / std
    
    def normalize_pose_torch(self, pose):
        """Normalize 9D pose values (torch tensor)."""
        mean = torch.tensor(self.pose_mean, dtype=pose.dtype, device=pose.device)
        std = torch.tensor(self.pose_std, dtype=pose.dtype, device=pose.device) + 1e-8
        return (pose - mean) / std


def get_dataloader(replay, mode, num_workers, batch_size):
    loader = DataLoader(
        replay,
        shuffle=(mode == "train"),
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None
    )
    return loader


def load_rgb(file_name):
    return np.array(Image.open(file_name))


# ==============================================================================
# AUGMENTATION CLASSES (Dict-based for unified 2D/3D format)
# ==============================================================================
# Input format: dict with keys:
#   - imgs: (b, t, C, H, W)
#   - query_coords: (b, N, 2) or (b, N, 3) - (u, v) or (u, v, d)
#   - displacements: (b, T, N, 2) or (b, T, N, 3) - (du, dv) or (du, dv, dd)
#   - depth: (b, T, H, W) or None
#   - poses: (b, T, 9) or None

import torchvision.transforms.functional as TF
import math


class DictColorJitter(torchvision.transforms.ColorJitter):
    """Color jitter augmentation - only affects images."""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, data):
        data = data.copy()
        data['imgs'] = super().forward(data['imgs'])
        return data


class CropRandomizerReturnCoords(CropRandomizer):
    def _forward_in(self, inputs, return_crop_inds=False):
        assert len(inputs.shape) >= 3
        out, crop_inds = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        if return_crop_inds:
            return TensorUtils.join_dimensions(out, 0, 1), crop_inds
        else:
            return TensorUtils.join_dimensions(out, 0, 1)


class DictTranslationAug(nn.Module):
    """
    Translation augmentation.
    - Shifts images spatially
    - Shifts query_coords (u, v)
    - Displacements unchanged (relative motion stays the same)
    - Shifts depth spatially
    - Poses unchanged
    """

    def __init__(self, input_shape, translation):
        super().__init__()
        self.pad_translation = translation // 2
        self.input_shape = input_shape
        pad_output_shape = (3, input_shape[0] + translation, input_shape[1] + translation)
        self.crop_randomizer = CropRandomizerReturnCoords(
            input_shape=pad_output_shape,
            crop_height=input_shape[0],
            crop_width=input_shape[1],
        )

    def forward(self, data):
        data = data.copy()
        imgs = data['imgs']
        
        batch_size, temporal_len, img_c, img_h, img_w = imgs.shape
        imgs_flat = imgs.reshape(batch_size, temporal_len * img_c, img_h, img_w)
        imgs_padded = F.pad(imgs_flat, pad=(self.pad_translation,) * 4, mode="replicate")
        imgs_cropped, crop_inds = self.crop_randomizer._forward_in(imgs_padded, return_crop_inds=True)
        data['imgs'] = imgs_cropped.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        
        # Compute translation in normalized coords
        translate_h = (crop_inds[:, 0, 0] - self.pad_translation) / img_h  # (b,)
        translate_w = (crop_inds[:, 0, 1] - self.pad_translation) / img_w
        
        # Shift query_coords (u, v) - only first 2 dimensions
        if data.get('query_coords') is not None:
            query_coords = data['query_coords'].clone()
            query_coords[..., 0] -= translate_w.unsqueeze(-1)  # u
            query_coords[..., 1] -= translate_h.unsqueeze(-1)  # v
            data['query_coords'] = query_coords
        
        # Displacements unchanged - relative motion stays the same
        
        # Shift depth spatially if present
        if data.get('depth') is not None:
            depth = data['depth']
            b, t, h, w = depth.shape
            depth_flat = depth.reshape(b, t, 1, h, w).reshape(b * t, 1, h, w)
            depth_padded = F.pad(depth_flat, pad=(self.pad_translation,) * 4, mode="replicate")
            # Apply same crop
            depth_cropped = []
            for i in range(b):
                crop_h, crop_w = crop_inds[i, 0, 0].item(), crop_inds[i, 0, 1].item()
                depth_cropped.append(depth_padded[i*t:(i+1)*t, :, crop_h:crop_h+h, crop_w:crop_w+w])
            data['depth'] = torch.cat(depth_cropped, dim=0).reshape(b, t, h, w)
        
        return data


class DictRandomRotate(nn.Module):
    """
    Rotation augmentation.
    - Rotates images
    - Rotates query_coords (u, v) around center
    - Rotates displacements (du, dv) - direction changes
    - Rotates depth spatially
    - Poses unchanged
    """

    def __init__(self, degrees=30):
        super().__init__()
        self.degrees = degrees

    def forward(self, data):
        data = data.copy()
        
        # Sample random angle
        angle = torch.rand(1).item() * 2 * self.degrees - self.degrees
        angle_rad = math.radians(-angle)  # Negate for image coordinates
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotate images
        imgs = data['imgs']
        b, t, c, h, w = imgs.shape
        imgs_flat = imgs.reshape(b * t, c, h, w)
        imgs_rotated = TF.rotate(imgs_flat, angle, interpolation=TF.InterpolationMode.BILINEAR)
        data['imgs'] = imgs_rotated.reshape(b, t, c, h, w)
        
        # Rotate query_coords (u, v) around center (0.5, 0.5)
        if data.get('query_coords') is not None:
            qc = data['query_coords'].clone()
            u_centered = qc[..., 0] - 0.5
            v_centered = qc[..., 1] - 0.5
            qc[..., 0] = u_centered * cos_a - v_centered * sin_a + 0.5
            qc[..., 1] = u_centered * sin_a + v_centered * cos_a + 0.5
            data['query_coords'] = qc
        
        # Rotate displacements (du, dv) - these are vectors, rotate without centering
        if data.get('displacements') is not None:
            disp = data['displacements'].clone()
            du = disp[..., 0]
            dv = disp[..., 1]
            disp[..., 0] = du * cos_a - dv * sin_a
            disp[..., 1] = du * sin_a + dv * cos_a
            data['displacements'] = disp
        
        # Rotate depth spatially
        if data.get('depth') is not None:
            depth = data['depth']
            b, t, h, w = depth.shape
            depth_flat = depth.reshape(b * t, 1, h, w)
            depth_rotated = TF.rotate(depth_flat, angle, interpolation=TF.InterpolationMode.NEAREST)
            data['depth'] = depth_rotated.reshape(b, t, h, w)
        
        return data


class DictRandomHorizontalFlip(nn.Module):
    """
    Horizontal flip augmentation.
    - Flips images horizontally
    - Flips query_coords u: u' = 1 - u
    - Negates displacements du: du' = -du
    - Flips depth horizontally
    - Poses unchanged
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() >= self.p:
            return data
        
        data = data.copy()
        
        # Flip images
        imgs = data['imgs']
        b, t, c, h, w = imgs.shape
        imgs_flat = imgs.reshape(b * t, c, h, w)
        data['imgs'] = TF.hflip(imgs_flat).reshape(b, t, c, h, w)
        
        # Flip query_coords u
        if data.get('query_coords') is not None:
            qc = data['query_coords'].clone()
            qc[..., 0] = 1.0 - qc[..., 0]
            data['query_coords'] = qc
        
        # Negate displacements du
        if data.get('displacements') is not None:
            disp = data['displacements'].clone()
            disp[..., 0] = -disp[..., 0]
            data['displacements'] = disp
        
        # Flip depth
        if data.get('depth') is not None:
            depth = data['depth']
            b, t, h, w = depth.shape
            depth_flat = depth.reshape(b * t, 1, h, w)
            data['depth'] = TF.hflip(depth_flat).reshape(b, t, h, w)
        
        return data


class DictRandomVerticalFlip(nn.Module):
    """
    Vertical flip augmentation.
    - Flips images vertically
    - Flips query_coords v: v' = 1 - v
    - Negates displacements dv: dv' = -dv
    - Flips depth vertically
    - Poses unchanged
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1).item() >= self.p:
            return data
        
        data = data.copy()
        
        # Flip images
        imgs = data['imgs']
        b, t, c, h, w = imgs.shape
        imgs_flat = imgs.reshape(b * t, c, h, w)
        data['imgs'] = TF.vflip(imgs_flat).reshape(b, t, c, h, w)
        
        # Flip query_coords v
        if data.get('query_coords') is not None:
            qc = data['query_coords'].clone()
            qc[..., 1] = 1.0 - qc[..., 1]
            data['query_coords'] = qc
        
        # Negate displacements dv
        if data.get('displacements') is not None:
            disp = data['displacements'].clone()
            disp[..., 1] = -disp[..., 1]
            data['displacements'] = disp
        
        # Flip depth
        if data.get('depth') is not None:
            depth = data['depth']
            b, t, h, w = depth.shape
            depth_flat = depth.reshape(b * t, 1, h, w)
            data['depth'] = TF.vflip(depth_flat).reshape(b, t, h, w)
        
        return data


class DictGaussianNoise(nn.Module):
    """
    Gaussian noise augmentation for images and depth.
    """

    def __init__(self, img_std=0.02, depth_std=0.02):
        super().__init__()
        self.img_std = img_std
        self.depth_std = depth_std

    def forward(self, data):
        data = data.copy()
        
        # Add noise to images
        if self.img_std > 0:
            imgs = data['imgs']
            noise = torch.randn_like(imgs) * self.img_std
            data['imgs'] = torch.clamp(imgs + noise, 0, 1)
        
        # Add noise to depth
        if self.depth_std > 0 and data.get('depth') is not None:
            depth = data['depth']
            noise = torch.randn_like(depth) * self.depth_std
            data['depth'] = depth + noise  # Don't clamp depth
        
        return data


