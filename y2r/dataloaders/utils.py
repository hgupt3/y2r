import numpy as np
from PIL import Image
from einops import repeat
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_core import CropRandomizer


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


class ImgTrackColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, inputs):
        img, tracks = inputs
        img = super().forward(img)
        return img, tracks


class CropRandomizerReturnCoords(CropRandomizer):
    def _forward_in(self, inputs, return_crop_inds=False):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
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
            # [B, N, ...] -> [B * N, ...]
            return TensorUtils.join_dimensions(out, 0, 1)


class ImgViewDiffTranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic. Take the same translation for a batch of images.
    """

    def __init__(
        self,
        input_shape,
        translation,
        augment_track=True,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            3,
            input_shape[0] + translation,
            input_shape[1] + translation,
        )

        self.crop_randomizer = CropRandomizerReturnCoords(
            input_shape=pad_output_shape,
            crop_height=input_shape[0],
            crop_width=input_shape[1],
        )
        self.augment_track = augment_track

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W]
            tracks: [b, t, track_len, n, 2]
        """
        img, tracks = inputs

        batch_size, temporal_len, img_c, img_h, img_w = img.shape
        img = img.reshape(batch_size, temporal_len * img_c, img_h, img_w)
        out = F.pad(img, pad=(self.pad_translation,) * 4, mode="replicate")
        out, crop_inds = self.crop_randomizer._forward_in(out, return_crop_inds=True)  # crop_inds: (b, num_crops, 2), where we already set num_crops=1
        out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)

        if self.augment_track:
            translate_h = (crop_inds[:, 0, 0] - self.pad_translation) / img_h  # (b,)
            translate_w = (crop_inds[:, 0, 1] - self.pad_translation) / img_w
            translate_h = repeat(translate_h, "b -> b 1 1 1")  # (b, 1, 1, 1)
            translate_w = repeat(translate_w, "b -> b 1 1 1")

            # tracks[..., 0] is x (width), tracks[..., 1] is y (height)
            # When cropping, points move in the OPPOSITE direction of crop offset
            tracks[..., 0] -= translate_w  # x gets negative width translation
            tracks[..., 1] -= translate_h  # y gets negative height translation

        return out, tracks


class ImgTrackRandomRotate(nn.Module):
    """
    Random rotation augmentation for images and tracks.
    Rotates image and applies corresponding 2D rotation to track coordinates.
    """

    def __init__(self, degrees=30):
        """
        Args:
            degrees: rotation range will be [-degrees, +degrees]
        """
        super().__init__()
        self.degrees = degrees

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W]
            tracks: [b, t, track_len, n, 2] where [..., 0] is x, [..., 1] is y
        """
        import torch
        import torchvision.transforms.functional as TF
        import math
        
        img, tracks = inputs
        
        # Sample random rotation angle (same for entire batch)
        angle = torch.rand(1).item() * 2 * self.degrees - self.degrees
        
        # Rotate image
        batch_size, temporal_len, img_c, img_h, img_w = img.shape
        img_flat = img.reshape(batch_size * temporal_len, img_c, img_h, img_w)
        img_rotated = TF.rotate(img_flat, angle, interpolation=TF.InterpolationMode.BILINEAR)
        img_rotated = img_rotated.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        
        # Rotate tracks around center (0.5, 0.5)
        # Note: torchvision rotate uses counter-clockwise rotation in standard coords
        # But in image coords where y increases downward, we need to negate the angle
        angle_rad = math.radians(-angle)  # Negate for image coordinates
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Extract coordinates (tracks stored as [x, y])
        x_centered = tracks[..., 0] - 0.5
        y_centered = tracks[..., 1] - 0.5
        
        # Apply 2D rotation matrix: [x', y'] = R * [x, y]
        # x' = x*cos(θ) - y*sin(θ)
        # y' = x*sin(θ) + y*cos(θ)
        x_new = x_centered * cos_a - y_centered * sin_a + 0.5
        y_new = x_centered * sin_a + y_centered * cos_a + 0.5
        
        tracks_rotated = tracks.clone()
        tracks_rotated[..., 0] = x_new
        tracks_rotated[..., 1] = y_new
        
        return img_rotated, tracks_rotated


class ImgTrackRandomHorizontalFlip(nn.Module):
    """
    Random horizontal flip augmentation for images and tracks.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p: probability of applying the flip
        """
        super().__init__()
        self.p = p

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W]
            tracks: [b, t, track_len, n, 2] where [..., 0] is x, [..., 1] is y
        """
        import torch
        import torchvision.transforms.functional as TF
        
        img, tracks = inputs
        
        # Decide whether to flip (same for entire batch)
        if torch.rand(1).item() < self.p:
            # Flip image
            batch_size, temporal_len, img_c, img_h, img_w = img.shape
            img_flat = img.reshape(batch_size * temporal_len, img_c, img_h, img_w)
            img_flipped = TF.hflip(img_flat)
            img_flipped = img_flipped.reshape(batch_size, temporal_len, img_c, img_h, img_w)
            
            # Flip tracks horizontally (flip x-coordinate)
            tracks_flipped = tracks.clone()
            tracks_flipped[..., 0] = 1.0 - tracks[..., 0]
            
            return img_flipped, tracks_flipped
        else:
            return img, tracks


class ImgTrackRandomVerticalFlip(nn.Module):
    """
    Random vertical flip augmentation for images and tracks.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p: probability of applying the flip
        """
        super().__init__()
        self.p = p

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W]
            tracks: [b, t, track_len, n, 2] where [..., 0] is x, [..., 1] is y
        """
        import torch
        import torchvision.transforms.functional as TF
        
        img, tracks = inputs
        
        # Decide whether to flip (same for entire batch)
        if torch.rand(1).item() < self.p:
            # Flip image
            batch_size, temporal_len, img_c, img_h, img_w = img.shape
            img_flat = img.reshape(batch_size * temporal_len, img_c, img_h, img_w)
            img_flipped = TF.vflip(img_flat)
            img_flipped = img_flipped.reshape(batch_size, temporal_len, img_c, img_h, img_w)
            
            # Flip tracks vertically (flip y-coordinate)
            tracks_flipped = tracks.clone()
            tracks_flipped[..., 1] = 1.0 - tracks[..., 1]
            
            return img_flipped, tracks_flipped
        else:
            return img, tracks


class ImgTrackGaussianNoise(nn.Module):
    """
    Gaussian noise augmentation for images (tracks unchanged).
    Adds random Gaussian noise to simulate camera sensor noise.
    """

    def __init__(self, std=0.02):
        """
        Args:
            std: standard deviation of Gaussian noise (0 = disabled)
        """
        super().__init__()
        self.std = std

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W] - normalized images in [0, 1] range
            tracks: [b, t, track_len, n, 2]
        """
        import torch
        
        img, tracks = inputs
        
        # Only apply if std > 0
        if self.std > 0:
            # Add Gaussian noise
            noise = torch.randn_like(img) * self.std
            img_noisy = img + noise
            # Clamp to valid range [0, 1] (before ImageNet normalization)
            img_noisy = torch.clamp(img_noisy, 0, 1)
            return img_noisy, tracks
        else:
            return img, tracks
