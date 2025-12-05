"""
GPU-accelerated augmentations using torchvision.transforms.v2 and torch.
Applied AFTER data is on GPU in the training loop for maximum efficiency.

Usage in train loop:
    augmenter = GPUAugmentations(...).to(device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if v is not None}
        if training and np.random.rand() < aug_prob:
            batch = augmenter(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torchvision.transforms import v2 as T
    TORCHVISION_V2_AVAILABLE = True
except ImportError:
    TORCHVISION_V2_AVAILABLE = False
    print("Warning: torchvision.transforms.v2 not available. Update torchvision.")


class GPUAugmentations(nn.Module):
    """
    GPU-accelerated augmentations for training.
    
    Combines rotation, translation, flip, color jitter, and noise into
    efficient batched GPU operations.
    
    Key optimizations:
    - Single affine transform combining rotation + translation + flip
    - torchvision.transforms.v2 for GPU-accelerated color jitter
    - All operations are batched and vectorized
    - Mean/std registered as buffers (no tensor creation overhead)
    
    Args:
        img_size: Image size (assumes square images)
        brightness: Color jitter brightness (0 = disabled)
        contrast: Color jitter contrast (0 = disabled)
        saturation: Color jitter saturation (0 = disabled)
        hue: Color jitter hue (0 = disabled)
        translation_px: Max translation in pixels
        rotation_deg: Max rotation in degrees
        hflip_prob: Horizontal flip probability
        vflip_prob: Vertical flip probability
        img_noise_std: Gaussian noise std for images
        depth_noise_std: Gaussian noise std for depth maps
    """
    
    def __init__(
        self,
        img_size=224,
        # Color
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        # Spatial
        translation_px=16,
        rotation_deg=20,
        hflip_prob=0.5,
        vflip_prob=0.25,
        # Noise
        img_noise_std=0.02,
        depth_noise_std=0.015,
    ):
        super().__init__()
        self.img_size = img_size
        self.translation = translation_px / img_size  # Normalize to [-1, 1] range
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.img_noise_std = img_noise_std
        self.depth_noise_std = depth_noise_std
        
        # Register ImageNet mean/std as buffers (avoids tensor creation overhead)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # torchvision.transforms.v2 color jitter (GPU-accelerated)
        self.use_color_jitter = TORCHVISION_V2_AVAILABLE and any([brightness, contrast, saturation, hue])
        if self.use_color_jitter:
            self.color_jitter = T.ColorJitter(
                brightness=brightness,
                contrast=contrast, 
                saturation=saturation,
                hue=hue,
            )
    
    @torch.no_grad()
    def forward(self, data):
        """
        Apply augmentations to batch on GPU.
        
        Args:
            data: dict with keys:
                - 'imgs': (B, T, C, H, W) - ImageNet normalized images
                - 'query_coords': (B, N, 2 or 3) - (u, v) or (u, v, d) in [0, 1]
                - 'displacements': (B, T, N, 2 or 3) - normalized displacements
                - 'depth': (B, T, H, W) or None - normalized depth maps
                - 'poses': (B, T, 9) or None - normalized poses (unchanged)
        
        Returns:
            Augmented data dict (same structure)
        """
        data = {k: v for k, v in data.items()}  # Shallow copy
        
        imgs = data['imgs']  # (B, T, C, H, W)
        B, T, C, H, W = imgs.shape
        device = imgs.device
        
        # Sample augmentation parameters for each batch item
        do_hflip = torch.rand(B, device=device) < self.hflip_prob
        do_vflip = torch.rand(B, device=device) < self.vflip_prob
        angles = (torch.rand(B, device=device) * 2 - 1) * self.rotation_deg
        tx = (torch.rand(B, device=device) * 2 - 1) * self.translation
        ty = (torch.rand(B, device=device) * 2 - 1) * self.translation
        
        # Build affine transformation matrix for each batch item
        # This combines rotation + translation into one operation (FAST!)
        angles_rad = angles * (math.pi / 180)
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)
        
        # Affine matrix: [cos, -sin, tx; sin, cos, ty]
        affine = torch.zeros(B, 2, 3, device=device, dtype=imgs.dtype)
        affine[:, 0, 0] = cos_a
        affine[:, 0, 1] = -sin_a
        affine[:, 0, 2] = tx * 2  # Scale to [-1, 1] grid space
        affine[:, 1, 0] = sin_a
        affine[:, 1, 1] = cos_a
        affine[:, 1, 2] = ty * 2
        
        # Apply horizontal flip by negating x-scale
        affine[do_hflip, 0, 0] *= -1
        affine[do_hflip, 0, 1] *= -1
        
        # Apply vertical flip by negating y-scale  
        affine[do_vflip, 1, 0] *= -1
        affine[do_vflip, 1, 1] *= -1
        
        # Compute inverse affine for coordinate transformation
        # grid_sample uses affine to map output→input, but for query_coords
        # we need input→output (the inverse)
        affine_3x3 = torch.eye(3, device=device, dtype=imgs.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        affine_3x3[:, :2, :] = affine
        affine_inv_3x3 = torch.linalg.inv(affine_3x3)
        affine_inv = affine_inv_3x3[:, :2, :]  # (B, 2, 3)
        
        # === TRANSFORM IMAGES (all frames at once) ===
        imgs_flat = imgs.reshape(B * T, C, H, W)
        # Expand affine to all frames
        affine_expanded = affine.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, 2, 3)
        
        grid = F.affine_grid(affine_expanded, imgs_flat.shape, align_corners=False)
        imgs_transformed = F.grid_sample(
            imgs_flat, grid, mode='bilinear', padding_mode='reflection', align_corners=False
        )
        data['imgs'] = imgs_transformed.reshape(B, T, C, H, W)
        
        # === COLOR JITTER (GPU via torchvision.transforms.v2) ===
        if self.use_color_jitter:
            # Denormalize from ImageNet, apply jitter, renormalize
            # torchvision.transforms.v2 expects [0, 1] images
            imgs_for_color = data['imgs'].reshape(B * T, C, H, W)
            imgs_denorm = imgs_for_color * self.std + self.mean  # [0, 1]
            imgs_denorm = imgs_denorm.clamp(0, 1)  # Prevent NaN in ColorJitter (HSV requires non-negative)
            imgs_jittered = self.color_jitter(imgs_denorm)
            imgs_renorm = (imgs_jittered - self.mean) / self.std  # Back to ImageNet norm
            data['imgs'] = imgs_renorm.reshape(B, T, C, H, W)
        
        # === TRANSFORM DEPTH ===
        if data.get('depth') is not None:
            depth = data['depth']  # (B, T, H, W)
            depth_flat = depth.reshape(B * T, 1, H, W)
            depth_transformed = F.grid_sample(
                depth_flat, grid, mode='nearest', padding_mode='reflection', align_corners=False
            )
            data['depth'] = depth_transformed.reshape(B, T, H, W)
        
        # === TRANSFORM QUERY_COORDS ===
        if data.get('query_coords') is not None:
            qc = data['query_coords'].clone()  # (B, N, 2 or 3)
            
            # Convert from [0,1] to [-1,1] for consistent math
            u = qc[..., 0] * 2 - 1  # (B, N)
            v = qc[..., 1] * 2 - 1
            
            # Apply INVERSE affine transform to coordinates
            # grid_sample maps output→input, so for coords we need input→output
            u_new = affine_inv[:, 0, 0:1] * u + affine_inv[:, 0, 1:2] * v + affine_inv[:, 0, 2:3]
            v_new = affine_inv[:, 1, 0:1] * u + affine_inv[:, 1, 1:2] * v + affine_inv[:, 1, 2:3]
            
            # Convert back to [0,1]
            qc[..., 0] = (u_new + 1) / 2
            qc[..., 1] = (v_new + 1) / 2
            data['query_coords'] = qc
        
        # === TRANSFORM DISPLACEMENTS ===
        if data.get('displacements') is not None:
            disp = data['displacements'].clone()  # (B, T, N, 2 or 3)
            
            # Displacements are vectors - apply INVERSE rotation only (no translation)
            du = disp[..., 0]  # (B, T, N)
            dv = disp[..., 1]
            
            # Extract inverse rotation part (no translation for vectors)
            rot_inv_00 = affine_inv[:, 0, 0].view(B, 1, 1)
            rot_inv_01 = affine_inv[:, 0, 1].view(B, 1, 1)
            rot_inv_10 = affine_inv[:, 1, 0].view(B, 1, 1)
            rot_inv_11 = affine_inv[:, 1, 1].view(B, 1, 1)
            
            disp[..., 0] = rot_inv_00 * du + rot_inv_01 * dv
            disp[..., 1] = rot_inv_10 * du + rot_inv_11 * dv
            # Note: dd (depth displacement) unchanged by 2D transforms
            
            data['displacements'] = disp
        
        # === GAUSSIAN NOISE ===
        if self.img_noise_std > 0:
            noise = torch.randn_like(data['imgs']) * self.img_noise_std
            data['imgs'] = data['imgs'] + noise
        
        if self.depth_noise_std > 0 and data.get('depth') is not None:
            noise = torch.randn_like(data['depth']) * self.depth_noise_std
            data['depth'] = data['depth'] + noise
        
        return data


def create_gpu_augmentations(cfg, device):
    """
    Create GPUAugmentations from config.
    
    Args:
        cfg: Config namespace with dataset_cfg containing augmentation params
        device: torch device
    
    Returns:
        GPUAugmentations instance on device
    """
    dataset_cfg = cfg.dataset_cfg
    
    # Extract color jitter params
    color_jitter = getattr(dataset_cfg, 'aug_color_jitter', None)
    if color_jitter is not None:
        if isinstance(color_jitter, dict):
            brightness = color_jitter.get('brightness', 0)
            contrast = color_jitter.get('contrast', 0)
            saturation = color_jitter.get('saturation', 0)
            hue = color_jitter.get('hue', 0)
        else:
            brightness = getattr(color_jitter, 'brightness', 0)
            contrast = getattr(color_jitter, 'contrast', 0)
            saturation = getattr(color_jitter, 'saturation', 0)
            hue = getattr(color_jitter, 'hue', 0)
    else:
        brightness = contrast = saturation = hue = 0
    
    augmenter = GPUAugmentations(
        img_size=dataset_cfg.img_size,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        translation_px=getattr(dataset_cfg, 'aug_translation_px', 0),
        rotation_deg=getattr(dataset_cfg, 'aug_rotation_deg', 0),
        hflip_prob=getattr(dataset_cfg, 'aug_hflip_prob', 0.0),
        vflip_prob=getattr(dataset_cfg, 'aug_vflip_prob', 0.0),
        img_noise_std=getattr(dataset_cfg, 'aug_noise_std', 0.0),
        depth_noise_std=getattr(dataset_cfg, 'aug_depth_noise_std', 0.0),
    ).to(device)
    
    return augmenter

