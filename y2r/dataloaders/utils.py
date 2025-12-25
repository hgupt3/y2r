"""
Dataloader utilities for track prediction.

Contains:
- NormalizationStats: Load and apply normalization from stats file
- get_dataloader: Create a DataLoader with standard settings
- load_rgb: Load RGB image from file

"""

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
import yaml


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
            
            # Hand pose stats (optional - only present if hand data was processed)
            if 'hand_uvd_disp_mean' in self.stats:
                self.hand_uvd_disp_mean = np.array(self.stats['hand_uvd_disp_mean'])
                self.hand_uvd_disp_std = np.array(self.stats['hand_uvd_disp_std'])
                self.hand_rot_disp_mean = np.array(self.stats['hand_rot_disp_mean'])
                self.hand_rot_disp_std = np.array(self.stats['hand_rot_disp_std'])
                self.has_hand_stats = True
            else:
                self.has_hand_stats = False
    
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
    
    def denormalize_depth_torch(self, d):
        """Denormalize depth values (torch tensor)."""
        return d * (self.depth_std + 1e-8) + self.depth_mean
    
    def denormalize_displacement_torch(self, disp):
        """Denormalize displacement values (torch tensor)."""
        mean = torch.tensor(self.disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.disp_std, dtype=disp.dtype, device=disp.device) + 1e-8
        return disp * std + mean
    
    def denormalize_pose_torch(self, pose):
        """Denormalize 9D pose values (torch tensor)."""
        mean = torch.tensor(self.pose_mean, dtype=pose.dtype, device=pose.device)
        std = torch.tensor(self.pose_std, dtype=pose.dtype, device=pose.device) + 1e-8
        return pose * std + mean
    
    # Hand UVD displacement normalization
    def normalize_hand_uvd_disp(self, disp):
        """Normalize hand UVD displacement values."""
        mean = self.hand_uvd_disp_mean.astype(np.float32)
        std = self.hand_uvd_disp_std.astype(np.float32) + 1e-8
        return (disp - mean) / std
    
    def denormalize_hand_uvd_disp(self, disp):
        """Denormalize hand UVD displacement values."""
        mean = self.hand_uvd_disp_mean.astype(np.float32)
        std = self.hand_uvd_disp_std.astype(np.float32)
        return disp * std + mean
    
    def normalize_hand_uvd_disp_torch(self, disp):
        """Normalize hand UVD displacement values (torch tensor)."""
        mean = torch.tensor(self.hand_uvd_disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.hand_uvd_disp_std, dtype=disp.dtype, device=disp.device) + 1e-8
        return (disp - mean) / std
    
    def denormalize_hand_uvd_disp_torch(self, disp):
        """Denormalize hand UVD displacement values (torch tensor)."""
        mean = torch.tensor(self.hand_uvd_disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.hand_uvd_disp_std, dtype=disp.dtype, device=disp.device)
        return disp * std + mean
    
    # Hand rotation displacement normalization
    def normalize_hand_rot_disp(self, disp):
        """Normalize hand 6D rotation displacement values."""
        mean = self.hand_rot_disp_mean.astype(np.float32)
        std = self.hand_rot_disp_std.astype(np.float32) + 1e-8
        return (disp - mean) / std
    
    def denormalize_hand_rot_disp(self, disp):
        """Denormalize hand 6D rotation displacement values."""
        mean = self.hand_rot_disp_mean.astype(np.float32)
        std = self.hand_rot_disp_std.astype(np.float32)
        return disp * std + mean
    
    def normalize_hand_rot_disp_torch(self, disp):
        """Normalize hand 6D rotation displacement values (torch tensor)."""
        mean = torch.tensor(self.hand_rot_disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.hand_rot_disp_std, dtype=disp.dtype, device=disp.device) + 1e-8
        return (disp - mean) / std
    
    def denormalize_hand_rot_disp_torch(self, disp):
        """Denormalize hand 6D rotation displacement values (torch tensor)."""
        mean = torch.tensor(self.hand_rot_disp_mean, dtype=disp.dtype, device=disp.device)
        std = torch.tensor(self.hand_rot_disp_std, dtype=disp.dtype, device=disp.device)
        return disp * std + mean


# ==============================================================================
# DATALOADER UTILITIES
# ==============================================================================

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
    """Load RGB image from file as numpy array."""
    return np.array(Image.open(file_name))
