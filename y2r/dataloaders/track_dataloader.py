import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
from pathlib import Path

from .base_dataset import SimpleBaseDataset
from .utils import NormalizationStats


def sample_points_uniform(query_coords, displacements, num_samples):
    """
    Uniformly sample num_samples points from available points.
    
    Args:
        query_coords: torch.Tensor of shape (total_points, coord_dim)
        displacements: torch.Tensor of shape (num_track_ts, total_points, coord_dim)
        num_samples: int, number of points to sample
    
    Returns:
        sampled_query_coords: (num_samples, coord_dim)
        sampled_displacements: (num_track_ts, num_samples, coord_dim)
    """
    total_points = query_coords.shape[0]
    
    if total_points <= num_samples:
        return query_coords, displacements
    
    indices = np.random.choice(total_points, num_samples, replace=False)
    indices = np.sort(indices)
    
    return query_coords[indices], displacements[:, indices, :]


class TrackDataset(SimpleBaseDataset):
    """
    Dataset for loading video frames and future track trajectories.
    Supports both 2D and 3D track formats with unified dict-based output.
    
    Returns dict with:
        imgs: (frame_stack, C, H, W) - past image observations
        query_coords: (N, 2) or (N, 3) - initial positions (u, v) or (u, v, d)
        displacements: (T, N, 2) or (T, N, 3) - future displacements
        poses: (T, 9) or None - camera poses (3D only)
        depth: (T, H, W) or None - depth maps (3D only, if available)
    """
    
    def __init__(self, h5_files=None, dataset_dir=None, img_size=224, num_track_ts=16, 
                 num_track_ids=64, frame_stack=1, downsample_factor=1, cache_all=False, cache_image=False, 
                 num_demos=None, aug_prob=0., track_type="2d",
                 aug_color_jitter=None, aug_translation_px=0, aug_rotation_deg=0,
                 aug_hflip_prob=0.0, aug_vflip_prob=0.0, aug_noise_std=0.0, aug_depth_noise_std=0.0):
        """
        Initialize TrackDataset.
        
        Args:
            h5_files: Optional list of specific .hdf5 file paths (for train/val split)
            dataset_dir: Optional directory to scan for .hdf5 files (alternative to h5_files)
            track_type: "2d" or "3d" - determines data format and normalization
            aug_*: Augmentation configuration parameters
            Other args: Same as SimpleBaseDataset
        """
        if h5_files is not None and dataset_dir is not None:
            raise ValueError("Provide either h5_files or dataset_dir, not both")
        elif h5_files is None and dataset_dir is None:
            raise ValueError("Either h5_files or dataset_dir must be provided")
        
        self._custom_h5_files = h5_files
        self.track_type = track_type
        
        # Determine dataset directory for stats loading
        if h5_files is not None:
            # Get directory from first h5 file
            stats_dir = Path(h5_files[0]).parent
            dataset_dir = []
        else:
            stats_dir = Path(dataset_dir) if isinstance(dataset_dir, str) else Path(dataset_dir[0])
        
        # Load normalization stats
        stats_path = stats_dir / "normalization_stats.yaml"
        if stats_path.exists():
            self.norm_stats = NormalizationStats(stats_path)
            print(f"Loaded normalization stats from {stats_path}")
        else:
            self.norm_stats = None
            print(f"Warning: No normalization_stats.yaml found in {stats_dir}")
        
        # Initialize base class attributes manually
        from torch.utils.data import Dataset
        Dataset.__init__(self)
        
        self.dataset_dir = dataset_dir if isinstance(dataset_dir, list) else [dataset_dir] if dataset_dir else []
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.frame_stack = frame_stack
        self.downsample_factor = downsample_factor
        self.num_demos = num_demos
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.aug_prob = aug_prob
        self.cache_all = cache_all
        self.cache_image = cache_image
        
        if not cache_all:
            assert not cache_image, "cache_image is only supported when cache_all is True."
        
        from .utils import load_rgb
        self.load_image_func = load_rgb
        
        # Setup buffer_fns from h5_files or scan directory
        if self._custom_h5_files is not None:
            self.buffer_fns = self._custom_h5_files
            print(f"Using {len(self.buffer_fns)} specified H5 files")
        else:
            from glob import glob
            from natsort import natsorted
            
            self.buffer_fns = []
            for d in self.dataset_dir:
                fn_list = glob(os.path.join(d, "*.hdf5"))
                fn_list = natsorted(fn_list)
                if self.num_demos is None:
                    n_demo = len(fn_list)
                else:
                    assert 0 < self.num_demos <= 1
                    n_demo = int(len(fn_list) * self.num_demos)
                for fn in fn_list[:n_demo]:
                    self.buffer_fns.append(fn)
            
            assert len(self.buffer_fns) > 0, f"No .hdf5 files found in {self.dataset_dir}"
            print(f"Found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")
        
        # Initialize caching and indexing structures
        self._cache = []
        self._index_to_demo_id = {}
        self._demo_id_to_path = {}
        self._demo_id_to_start_indices = {}
        self._demo_id_to_demo_length = {}
        self._demo_id_to_valid_indices = {}
        self._index_to_local_frame = {}
        
        # Load demo info
        self.load_demo_info()
        
        # Setup augmentation with new dict-based augmentors
        from torchvision import transforms
        from .utils import (DictColorJitter, DictTranslationAug, DictRandomRotate,
                           DictRandomHorizontalFlip, DictRandomVerticalFlip, DictGaussianNoise)
        
        aug_list = []
        
        # Color jitter
        if aug_color_jitter is not None:
            if isinstance(aug_color_jitter, dict):
                brightness = aug_color_jitter.get('brightness', 0)
                contrast = aug_color_jitter.get('contrast', 0)
                saturation = aug_color_jitter.get('saturation', 0)
                hue = aug_color_jitter.get('hue', 0)
            else:
                brightness = getattr(aug_color_jitter, 'brightness', 0)
                contrast = getattr(aug_color_jitter, 'contrast', 0)
                saturation = getattr(aug_color_jitter, 'saturation', 0)
                hue = getattr(aug_color_jitter, 'hue', 0)
            
            aug_list.append(DictColorJitter(
                brightness=brightness, contrast=contrast,
                saturation=saturation, hue=hue
            ))
        
        # Translation
        if aug_translation_px > 0:
            aug_list.append(DictTranslationAug(input_shape=img_size, translation=aug_translation_px))
        
        # Rotation
        if aug_rotation_deg > 0:
            aug_list.append(DictRandomRotate(degrees=aug_rotation_deg))
        
        # Horizontal flip
        if aug_hflip_prob > 0:
            aug_list.append(DictRandomHorizontalFlip(p=aug_hflip_prob))
        
        # Vertical flip
        if aug_vflip_prob > 0:
            aug_list.append(DictRandomVerticalFlip(p=aug_vflip_prob))
        
        # Gaussian noise
        if aug_noise_std > 0 or aug_depth_noise_std > 0:
            aug_list.append(DictGaussianNoise(img_std=aug_noise_std, depth_std=aug_depth_noise_std))
        
        self.augmentor = transforms.Compose(aug_list) if aug_list else None

    def _load_image_list_from_demo(self, demo, time_offset, num_frames=None, backward=False):
        """Load images from cached demo with temporal downsampling support."""
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"]["video"].shape[0]
        
        if backward:
            spacing = self.downsample_factor
            start_offset = time_offset - (num_frames - 1) * spacing
            image_indices = np.arange(start_offset, time_offset + 1, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            frames = demo['root']["video"][image_indices]
            
            if len(frames) < num_frames:
                first_frame = frames[0:1]
                num_padding = num_frames - len(frames)
                padding_frames = first_frame.repeat(num_padding, 1, 1, 1)
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            spacing = self.downsample_factor
            image_indices = np.arange(time_offset, time_offset + num_frames * spacing, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            return demo['root']["video"][image_indices]

    def _load_image_list_from_disk(self, demo_id, time_offset, num_frames=None, backward=False):
        """Load images from disk with temporal downsampling support."""
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        spacing = self.downsample_factor
        
        if backward:
            start_offset = time_offset - (num_frames - 1) * spacing
            image_indices = np.arange(start_offset, time_offset + 1, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{img_idx}.png")) for img_idx in image_indices]
            num_padding = num_frames - len(frames)
            if num_padding > 0:
                frames = [frames[0]] * num_padding + frames
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames * spacing, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)
        frames = torch.Tensor(frames)
        from einops import rearrange
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def __getitem__(self, index):
        """
        Load a single sample.
        
        Returns dict with:
            imgs: (frame_stack, C, H, W)
            query_coords: (N, 2) or (N, 3)
            displacements: (T, N, 2) or (T, N, 3)
            poses: (T, 9) or None
            depth: (T, H, W) or None
        """
        demo_id = self._index_to_demo_id[index]
        time_offset = self._index_to_local_frame[index]

        # Load demo
        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                imgs = self._load_image_list_from_demo(demo, time_offset, backward=True)
            else:
                imgs = self._load_image_list_from_disk(demo_id, time_offset, backward=True)
        else:
            demo_path = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_path))
            imgs = self._load_image_list_from_demo(demo, time_offset, backward=True)

        # Load track data at this timestep
        track_key = f"frame_{time_offset:04d}"
        frame_data = demo["root"]["tracks"][track_key]
        
        # Load query_coords and displacements
        query_coords = torch.Tensor(frame_data['query_coords'])  # (N, coord_dim)
        displacements = torch.Tensor(frame_data['displacements'])  # (T_stored, N, coord_dim)
        
        # Load poses (3D only)
        poses = None
        if self.track_type == '3d' and 'poses' in frame_data:
            poses = torch.Tensor(frame_data['poses'])  # (T_stored, 9)
        
        # Apply temporal downsampling to displacements and poses
        stored_num_track_ts = displacements.shape[0]
        required_stored_ts = self.num_track_ts * self.downsample_factor
        
        if stored_num_track_ts < required_stored_ts:
            raise ValueError(
                f"Requested {self.num_track_ts} track timesteps with {self.downsample_factor}x downsample "
                f"(requires {required_stored_ts} stored timesteps) but HDF5 only contains {stored_num_track_ts}."
            )
        
        if self.downsample_factor > 1:
            indices = torch.arange(0, required_stored_ts, self.downsample_factor)
            displacements = displacements[indices]
            if poses is not None:
                poses = poses[indices]
        else:
            indices = torch.arange(self.num_track_ts)
            displacements = displacements[:self.num_track_ts]
            if poses is not None:
                poses = poses[:self.num_track_ts]
        
        # Load depth (required for 3D) - slice to match track timesteps
        depth = None
        if self.track_type == '3d':
            if 'depth' not in demo['root']:
                raise ValueError(f"3D mode requires depth data but none found in {demo_id}")
            full_depth = demo['root']['depth']  # (T_video, H, W)
            # Extract frames starting from time_offset, matching the track timesteps
            depth_indices = time_offset + indices.numpy()
            depth = torch.Tensor(full_depth[depth_indices])  # (num_track_ts, H, W)
        
        # Data augmentation
        if self.augmentor is not None and np.random.rand() < self.aug_prob:
            # Create augmentation dict with batch dimension
            aug_data = {
                'imgs': (imgs / 255.0)[None],  # (1, T, C, H, W)
                'query_coords': query_coords[None],  # (1, N, coord_dim)
                'displacements': displacements[None],  # (1, T, N, coord_dim)
                'depth': depth[None] if depth is not None else None,  # (1, T, H, W)
                'poses': poses,  # Not augmented, keep as-is
            }
            
            # Apply augmentation
            aug_data = self.augmentor(aug_data)
            
            # Remove batch dimension
            imgs = aug_data['imgs'][0]
            query_coords = aug_data['query_coords'][0]
            displacements = aug_data['displacements'][0]
            if depth is not None:
                depth = aug_data['depth'][0]
        else:
            imgs = imgs / 255.0
        
        # Clip coordinates to valid [0, 1] range
        query_coords[..., :2] = torch.clamp(query_coords[..., :2], 0, 1)
        displacements[..., :2] = torch.clamp(displacements[..., :2], -1, 1)
        
        # Apply normalization
        if self.norm_stats is not None:
            if self.track_type == '3d':
                # Normalize depth in query_coords (3rd dimension)
                query_coords[:, 2] = self.norm_stats.normalize_depth_torch(query_coords[:, 2])
                # Normalize displacements
                displacements = self.norm_stats.normalize_displacement_torch(displacements)
                # Normalize poses
                if poses is not None:
                    poses = self.norm_stats.normalize_pose_torch(poses)
                # Normalize depth maps
                if depth is not None:
                    depth = self.norm_stats.normalize_depth_torch(depth)
            else:
                # 2D: just normalize displacements
                displacements = self.norm_stats.normalize_displacement_torch(displacements)
        
        # Apply ImageNet normalization for DINOv2
        imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Uniformly sample points
        query_coords, displacements = sample_points_uniform(
            query_coords, displacements, num_samples=self.num_track_ids
        )
        
        return {
            'imgs': imgs,
            'query_coords': query_coords,
            'displacements': displacements,
            'poses': poses,
            'depth': depth,
        }
