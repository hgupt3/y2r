import numpy as np
import os
import torch
import torchvision.transforms.functional as TF

from .base_dataset import SimpleBaseDataset


def sample_tracks_uniform(tracks, num_samples):
    """
    Uniformly sample num_samples tracks from available tracks.
    
    Args:
        tracks: torch.Tensor of shape (num_track_ts, total_points, 2)
        num_samples: int, number of tracks to sample
    
    Returns:
        sampled_tracks: torch.Tensor of shape (num_track_ts, num_samples, 2)
    """
    num_track_ts, total_points, _ = tracks.shape
    
    if total_points <= num_samples:
        # If we have fewer points than requested, return all
        return tracks
    
    # Uniformly sample without replacement
    indices = np.random.choice(total_points, num_samples, replace=False)
    indices = np.sort(indices)  # Sort to maintain some consistency
    
    return tracks[:, indices, :]


class TrackDataset(SimpleBaseDataset):
    """
    Dataset for loading video frames and future track trajectories.
    
    Returns:
        imgs: torch.Tensor of shape (frame_stack, C, H, W) - past image observations
        tracks: torch.Tensor of shape (num_track_ts, num_track_ids, 2) - future track predictions
    """
    
    def __init__(self, h5_files=None, dataset_dir=None, img_size=224, num_track_ts=16, 
                 num_track_ids=64, frame_stack=1, downsample_factor=1, cache_all=False, cache_image=False, 
                 num_demos=None, aug_prob=0.,
                 aug_color_jitter=None, aug_translation_px=0, aug_rotation_deg=0,
                 aug_hflip_prob=0.0, aug_vflip_prob=0.0, aug_noise_std=0.0):
        """
        Initialize TrackDataset.
        
        Args:
            h5_files: Optional list of specific .hdf5 file paths (for train/val split)
            dataset_dir: Optional directory to scan for .hdf5 files (alternative to h5_files)
            aug_*: Augmentation configuration parameters
            Other args: Same as SimpleBaseDataset
        """
        if h5_files is not None and dataset_dir is not None:
            raise ValueError("Provide either h5_files or dataset_dir, not both")
        elif h5_files is None and dataset_dir is None:
            raise ValueError("Either h5_files or dataset_dir must be provided")
        
        # Store h5_files before calling parent init
        self._custom_h5_files = h5_files
        
        # If h5_files is provided, we need to bypass parent's file scanning
        # Pass empty list to prevent scanning, we'll populate buffer_fns manually
        if h5_files is not None:
            dataset_dir = []
        
        # Manually initialize what the parent would do, but skip file scanning
        from torch.utils.data import Dataset
        Dataset.__init__(self)  # Initialize torch Dataset
        
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
            # Scan directories for .hdf5 files (original behavior)
            import os
            from glob import glob
            from natsort import natsorted
            
            self.buffer_fns = []
            for d in self.dataset_dir:
                fn_list = glob(os.path.join(d, "*.hdf5"))
                fn_list = natsorted(fn_list)
                if self.num_demos is None:
                    n_demo = len(fn_list)
                else:
                    assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
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
        
        # Load demo info (same as parent)
        self.load_demo_info()
        
        # Setup augmentation
        from torchvision import transforms
        from .utils import (ImgTrackColorJitter, ImgViewDiffTranslationAug, 
                           ImgTrackRandomRotate, ImgTrackRandomHorizontalFlip,
                           ImgTrackRandomVerticalFlip, ImgTrackGaussianNoise)
        
        # Conditionally build augmentor based on config (0 or None means disabled)
        aug_list = []
        
        # Color jitter (dict or Namespace with brightness, contrast, saturation, hue)
        if aug_color_jitter is not None:
            # Handle both dict and Namespace
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
            
            aug_list.append(ImgTrackColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
        
        # Translation (in pixels)
        if aug_translation_px > 0:
            aug_list.append(ImgViewDiffTranslationAug(
                input_shape=img_size,
                translation=aug_translation_px,
                augment_track=True
            ))
        
        # Rotation (in degrees)
        if aug_rotation_deg > 0:
            aug_list.append(ImgTrackRandomRotate(degrees=aug_rotation_deg))
        
        # Horizontal flip (probability)
        if aug_hflip_prob > 0:
            aug_list.append(ImgTrackRandomHorizontalFlip(p=aug_hflip_prob))
        
        # Vertical flip (probability)
        if aug_vflip_prob > 0:
            aug_list.append(ImgTrackRandomVerticalFlip(p=aug_vflip_prob))
        
        # Gaussian noise (std)
        if aug_noise_std > 0:
            aug_list.append(ImgTrackGaussianNoise(std=aug_noise_std))
        
        # Only create augmentor if there are augmentations to apply
        self.augmentor = transforms.Compose(aug_list) if aug_list else None

    def _load_image_list_from_demo(self, demo, time_offset, num_frames=None, backward=False):
        """
        Load images from cached demo with temporal downsampling support.
        
        Args:
            demo: Cached demo dict
            time_offset: Current frame index
            num_frames: Number of frames to load (defaults to frame_stack)
            backward: If True, load past frames with spacing based on downsample_factor
        """
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"]["video"].shape[0]
        
        if backward:
            # Load frames with spacing based on downsample_factor
            # For downsample_factor=1: [t-1, t] (consecutive)
            # For downsample_factor=2: [t-2, t] (skip 1 frame)
            spacing = self.downsample_factor
            start_offset = time_offset - (num_frames - 1) * spacing
            image_indices = np.arange(start_offset, time_offset + 1, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            frames = demo['root']["video"][image_indices]
            
            # Pad with first frame if needed (when at start of trajectory)
            if len(frames) < num_frames:
                first_frame = frames[0:1]  # Keep dims: (1, C, H, W)
                num_padding = num_frames - len(frames)
                padding_frames = first_frame.repeat(num_padding, 1, 1, 1)
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            # Forward loading with spacing (if needed)
            spacing = self.downsample_factor
            image_indices = np.arange(time_offset, time_offset + num_frames * spacing, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            return demo['root']["video"][image_indices]

    def _load_image_list_from_disk(self, demo_id, time_offset, num_frames=None, backward=False):
        """
        Load images from disk with temporal downsampling support.
        
        Args:
            demo_id: Demo ID
            time_offset: Current frame index
            num_frames: Number of frames to load (defaults to frame_stack)
            backward: If True, load past frames with spacing based on downsample_factor
        """
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
            # Pad with first frame if needed (when at start of trajectory)
            num_padding = num_frames - len(frames)
            if num_padding > 0:
                frames = [frames[0]] * num_padding + frames
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames * spacing, spacing)
            image_indices = np.clip(image_indices, 0, demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # T, H, W, C
        frames = torch.Tensor(frames)
        from einops import rearrange
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def __getitem__(self, index):
        """
        Load a single sample.
        
        Args:
            index: int, global index into the dataset
            
        Returns:
            imgs: (frame_stack, C, H, W) - image observations at time t
            tracks: (num_track_ts, num_track_ids, 2) - future tracks from time t
        """
        demo_id = self._index_to_demo_id[index]
        time_offset = self._index_to_local_frame[index]  # Use actual frame index in video

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

        # Load future tracks at this timestep
        # tracks shape: (stored_num_track_ts, num_points, 2)
        track_key = f"frame_{time_offset:04d}"
        tracks = demo["root"]["tracks"][track_key]  # Load from per-frame dataset
        tracks = torch.Tensor(tracks)
        
        # Apply temporal downsampling
        stored_num_track_ts = tracks.shape[0]
        required_stored_ts = self.num_track_ts * self.downsample_factor
        
        if stored_num_track_ts < required_stored_ts:
            raise ValueError(
                f"Requested {self.num_track_ts} track timesteps with {self.downsample_factor}x downsample "
                f"(requires {required_stored_ts} stored timesteps) but HDF5 only contains {stored_num_track_ts}. "
                f"Please regenerate data with more timesteps or reduce num_track_ts/downsample_factor in config."
            )
        
        # Downsample: take indices [0, step, 2*step, ...]
        if self.downsample_factor > 1:
            indices = torch.arange(0, required_stored_ts, self.downsample_factor)
            tracks = tracks[indices]  # Shape: (num_track_ts, num_points, 2)
        else:
            # No downsampling, just truncate to num_track_ts
            tracks = tracks[:self.num_track_ts]
        
        # Data augmentation
        if self.augmentor is not None and np.random.rand() < self.aug_prob:
            # Expand dimensions for augmentor
            # imgs: (frame_stack, c, h, w) -> (1, frame_stack, c, h, w)
            # tracks: (num_track_ts, n, 2) -> (1, 1, num_track_ts, n, 2)
            imgs = imgs[None]  # Add batch dimension
            tracks = tracks[None, None]  # Add batch and view dimensions
            
            # Apply augmentation (expects normalized images)
            imgs, tracks = self.augmentor((imgs / 255., tracks))
            
            # Remove extra dimensions
            imgs = imgs[0]  # Back to (frame_stack, c, h, w) in [0, 1] range
            tracks = tracks[0, 0]  # Back to (num_track_ts, n, 2)
        else:
            # No augmentation, normalize to [0, 1]
            imgs = imgs / 255.0
        
        # Clip tracks to valid [0, 1] range (handles out-of-bounds after augmentation)
        tracks = torch.clamp(tracks, 0, 1)
        
        # Apply ImageNet normalization for DINOv2
        # imgs: (frame_stack, c, h, w) in [0, 1] range
        imgs = TF.normalize(
            imgs,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Uniformly sample tracks to get num_track_ids tracks
        tracks = sample_tracks_uniform(tracks, num_samples=self.num_track_ids)

        return imgs, tracks
