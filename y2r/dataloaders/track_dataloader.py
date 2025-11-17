import numpy as np
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
                 num_track_ids=64, frame_stack=1, cache_all=False, cache_image=False, 
                 num_demos=None, aug_prob=0.):
        """
        Initialize TrackDataset.
        
        Args:
            h5_files: Optional list of specific .hdf5 file paths (for train/val split)
            dataset_dir: Optional directory to scan for .hdf5 files (alternative to h5_files)
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
        from .utils import ImgTrackColorJitter, ImgViewDiffTranslationAug
        
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=True),
        ])

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
        # tracks shape: (num_track_ts, num_points, 2)
        track_key = f"frame_{time_offset:04d}"
        tracks = demo["root"]["tracks"][track_key]  # Load from per-frame dataset
        tracks = torch.Tensor(tracks)
        
        # Data augmentation
        if np.random.rand() < self.aug_prob:
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
