import numpy as np
import torch

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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            imgs = imgs[0] * 255.  # Back to (frame_stack, c, h, w) and 0-255 range
            tracks = tracks[0, 0]  # Back to (num_track_ts, n, 2)

        # Uniformly sample tracks to get num_track_ids tracks
        tracks = sample_tracks_uniform(tracks, num_samples=self.num_track_ids)

        return imgs, tracks
