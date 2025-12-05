import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
import os
from glob import glob
from natsort import natsorted
import h5py

from .utils import load_rgb


class SimpleBaseDataset(Dataset):
    """
    Simplified base dataset for loading video and track data without task embeddings,
    visibility flags, or multi-view support.
    
    Expected h5 structure:
        root/
            video: (T, C, H, W) - RGB video frames
            tracks: (T, num_track_ts, num_points, 2) - future tracks at each frame
    """
    
    def __init__(self,
                 dataset_dir,
                 img_size,
                 num_track_ts,
                 num_track_ids,
                 frame_stack=1,
                 cache_all=False,
                 cache_image=False,
                 num_demos=None,
                 aug_prob=0.):
        super().__init__()
        self.dataset_dir = dataset_dir
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

        self.load_image_func = load_rgb

        if isinstance(self.dataset_dir, str):
            self.dataset_dir = [self.dataset_dir]

        # Load all h5 files
        self.buffer_fns = []
        for dir_idx, d in enumerate(self.dataset_dir):
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
        
        self.load_demo_info()
        
        # Augmentor should be set up by subclass with config params
        self.augmentor = None

    def load_demo_info(self):
        """Load metadata for all demos and create index mappings, filtering by track availability."""
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)
            
            # Get demo length from video shape
            demo_len = demo["root"]["video"].shape[0]
            
            # Determine valid frame indices based on track availability
            valid_frame_indices = []
            for t in range(demo_len):
                track_key = f"frame_{t:04d}"
                if track_key in demo["root"]["tracks"]:
                    frame_data = demo["root"]["tracks"][track_key]
                    num_points = frame_data['query_coords'].shape[0]
                    if num_points >= self.num_track_ids:
                        valid_frame_indices.append(t)
            
            num_valid = len(valid_frame_indices)
            
            if self.cache_all:
                demo = self.process_demo(demo)
                if not self.cache_image:
                    del demo["root"]["video"]
                demo["_valid_frame_indices"] = valid_frame_indices
                self._cache.append(demo)
            
            self._demo_id_to_path[demo_idx] = fn
            self._demo_id_to_valid_indices[demo_idx] = valid_frame_indices
            for local_idx in valid_frame_indices:
                self._index_to_demo_id[start_idx] = demo_idx
                self._index_to_local_frame[start_idx] = local_idx
                start_idx += 1
            
            self._demo_id_to_demo_length[demo_idx] = demo_len
            
            print(f"Demo {demo_idx} ({os.path.basename(fn)}): {num_valid}/{demo_len} valid frames")

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx
        print(f"Total samples: {num_samples}")

    def process_demo(self, demo):
        """Process demo data (resize images if needed)."""
        video = demo["root"]['video']  # T, C, H, W
        # Note: tracks are now stored as separate per-frame datasets in demo["root"]["tracks"]
        # We don't need to process them here, they're loaded on-demand in __getitem__
        
        t, c, h, w = video.shape
        
        # Convert to tensors
        video = torch.Tensor(video)
        
        # Resize images to desired size if needed
        if h != self.img_size[0] or w != self.img_size[1]:
            video = F.interpolate(video, size=self.img_size, mode="bilinear", align_corners=False)
        
        demo["root"]['video'] = video
        # Keep tracks as-is (dict of per-frame datasets)
        
        return demo

    def _load_image_list_from_demo(self, demo, time_offset, num_frames=None, backward=False):
        """Load images from cached demo."""
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"]["video"].shape[0]
        
        if backward:
            # Load frames from time_offset - num_frames + 1 to time_offset (inclusive)
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root']["video"][image_indices]
            
            # Pad with first frame if needed (when at start of trajectory)
            if len(frames) < num_frames:
                first_frame = frames[0:1]  # Keep dims: (1, C, H, W)
                num_padding = num_frames - len(frames)
                padding_frames = first_frame.repeat(num_padding, 1, 1, 1)
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root']["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, time_offset, num_frames=None, backward=False):
        """Load images from disk (when cache_all=False and cache_image=False)."""
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{img_idx}.png")) for img_idx in image_indices]
            # Pad with first frame if needed (when at start of trajectory)
            num_padding = num_frames - len(frames)
            if num_padding > 0:
                frames = [frames[0]] * num_padding + frames
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # T, H, W, C
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def load_h5(self, fn):
        """Load h5 file and return as nested dict."""
        def h5_to_dict(h5):
            d = {}
            for k, v in h5.items():
                if isinstance(v, h5py._hl.group.Group):
                    d[k] = h5_to_dict(v)
                else:
                    d[k] = np.array(v)
            return d

        with h5py.File(fn, 'r') as f:
            return h5_to_dict(f)

    def __len__(self):
        return len(self._index_to_demo_id)

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__")
