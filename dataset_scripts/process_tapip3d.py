"""
TAPIP3D Processing Script
Runs TAPIP3D on ViPE outputs to extract 3D point tracks with sliding windows.
Similar to process_cotracker.py but for 3D tracking.

Usage:
    python process_tapip3d.py
"""

import os
import sys
import warnings
# Suppress warnings early (before torch import)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch
from omegaconf import OmegaConf
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import json
import struct
import zlib

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TAPIP3D_ROOT = PROJECT_ROOT / "thirdparty" / "TAPIP3D"

# Add script directory to path FIRST (for local utils.adaptive_workers, etc.)
sys.path.insert(0, str(SCRIPT_DIR))

# Save original directory and change to TAPIP3D root for TAPIP3D imports
ORIGINAL_CWD = Path.cwd()
os.chdir(TAPIP3D_ROOT)

# Temporarily add TAPIP3D to path for its imports
sys.path.insert(0, str(TAPIP3D_ROOT))

# Default TAPIP3D settings
DEFAULT_CHECKPOINT = TAPIP3D_ROOT / "checkpoints" / "tapip3d_final.pth"
DEFAULT_NUM_ITERS = 6  # TAPIP3D default
VIZ_HTML_TEMPLATE = TAPIP3D_ROOT / "utils" / "viz.html"

# Import TAPIP3D utilities using absolute path to avoid conflict with dataset_scripts/utils
import importlib.util
spec_inf = importlib.util.spec_from_file_location("tapip3d_inference_utils", TAPIP3D_ROOT / "utils" / "inference_utils.py")
spec_common = importlib.util.spec_from_file_location("tapip3d_common_utils", TAPIP3D_ROOT / "utils" / "common_utils.py")
inference_utils = importlib.util.module_from_spec(spec_inf)
common_utils = importlib.util.module_from_spec(spec_common)
spec_inf.loader.exec_module(inference_utils)
spec_common.loader.exec_module(common_utils)

# Extract functions from TAPIP3D modules for direct use
inference = inference_utils.inference

import logging

# Change back to original directory
os.chdir(ORIGINAL_CWD)

# Remove TAPIP3D from the front of sys.path
sys.path.remove(str(TAPIP3D_ROOT))
# Add it back at the end (after SCRIPT_DIR) so dataset_scripts/utils takes precedence
sys.path.append(str(TAPIP3D_ROOT))

logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    # Resolve relative path from script directory
    if not Path(config_path).is_absolute():
        config_path = SCRIPT_DIR / config_path
    return OmegaConf.load(config_path)


def load_vipe_data(npz_path: Path, frames_dir: Path):
    """
    Load ViPE output data from .npz file.
    RGB frames loaded from frames_dir instead of npz.
    
    Args:
        npz_path: Path to ViPE .npz file
        frames_dir: Path to frames directory for this video
    
    Returns:
        video: (T, H, W, 3) uint8 RGB frames
        depths: (T, H, W) float32 depth maps
        intrinsics: (T, 3, 3) float32 camera matrices
        extrinsics: (T, 4, 4) float32 world2cam matrices
    """
    data = np.load(npz_path)
    
    # Load RGB from frames directory
    video = load_frames_from_directory(frames_dir)
    
    # Convert depth from float16 to float32
    depths = data['depths'].astype(np.float32)
    intrinsics = data['intrinsics']
    extrinsics = data['extrinsics']
    
    return video, depths, intrinsics, extrinsics


def load_frames_from_directory(frames_dir: Path) -> np.ndarray:
    """Load PNG frames into (T, H, W, 3) uint8 array."""
    frame_files = sorted(frames_dir.glob("*.png"))
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0).astype(np.uint8)


def generate_sliding_windows(num_frames, window_length, stride):
    """
    Generate sliding window indices.
    
    Args:
        num_frames: Total number of frames
        window_length: Length of each window
        stride: Step size between windows
    
    Returns:
        List of (start_idx, end_idx) tuples
    """
    windows = []
    for start in range(0, num_frames - window_length + 1, stride):
        end = start + window_length
        windows.append((start, end))
    return windows


def get_query_points_from_mask(
    mask: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    sampling_method: str = "grid",
    grid_size: int = 32,
    target_points: int = 500,
    min_mask_percent: float = 1.0,
    erosion_pixels: int = 2,
) -> torch.Tensor:
    """
    Generate 3D query points from mask using grid or random sampling.
    
    Args:
        mask: (H, W) binary mask
        depths: (T, H, W) depth maps
        intrinsics: (T, 3, 3) camera intrinsics
        extrinsics: (T, 4, 4) camera extrinsics (world2cam)
        sampling_method: "grid" or "random"
        grid_size: For grid method: creates grid_size × grid_size points
        target_points: For random method: number of points to sample
        min_mask_percent: Minimum mask area as percentage of image (0-100). Skip if below.
        erosion_pixels: Pixels to erode mask
    
    Returns:
        query_points: (N, 4) tensor with (t, x, y, z) in world coords
    """
    H, W = mask.shape
    device = mask.device
    total_pixels = H * W
    
    # Erode mask to avoid edge points
    if erosion_pixels > 0:
        kernel = np.ones((erosion_pixels * 2 + 1, erosion_pixels * 2 + 1), np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)
        mask = torch.from_numpy(mask_np).to(device)
    
    # Get valid pixels (mask > 0 AND depth > 0)
    depth_0 = depths[0]
    valid_mask = (mask > 0.5) & (depth_0 > 0)
    
    # Check mask percentage threshold
    mask_pixels = valid_mask.sum().item()
    mask_percent = (mask_pixels / total_pixels) * 100
    
    if mask_percent < min_mask_percent:
        return torch.zeros((0, 4), device=device)
    
    if sampling_method == "random":
        # Random sampling: pick target_points uniformly from valid pixels
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (M, 2)
        num_valid = valid_indices.shape[0]
        
        if num_valid == 0:
            return torch.zeros((0, 4), device=device)
        
        # Sample target_points (or all if fewer available)
        num_samples = min(target_points, num_valid)
        perm = torch.randperm(num_valid, device=device)[:num_samples]
        sampled = valid_indices[perm]
        
        # Sort in row-major order (by y first, then x)
        sort_keys = sampled[:, 0] * W + sampled[:, 1]  # y * width + x
        sort_order = torch.argsort(sort_keys)
        sampled = sampled[sort_order]
        
        y_coords = sampled[:, 0]
        x_coords = sampled[:, 1]
    
    else:  # grid sampling
        # Create grid_size × grid_size grid over the entire image
        step_y = H / grid_size
        step_x = W / grid_size
        
        # Grid points at center of each cell
        y_grid = (torch.arange(grid_size, device=device) * step_y + step_y / 2).long().clamp(0, H - 1)
        x_grid = (torch.arange(grid_size, device=device) * step_x + step_x / 2).long().clamp(0, W - 1)
        
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        y_coords = yy.reshape(-1)
        x_coords = xx.reshape(-1)
        
        # Filter by valid mask
        valid = valid_mask[y_coords, x_coords]
        y_coords = y_coords[valid]
        x_coords = x_coords[valid]
    
    if len(x_coords) == 0:
        return torch.zeros((0, 4), device=device)
    
    # Get depths at these points
    d = depth_0[y_coords, x_coords]
    
    # Unproject to 3D world coordinates
    xy = torch.stack([x_coords.float(), y_coords.float()], dim=-1)  # (N, 2)
    
    inv_intrinsics0 = torch.linalg.inv(intrinsics[0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0])
    
    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)  # (N, 3)
    xy_homo = torch.einsum('ij,nj->ni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)  # (N, 4)
    world_coords = torch.einsum('ij,nj->ni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]
    
    # Query format: (t, x, y, z) where t=0 for frame 0
    queries = torch.cat([torch.zeros_like(xy[:, :1]), world_coords], dim=-1)  # (N, 4)
    
    return queries


def relativize_poses(extrinsics: np.ndarray, window_start: int, window_end: int) -> np.ndarray:
    """
    Make camera poses relative to frame 0 of the window.
    
    Args:
        extrinsics: (T, 4, 4) full video extrinsics
        window_start: Start frame index
        window_end: End frame index
    
    Returns:
        relative_poses: (window_length, 4, 4) poses relative to window's frame 0
    """
    window_extrinsics = extrinsics[window_start:window_end]  # (T_window, 4, 4)
    
    # Make t=0 the identity (world origin)
    inv_first = np.linalg.inv(window_extrinsics[0])
    relative_poses = np.array([inv_first @ ext for ext in window_extrinsics])
    
    return relative_poses.astype(np.float32)


def compute_global_frame0_poses(extrinsics: np.ndarray, windows: list) -> np.ndarray:
    """
    Compute each window's frame 0 pose relative to the first window's frame 0.
    Used for visualization linking.
    
    Args:
        extrinsics: (T, 4, 4) full video extrinsics
        windows: List of (start, end) tuples
    
    Returns:
        global_frame0_poses: (num_windows, 4, 4) poses for linking
    """
    first_window_frame0 = extrinsics[windows[0][0]]
    inv_first_window = np.linalg.inv(first_window_frame0)
    
    global_frame0_poses = []
    for start_idx, end_idx in windows:
        this_frame0 = extrinsics[start_idx]
        relative_to_first = inv_first_window @ this_frame0
        global_frame0_poses.append(relative_to_first)
    
    return np.stack(global_frame0_poses, axis=0).astype(np.float32)


def process_window(
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_points: torch.Tensor,
    num_iters: int = 6,
    visibility_threshold: float = 0.9,
    device: str = "cuda",
) -> tuple:
    """
    Process a single window with TAPIP3D.
    
    Args:
        model: TAPIP3D model
        video: (T, C, H, W) video tensor
        depths: (T, H, W) depth tensor
        intrinsics: (T, 3, 3) intrinsics tensor
        extrinsics: (T, 4, 4) extrinsics tensor (should be relativized to window's t=0)
        query_points: (N, 4) query points (t, x, y, z)
        num_iters: Number of refinement iterations
        visibility_threshold: Minimum visibility to keep tracks
        device: Device to run on
    
    Returns:
        tracks: (1, T, N_filtered, 3) filtered 3D tracks
        visibility: (T, N_filtered) visibility mask
    """
    T, C, H, W = video.shape
    N = query_points.shape[0]
    
    if N == 0:
        return torch.zeros((1, T, 0, 3), device=device), torch.zeros((T, 0), device=device, dtype=torch.bool)
    
    # Run TAPIP3D inference
    coords, visibs = inference(
        model=model,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        query_point=query_points,
        num_iters=num_iters,
        grid_size=0,  # We provide our own query points
        bidrectional=False,  # Forward only since we query at t=0
        vis_threshold=visibility_threshold,
    )
    
    # coords: (T, N, 3), visibs: (T, N)
    
    # Filter points that become invisible at any frame
    visible_all_frames = visibs.all(dim=0)  # (N,)
    
    coords_filtered = coords[:, visible_all_frames, :]  # (T, N_filtered, 3)
    visibs_filtered = visibs[:, visible_all_frames]  # (T, N_filtered)
    
    # Add batch dimension
    tracks = coords_filtered.unsqueeze(0)  # (1, T, N_filtered, 3)
    
    return tracks, visibs_filtered


def process_video_with_tapip3d(
    model: torch.nn.Module,
    npz_path: Path,
    mask_path: Path,
    output_path: Path,
    config: dict,
) -> Path:
    """
    Process a single video with TAPIP3D sliding windows.
    
    Args:
        model: TAPIP3D model
        npz_path: Path to ViPE .npz file
        mask_path: Path to mask .pt file (optional)
        output_path: Path to save output .pt file
        config: Configuration dictionary
    
    Returns:
        Path to saved .pt file
    """
    device = config['device']
    window_length = config['window_length']
    stride = config['stride']
    grid_size = config['grid_size']
    use_masks = config['use_masks']
    mask_erosion_pixels = config['mask_erosion_pixels']
    visibility_threshold = config['visibility_threshold']
    resolution_factor = config['resolution_factor']
    sampling_method = config.get('sampling_method', 'grid')
    target_points = config.get('target_points', 500)
    min_mask_percent = config.get('min_mask_percent', 1.0)
    num_iters = DEFAULT_NUM_ITERS
    
    # Load ViPE data
    video_name = npz_path.stem
    frames_dir = Path(config['input_frames_dir']) / video_name
    video_np, depths_np, intrinsics_np, extrinsics_np = load_vipe_data(npz_path, frames_dir)
    
    num_frames = video_np.shape[0]
    H, W = video_np.shape[1:3]
    
    # Generate sliding windows
    windows = generate_sliding_windows(num_frames, window_length, stride)
    if len(windows) == 0:
        tqdm.write(f"  {npz_path.stem}: too short ({num_frames} frames) for window length {window_length}")
        return None
    
    # Set model image size based on resolution factor
    base_H, base_W = model.image_size
    inference_H = int(base_H * np.sqrt(resolution_factor))
    inference_W = int(base_W * np.sqrt(resolution_factor))
    model.set_image_size((inference_H, inference_W))
    
    # Load mask if using masks
    mask = None
    if use_masks and mask_path.exists():
        mask_data = torch.load(mask_path, map_location='cpu', weights_only=False)
        masks = mask_data['masks']

        # Convert to torch tensor if it's numpy
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        # Combine all objects: (T, O, H, W) -> (T, H, W)
        if masks.dim() == 4:
            mask = torch.any(masks > 0.5, dim=1).float()  # (T, H, W)
        else:
            mask = masks.float()
    
    # Compute global frame 0 poses for visualization linking
    global_frame0_poses = compute_global_frame0_poses(extrinsics_np, windows)
    
    # Process each window
    all_tracks = []
    all_window_poses = []
    
    for window_idx, (start_idx, end_idx) in enumerate(tqdm(windows, desc="Windows", leave=False)):
        # Extract window data
        window_video = video_np[start_idx:end_idx]  # (T_win, H, W, 3)
        window_depths = depths_np[start_idx:end_idx]  # (T_win, H, W)
        window_intrinsics = intrinsics_np[start_idx:end_idx]  # (T_win, 3, 3)
        
        # Relativize poses to window's frame 0
        window_poses = relativize_poses(extrinsics_np, start_idx, end_idx)  # (T_win, 4, 4)
        all_window_poses.append(window_poses)
        
        # Convert to tensors
        video_tensor = torch.from_numpy(window_video).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        depths_tensor = torch.from_numpy(window_depths).float()  # (T, H, W)
        intrinsics_tensor = torch.from_numpy(window_intrinsics).float()  # (T, 3, 3)
        extrinsics_tensor = torch.from_numpy(window_poses).float()  # (T, 4, 4)
        
        # Move to device
        video_tensor = video_tensor.to(device)
        depths_tensor = depths_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)
        
        # Generate query points
        if use_masks and mask is not None:
            window_mask = mask[start_idx].to(device)
            query_points = get_query_points_from_mask(
                mask=window_mask,
                depths=depths_tensor,
                intrinsics=intrinsics_tensor,
                extrinsics=extrinsics_tensor,
                sampling_method=sampling_method,
                grid_size=grid_size,
                target_points=target_points,
                min_mask_percent=min_mask_percent,
                erosion_pixels=mask_erosion_pixels,
            )
        else:
            # Use grid queries from TAPIP3D
            query_points = inference_utils.get_grid_queries(
                grid_size=grid_size,
                depths=depths_tensor,
                intrinsics=intrinsics_tensor,
                extrinsics=extrinsics_tensor,
            )
        
        # Run TAPIP3D on window
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tracks, visibility = process_window(
                model=model,
                video=video_tensor,
                depths=depths_tensor,
                intrinsics=intrinsics_tensor,
                extrinsics=extrinsics_tensor,
                query_points=query_points,
                num_iters=num_iters,
                visibility_threshold=visibility_threshold,
                device=device,
            )
        
        all_tracks.append(tracks.cpu())
        
        # Clear GPU cache periodically
        if (window_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'tracks': all_tracks,  # List[(1, T_window, N_i, 3)] window-local 3D
        'windows': windows,  # List[(start, end)]
        'window_poses': all_window_poses,  # List[(T_window, 4, 4)] relative poses
        'global_frame0_pose': global_frame0_poses,  # (num_windows, 4, 4) for vis linking
        'intrinsics': intrinsics_np[0],  # (3, 3) camera intrinsics
        'video_name': npz_path.stem,
        'window_length': window_length,
        'stride': stride,
        'grid_size': grid_size,
        'num_frames': num_frames,
        'resolution': (H, W),
    }, output_path)
    
    return output_path


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def compress_and_write(filename, header, blob):
    """Write compressed binary data with header."""
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)


def create_visualization_data(tracks_file, vipe_file, output_file, width=256, height=192, fps=12, config=None):
    """
    Create visualization binary data for a single video.
    
    For each window:
    - RGB/depth at t=0 (starting scene)
    - trajectories_start: positions at t=0
    - trajectories_end: positions at t=T_window-1
    """
    fixed_size = (width, height)
    
    # Load tracks data
    tracks_data = torch.load(tracks_file, map_location='cpu', weights_only=False)
    all_tracks = tracks_data['tracks']  # List[(1, T_window, N, 3)]
    windows = tracks_data['windows']
    global_frame0_poses = tracks_data['global_frame0_pose']
    intrinsics_single = tracks_data['intrinsics']
    
    num_windows = len(windows)
    window_length = tracks_data['window_length']
    
    # Load ViPE data
    vipe_data = np.load(vipe_file)
    depths_full = vipe_data['depths'].astype(np.float32)  # Convert float16 to float32
    
    # Load RGB from frames directory
    video_name = vipe_file.stem
    frames_dir = Path(config['input_frames_dir']) / video_name
    video_full = load_frames_from_directory(frames_dir)
    
    T_full, H_orig, W_orig = depths_full.shape[:3]
    
    # Extract frame 0 of each window
    rgb_frames = []
    depth_frames = []
    
    for start_idx, end_idx in windows:
        rgb_resized = cv2.resize(video_full[start_idx], fixed_size, interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depths_full[start_idx], fixed_size, interpolation=cv2.INTER_NEAREST)
        rgb_frames.append(rgb_resized)
        depth_frames.append(depth_resized)
    
    rgb_video = np.stack(rgb_frames)
    depth_video = np.stack(depth_frames).astype(np.float32)
    
    # Scale intrinsics
    scale_x = fixed_size[0] / W_orig
    scale_y = fixed_size[1] / H_orig
    intrinsics = np.tile(intrinsics_single[np.newaxis, :, :], (num_windows, 1, 1))
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    # Compute FOV
    fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(fixed_size[1] / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(fixed_size[0] / (2 * fx)) * (180 / np.pi)
    
    # Process depth
    valid_depths = depth_video[depth_video > 0]
    min_depth = float(valid_depths.min()) * 0.8 if len(valid_depths) > 0 else 0.1
    max_depth = float(valid_depths.max()) * 1.5 if len(valid_depths) > 0 else 10.0
    
    depth_normalized = np.clip((depth_video - min_depth) / (max_depth - min_depth), 0, 1)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((num_windows, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    normalized_extrinsics = global_frame0_poses.astype(np.float32)
    
    # Process trajectories - extract FULL trajectory for each window
    # Shape: (num_windows, T_window, N, 3)
    max_points = max(t.shape[2] for t in all_tracks) if all_tracks else 0
    
    if max_points == 0:
        trajectories = np.zeros((num_windows, window_length, 0, 3), dtype=np.float32)
    else:
        traj_list = []
        
        for tracks, global_pose in zip(all_tracks, global_frame0_poses):
            tracks_np = tracks.numpy().squeeze(0)  # (T_window, N, 3)
            T_win, N = tracks_np.shape[:2]
            
            # Transform ALL timesteps to global coordinates
            inv_global_pose = np.linalg.inv(global_pose)
            
            # Transform each timestep
            global_tracks = np.zeros((T_win, N, 3), dtype=np.float32)
            for t in range(T_win):
                positions = tracks_np[t]  # (N, 3)
                positions_homo = np.concatenate([positions, np.ones((N, 1))], axis=1)
                global_positions = (inv_global_pose @ positions_homo.T).T[:, :3]
                global_tracks[t] = global_positions
            
            # Pad points if needed (use large values to place off-screen, not at origin)
            if N < max_points:
                # Place padding points far behind camera so they're not visible
                padding = np.full((T_win, max_points - N, 3), 1e6, dtype=np.float32)
                global_tracks = np.concatenate([global_tracks, padding], axis=1)
            
            traj_list.append(global_tracks[:, :max_points, :])
        
        trajectories = np.stack(traj_list, axis=0).astype(np.float32)
    
    # Build output - include full trajectories
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics.astype(np.float32),
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics).astype(np.float32),
        "trajectories": trajectories,  # (num_windows, T_window, N, 3)
        "cameraZ": np.float64(0.0),
    }
    
    header = {}
    blob_parts = []
    offset = 0
    
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset, "length": len(arr_bytes)}
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    compressed_blob = zlib.compress(b"".join(blob_parts), level=9)
    
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": num_windows,
        "resolution": list(fixed_size),
        "baseFrameRate": fps,
        "numTrajectoryPoints": max_points,
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "windowLength": window_length,
        "stride": tracks_data.get('stride', 1),
        "windowedMode": True,
    }
    
    compress_and_write(output_file, header, compressed_blob)
    return num_windows, max_points


def create_windowed_viz_html(vis_path):
    """Create the windowed visualization HTML that shows full trajectory polylines."""
    
    # Read the original viz.html as template
    with open(VIZ_HTML_TEMPLATE, 'r') as f:
        html = f.read()
    
    # Detect windowed mode by checking for 4D trajectory shape (num_windows, T_window, N, 3)
    html = html.replace(
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        const [totalFrames, numTrajectories] = shape;''',
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        // Windowed mode: shape is (num_windows, T_window, N, 3)
        // Original mode: shape is (totalFrames, N, 3)
        const isWindowedMode = shape.length === 4;
        let totalFrames, numTrajectories, windowLength;
        if (isWindowedMode) {
          [totalFrames, windowLength, numTrajectories] = shape;
        } else {
          [totalFrames, numTrajectories] = shape;
          windowLength = 1;
        }
        
        // Store for updateTrajectories
        this.isWindowedMode = isWindowedMode;
        this.windowLength = windowLength;'''
    )
    
    # Add end marker creation for windowed mode
    html = html.replace(
        '''trajectoryGroup.userData = {
            marker: positionMarker,
            line: trajectoryLine,
            color: colors[i]
          };''',
        '''// Create end marker for windowed mode
          let endMarker = null;
          if (this.isWindowedMode) {
            const endSphereGeometry = new THREE.SphereGeometry(ballSize * 1.5, 16, 16);
            const endSphereMaterial = new THREE.MeshBasicMaterial({ 
              color: colors[i],
              transparent: true,
              opacity: 1.0
            });
            endMarker = new THREE.Mesh(endSphereGeometry, endSphereMaterial);
            trajectoryGroup.add(endMarker);
          }
          
          trajectoryGroup.userData = {
            marker: positionMarker,
            line: trajectoryLine,
            endMarker: endMarker,
            color: colors[i]
          };'''
    )
    
    # Replace updateTrajectories function to handle full polylines
    old_update = '''updateTrajectories(frameIndex) {
        if (!this.data.trajectories || this.trajectories.length === 0) return;
        
        const trajectoryData = this.data.trajectories.data;
        const [totalFrames, numTrajectories] = this.data.trajectories.shape;
        const historyFramesSetting = parseInt(this.ui.trajectoryHistory.value);
        const historyFrames = Math.min(historyFramesSetting, this.config.totalFrames);
        
        for (let i = 0; i < numTrajectories; i++) {
          const trajectoryGroup = this.trajectories[i];
          const { marker, line } = trajectoryGroup.userData;
          
          const currentPos = new THREE.Vector3();
          const currentOffset = (frameIndex * numTrajectories + i) * 3;
          
          currentPos.x = trajectoryData[currentOffset];
          currentPos.y = -trajectoryData[currentOffset + 1];
          currentPos.z = -trajectoryData[currentOffset + 2];
          
          marker.position.copy(currentPos);
          
          const positions = [];
          const historyToShow = Math.min(historyFrames, frameIndex + 1);
          
          for (let j = 0; j < historyToShow; j++) {
            const historyFrame = Math.max(0, frameIndex - j);
            const historyOffset = (historyFrame * numTrajectories + i) * 3;
            
            positions.push(
              trajectoryData[historyOffset],
              -trajectoryData[historyOffset + 1],
              -trajectoryData[historyOffset + 2]
            );
          }
          
          for (let j = historyToShow; j < historyFrames; j++) {
            positions.push(currentPos.x, currentPos.y, currentPos.z);
          }
          
          line.geometry.setPositions(positions);
          
          line.visible = frameIndex > 0;
        }
      }'''
    
    new_update = '''updateTrajectories(frameIndex) {
        if (!this.data.trajectories || this.trajectories.length === 0) return;
        
        const trajectoryData = this.data.trajectories.data;
        const shape = this.data.trajectories.shape;
        
        if (this.isWindowedMode) {
          // Windowed mode: shape is (num_windows, T_window, N, 3)
          // Draw full polyline through all T_window timesteps
          const [numWindows, windowLength, numTrajectories] = shape;
          
          for (let i = 0; i < numTrajectories; i++) {
            const trajectoryGroup = this.trajectories[i];
            const { marker, line, endMarker } = trajectoryGroup.userData;
            
            // Collect all positions for this trajectory in this window
            const positions = [];
            for (let t = 0; t < windowLength; t++) {
              // Index: [frameIndex][t][i][xyz] -> flat: ((frameIndex * windowLength + t) * numTrajectories + i) * 3
              const offset = ((frameIndex * windowLength + t) * numTrajectories + i) * 3;
              positions.push(
                trajectoryData[offset],
                -trajectoryData[offset + 1],
                -trajectoryData[offset + 2]
              );
            }
            
            // Start position (t=0)
            const startOffset = (frameIndex * windowLength * numTrajectories + i) * 3;
            marker.position.set(
              trajectoryData[startOffset],
              -trajectoryData[startOffset + 1],
              -trajectoryData[startOffset + 2]
            );
            
            // End position (t=windowLength-1)
            if (endMarker) {
              const endOffset = ((frameIndex * windowLength + windowLength - 1) * numTrajectories + i) * 3;
              endMarker.position.set(
                trajectoryData[endOffset],
                -trajectoryData[endOffset + 1],
                -trajectoryData[endOffset + 2]
              );
            }
            
            // Draw polyline through all timesteps
            line.geometry.setPositions(positions);
            line.visible = true;
          }
        } else {
          // Original mode: history-based trajectories
          const [totalFrames, numTrajectories] = shape;
          const historyFramesSetting = parseInt(this.ui.trajectoryHistory.value);
          const historyFrames = Math.min(historyFramesSetting, this.config.totalFrames);
          
          for (let i = 0; i < numTrajectories; i++) {
            const trajectoryGroup = this.trajectories[i];
            const { marker, line } = trajectoryGroup.userData;
            
            const currentPos = new THREE.Vector3();
            const currentOffset = (frameIndex * numTrajectories + i) * 3;
            
            currentPos.x = trajectoryData[currentOffset];
            currentPos.y = -trajectoryData[currentOffset + 1];
            currentPos.z = -trajectoryData[currentOffset + 2];
            
            marker.position.copy(currentPos);
            
            const positions = [];
            const historyToShow = Math.min(historyFrames, frameIndex + 1);
            
            for (let j = 0; j < historyToShow; j++) {
              const historyFrame = Math.max(0, frameIndex - j);
              const historyOffset = (historyFrame * numTrajectories + i) * 3;
              
              positions.push(
                trajectoryData[historyOffset],
                -trajectoryData[historyOffset + 1],
                -trajectoryData[historyOffset + 2]
              );
            }
            
            for (let j = historyToShow; j < historyFrames; j++) {
              positions.push(currentPos.x, currentPos.y, currentPos.z);
            }
            
            line.geometry.setPositions(positions);
            line.visible = frameIndex > 0;
          }
        }
      }'''
    
    html = html.replace(old_update, new_update)
    
    # Fix FPS bug
    html = html.replace(
        'this.playbackSpeed = this.config.baseFrameRate;',
        'this.playbackSpeed = 1;  // Speed multiplier (1x normal speed)'
    )
    html = html.replace(
        'const speedRates = speeds.map(s => s * this.config.baseFrameRate);',
        'const speedRates = speeds;  // Speed multipliers directly'
    )
    html = html.replace(
        'const normalizedSpeed = this.playbackSpeed / this.config.baseFrameRate;',
        'const normalizedSpeed = this.playbackSpeed;  // Already a multiplier'
    )
    html = html.replace(
        'this.playbackSpeed = speedRates[nextIndex];',
        'this.playbackSpeed = speeds[nextIndex];'
    )
    
    # Save
    output_path = vis_path / "viz_windowed.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def create_index_html(vis_path, video_list):
    """Create index page with all videos."""
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>TAPIP3D Visualization</title>
<style>
body { margin: 0; padding: 20px; font-family: system-ui; background: #111; color: #eee; }
h1 { color: #a78bfa; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.card { background: #222; border-radius: 8px; padding: 16px; text-decoration: none; color: #eee; }
.card:hover { background: #333; }
.card h3 { margin: 0; color: #a78bfa; }
</style></head><body>
<h1>TAPIP3D Visualization</h1>
<div class="grid">
"""
    for i, v in enumerate(video_list):
        html += f'<a href="{v}.html" class="card"><h3>{v}</h3><p>{i+1}/{len(video_list)}</p></a>\n'
    html += "</div></body></html>"
    
    with open(vis_path / "index.html", 'w') as f:
        f.write(html)


def create_video_html(video_name, video_list, vis_path):
    """Create HTML for a video with prev/next navigation."""
    idx = video_list.index(video_name)
    prev_v = video_list[idx - 1] if idx > 0 else None
    next_v = video_list[idx + 1] if idx < len(video_list) - 1 else None
    
    with open(vis_path / "viz_windowed.html", 'r') as f:
        html = f.read()
    
    html = html.replace('data.bin', f'{video_name}_data.bin')
    
    nav = f"""<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(0,0,0,0.8);padding:8px 12px;border-radius:8px;display:flex;gap:10px;font-family:system-ui;color:#eee;">
<a href="index.html" style="color:#a78bfa;">Index</a>
<span>|</span>
{"<a href='" + prev_v + ".html' style='color:#a78bfa;'>← Prev</a>" if prev_v else "<span style='color:#666;'>← Prev</span>"}
<span style="font-weight:600;">{video_name}</span>
<span style="color:#888;">({idx+1}/{len(video_list)})</span>
{"<a href='" + next_v + ".html' style='color:#a78bfa;'>Next →</a>" if next_v else "<span style='color:#666;'>Next →</span>"}
</div>"""
    
    html = html.replace('<body>', '<body>' + nav)
    
    with open(vis_path / f"{video_name}.html", 'w') as f:
        f.write(html)


def generate_visualizations(
    tracks_dir: Path,
    vipe_dir: Path,
    vis_path: Path,
    vis_fps: int,
    video_list: list,
    config: dict,
):
    """Generate all visualization files for processed videos."""
    print(f"\nGenerating 3D visualizations to {vis_path}")
    
    # Create windowed viz HTML template
    create_windowed_viz_html(vis_path)
    
    # Process each video
    for video_name in tqdm(video_list, desc="Creating viz", leave=False):
        tracks_file = tracks_dir / f"{video_name}.pt"
        vipe_file = vipe_dir / f"{video_name}.npz"
        
        if not tracks_file.exists() or not vipe_file.exists():
            continue
        
        create_visualization_data(
            tracks_file,
            vipe_file,
            vis_path / f"{video_name}_data.bin",
            fps=vis_fps,
            config=config,
        )
        create_video_html(video_name, video_list, vis_path)
    
    # Create index
    create_index_html(vis_path, video_list)


# ============================================================================
# ADAPTIVE WORKER SUPPORT
# ============================================================================

class TAPIP3DWorkerFunction:
    """Worker function for adaptive pool"""

    def __init__(self, config, tapip3d_config):
        self.config = config
        self.tapip3d_config = tapip3d_config
        self.checkpoint_path = DEFAULT_CHECKPOINT

    def load_model(self):
        """Load TAPIP3D model (called once per worker)"""
        model = inference_utils.load_model(str(self.checkpoint_path))
        model.to("cuda:0")  # Always cuda:0 in worker (CUDA_VISIBLE_DEVICES set by pool)
        model.eval()
        return model

    def process(self, model, video_info):
        """Process single video"""
        npz_path, mask_path, output_path = video_info

        try:
            # Process video with TAPIP3D
            result = process_video_with_tapip3d(
                model=model,
                npz_path=npz_path,
                mask_path=mask_path if mask_path.exists() else None,
                output_path=output_path,
                config=self.tapip3d_config,  # Pass tapip3d_config, not general config
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to process {npz_path.stem}: {e}")


def main():
    # Set up logging - suppress verbose output
    common_utils.setup_logger()
    logging.getLogger().setLevel(logging.WARNING)
    
    # Load configuration
    config = load_config("config.yaml")
    tapip3d_config = config['tapip3d']
    
    # Extract configuration
    input_vipe_dir = Path(tapip3d_config['input_vipe_dir'])
    input_masks_dir = Path(tapip3d_config['input_masks_dir'])
    output_tracks_dir = Path(tapip3d_config['output_tracks_dir'])
    checkpoint_path = DEFAULT_CHECKPOINT
    device = tapip3d_config['device']
    num_videos_to_process = tapip3d_config['num_videos_to_process']
    continue_mode = tapip3d_config.get('continue', False)
    vis_flag = tapip3d_config['vis_flag']
    vis_path = Path(tapip3d_config['vis_path']) if vis_flag else None
    vis_fps = tapip3d_config.get('vis_fps', config['common']['target_fps'])
    
    # Delete previous output if not continuing
    if not continue_mode:
        if output_tracks_dir.exists():
            print(f"\nDeleting previous tracks directory: {output_tracks_dir}")
            shutil.rmtree(output_tracks_dir)
        if vis_flag and vis_path and vis_path.exists():
            print(f"Deleting previous visualizations directory: {vis_path}")
            shutil.rmtree(vis_path)
    
    # Create output directories
    output_tracks_dir.mkdir(parents=True, exist_ok=True)
    if vis_flag and vis_path:
        vis_path.mkdir(parents=True, exist_ok=True)
    
    # Find all ViPE .npz files
    npz_files = sorted(input_vipe_dir.glob("*.npz"))
    
    if num_videos_to_process is not None:
        npz_files = npz_files[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"TAPIP3D PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Window length: {tapip3d_config['window_length']} frames")
    print(f"Stride: {tapip3d_config['stride']} frames")
    print(f"Use masks: {tapip3d_config['use_masks']}")
    sampling_method = tapip3d_config.get('sampling_method', 'grid')
    print(f"Sampling method: {sampling_method}")
    if sampling_method == 'grid':
        print(f"Grid size: {tapip3d_config['grid_size']}")
    else:
        print(f"Target points: {tapip3d_config.get('target_points', 500)}")
    print(f"Min mask percent: {tapip3d_config.get('min_mask_percent', 1.0)}%")
    print(f"Visibility threshold: {tapip3d_config['visibility_threshold']}")
    print(f"Resolution factor: {tapip3d_config['resolution_factor']}")
    print(f"Num iterations: {DEFAULT_NUM_ITERS} (TAPIP3D default)")
    print(f"Input ViPE directory: {input_vipe_dir}")
    print(f"Output tracks directory: {output_tracks_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Videos to process: {len(npz_files)}")
    print(f"Visualization: {'Enabled' if vis_flag else 'Disabled'}")
    if vis_flag:
        print(f"Visualization path: {vis_path}")
    print(f"{'='*60}\n")
    
    # Check if adaptive workers enabled
    use_adaptive = config.get('optimization', {}).get('use_adaptive_workers', False)

    # Track statistics
    total_videos_processed = 0
    total_windows_processed = 0
    skipped_videos = 0
    processed_video_names = []
    start_time = time.time()

    if use_adaptive:
        # ====================================================================
        # ADAPTIVE WORKER MODE
        # ====================================================================
        # Now that utils is renamed to pipeline_utils, no conflicts with TAPIP3D!
        from pipeline_utils.adaptive_workers import AdaptiveWorkerPool
        from pipeline_utils.gpu_utils import detect_gpus

        num_gpus, gpu_info = detect_gpus()
        max_workers_per_gpu = tapip3d_config['max_workers']

        print(f"{'='*60}")
        print("ADAPTIVE WORKER MODE")
        print(f"{'='*60}")
        print(f"🚀 Detected {num_gpus} GPU(s), max {max_workers_per_gpu} workers/GPU")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        print(f"{'='*60}\n")

        # Prepare video list
        video_infos = []
        for npz_path in npz_files:
            video_name = npz_path.stem
            mask_path = input_masks_dir / f"{video_name}.pt"
            output_path = output_tracks_dir / f"{video_name}.pt"

            # Skip if already processed
            if continue_mode and output_path.exists():
                skipped_videos += 1
                processed_video_names.append(video_name)
                continue

            video_infos.append((npz_path, mask_path, output_path))

        if len(video_infos) == 0:
            print("✅ All videos already processed!")
        else:
            # Create worker function
            worker_fn = TAPIP3DWorkerFunction(config, tapip3d_config)

            # Create adaptive pool
            checkpoint_dir = config.get('optimization', {}).get('checkpoint_dir', None)
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir) / "tapip3d"

            pool = AdaptiveWorkerPool(
                num_gpus=num_gpus,
                max_workers_per_gpu=max_workers_per_gpu,
                worker_fn=worker_fn,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=config.get('optimization', {}).get('save_checkpoint_every', 100),
                spawn_delay=config['optimization']['spawn_delay'],
                verbose_workers=config['optimization'].get('verbose_workers', False),
                save_worker_logs=config['optimization'].get('save_worker_logs', True),
                log_dir=Path(tapip3d_config['worker_log_dir']),
            )

            # Process videos
            results, stable_workers = pool.process_items(video_infos, desc="TAPIP3D")

            print(f"\n✅ Completed with {stable_workers} stable workers")

            # Summary
            successful = sum(1 for r in results.values() if r is not None)
            failed = len(results) - successful
            total_videos_processed = successful
            processed_video_names.extend([v[0].stem for v, r in results.items() if r is not None])

            if failed > 0:
                print(f"⚠️ {failed} videos failed (see error log)")

    else:
        # ====================================================================
        # SEQUENTIAL MODE (LEGACY)
        # ====================================================================
        print(f"{'='*60}")
        print("SEQUENTIAL MODE (LEGACY)")
        print(f"{'='*60}\n")

        # Load TAPIP3D model
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = inference_utils.load_model(str(checkpoint_path))
        model.to(device)
        print("TAPIP3D model loaded\n")

        # Process each video
        pbar = tqdm(npz_files, desc="Processing videos")
        for npz_path in pbar:
            video_name = npz_path.stem
            pbar.set_postfix_str(video_name)
            output_path = output_tracks_dir / f"{video_name}.pt"

            # Check if already processed
            if continue_mode and output_path.exists():
                skipped_videos += 1
                processed_video_names.append(video_name)
                continue

            # Find corresponding mask file
            mask_path = input_masks_dir / f"{video_name}.pt"

            result_path = process_video_with_tapip3d(
                model=model,
                npz_path=npz_path,
                mask_path=mask_path,
                output_path=output_path,
                config=tapip3d_config,
            )

            if result_path is not None:
                # Load result to count windows
                result = torch.load(result_path, map_location='cpu', weights_only=False)
                num_windows = len(result['windows'])
                total_windows_processed += num_windows
                total_videos_processed += 1
                processed_video_names.append(video_name)

            # Clear GPU cache between videos
            torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    if total_videos_processed > 0:
        print(f"✅ PROCESSING COMPLETE!")
    else:
        print(f"❌ PROCESSING FAILED!")
    print(f"{'='*60}")
    print(f"Total videos: {len(npz_files)}")
    print(f"Videos processed: {total_videos_processed}")
    if skipped_videos > 0:
        print(f"Videos skipped: {skipped_videos}")
    print(f"Total windows: {total_windows_processed}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
        print(f"Tracks saved to: {output_tracks_dir}")
    
    # Generate visualizations if enabled
    if vis_flag and vis_path and processed_video_names:
        generate_visualizations(
            tracks_dir=output_tracks_dir,
            vipe_dir=input_vipe_dir,
            vis_path=vis_path,
            vis_fps=vis_fps,
            video_list=processed_video_names,
            config=tapip3d_config,
        )
        
        print(f"\n{'='*60}")
        print("TO VIEW 3D VISUALIZATION:")
        print(f"{'='*60}")
        print(f"  python visualize_tapip3d.py")
        print(f"  OR:")
        print(f"  cd {vis_path}")
        print(f"  python -m http.server 8000")
        print(f"  Then open: http://localhost:8000/index.html")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
