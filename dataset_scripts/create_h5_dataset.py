#!/usr/bin/env python3
"""
Create HDF5 dataset from processed frames and tracks.
Supports both 2D (CoTracker) and 3D (TAPIP3D) tracks.

For 2D tracks:
- Converts clean frames to center-cropped & resized images
- Transforms 2D track coordinates to normalized [0,1] space

For 3D tracks:
- Same image processing
- Projects 3D tracks to camera-0 frame as (u, v, d)
- Computes displacements (du, dv, dd) relative to t=0
- Converts poses to 9D representation
- Applies normalization to all data types
- Optionally stores depth maps
"""
import os
import sys
import yaml
import torch
from omegaconf import OmegaConf
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import h5py


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def clip_depth(depth, min_depth=None, max_depth=None):
    """
    Clip depth values to valid range and handle invalid values.
    
    Args:
        depth: numpy array of depth values
        min_depth: Minimum valid depth (None = no lower clip)
        max_depth: Maximum valid depth (None = no upper clip)
    
    Returns:
        Clipped depth array (0 values are preserved as invalid markers)
    """
    # Replace invalid values (inf, nan) with 0
    depth = np.where(np.isfinite(depth), depth, 0.0)
    
    # Clip to valid range (keep 0 as invalid marker)
    valid_mask = depth > 0
    if min_depth is not None:
        depth = np.where(valid_mask & (depth < min_depth), min_depth, depth)
    if max_depth is not None:
        depth = np.where(valid_mask & (depth > max_depth), max_depth, depth)
    
    return depth


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    return OmegaConf.load(config_path)


# ==============================================================================
# POSE CONVERSION FUNCTIONS
# ==============================================================================

def rotation_matrix_to_6d(R):
    # Column-major order: first column, then second column
    # This matches the standard 6D representation and the decoding in test_dataloader.py
    return np.concatenate([R[:, 0], R[:, 1]])  # (6,) = [r00, r10, r20, r01, r11, r21]


def pose_matrix_to_9d(pose_4x4):
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]
    return np.concatenate([rotation_matrix_to_6d(R), t])  # (9,)


def poses_to_9d(poses):
    T = poses.shape[0]
    poses_9d = np.zeros((T, 9), dtype=np.float32)
    for t in range(T):
        poses_9d[t] = pose_matrix_to_9d(poses[t])
    return poses_9d


def compute_relative_rotation_6d(R_t, R_0):
    """
    Compute relative rotation from R_0 to R_t and convert to 6D representation.
    
    Args:
        R_t: (3, 3) rotation matrix at time t
        R_0: (3, 3) rotation matrix at time 0
    
    Returns:
        (6,) 6D representation of relative rotation R_rel = R_t @ R_0.T
    """
    R_rel = R_t @ R_0.T
    return rotation_matrix_to_6d(R_rel)


def project_wrist_to_uvd(wrist_3d, intrinsics, img_size):
    """
    Project 3D wrist position to normalized (u, v, d) coordinates.
    
    Args:
        wrist_3d: (3,) wrist position in camera frame [x, y, z]
        intrinsics: (3, 3) camera intrinsics matrix
        img_size: (height, width) of the image
    
    Returns:
        (3,) normalized (u, v, d) where u, v in [0, 1] and d in meters
    """
    # Project to pixel coords: uv_h = K @ [x, y, z]^T
    uv = intrinsics @ wrist_3d  # (3,)
    
    # Homogeneous division
    u_px = uv[0] / uv[2]  # pixel x
    v_px = uv[1] / uv[2]  # pixel y
    d = wrist_3d[2]  # z-coordinate is depth
    
    # Normalize u, v to [0, 1]
    h, w = img_size
    u = u_px / w
    v = v_px / h
    
    return np.array([u, v, d], dtype=np.float32)


# ==============================================================================
# PROJECTION FUNCTIONS
# ==============================================================================

def project_to_camera_frame(points_local, intrinsics, img_size):
    """
    Project 3D points (already in camera-0 frame) to (u, v, depth).
    
    Args:
        points_local: (N, 3) points in camera-0's coordinate system
        intrinsics: (3, 3) camera intrinsics matrix
        img_size: (height, width) of the image
    
    Returns:
        u: (N,) normalized u coordinates [0, 1]
        v: (N,) normalized v coordinates [0, 1]
        d: (N,) depth values in meters
    """
    # Project to pixel coords: uv_h = K @ [x, y, z]^T
    uv = (intrinsics @ points_local.T).T  # (N, 3)
    
    # Homogeneous division
    u_px = uv[:, 0] / uv[:, 2]  # pixel x
    v_px = uv[:, 1] / uv[:, 2]  # pixel y
    d = points_local[:, 2]  # z-coordinate is depth
    
    # Normalize u, v to [0, 1]
    h, w = img_size
    u = u_px / w
    v = v_px / h
    
    return u, v, d


# ==============================================================================
# HAND POSE PROCESSING FUNCTIONS
# ==============================================================================

def process_hand_poses_for_window(wrist_poses_4x4, intrinsics, img_size, num_track_ts):
    """
    Process hand poses for a window into query coords and displacements.
    
    Args:
        wrist_poses_4x4: (T, 4, 4) wrist SE3 transforms for the window
        intrinsics: (3, 3) transformed camera intrinsics
        img_size: (height, width) of the processed image
        num_track_ts: Number of timesteps to extract
    
    Returns:
        query_uvd: (3,) initial wrist (u, v, d) at t=0
        uvd_displacements: (T, 3) position displacements relative to t=0
        query_rot_6d: (6,) initial wrist rotation (6D) at t=0
        rot_displacements: (T, 6) rotation displacements (relative rotation -> 6D)
    """
    T = min(wrist_poses_4x4.shape[0], num_track_ts)
    
    # Extract positions and rotations
    wrist_positions = wrist_poses_4x4[:T, :3, 3]  # (T, 3)
    wrist_rotations = wrist_poses_4x4[:T, :3, :3]  # (T, 3, 3)
    
    # Project positions to (u, v, d)
    uvd_all = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        uvd_all[t] = project_wrist_to_uvd(wrist_positions[t], intrinsics, img_size)
    
    # Query coords at t=0
    query_uvd = uvd_all[0]  # (3,)
    
    # Position displacements relative to t=0
    uvd_displacements = uvd_all - uvd_all[0:1]  # (T, 3)
    
    # Extract rotation at t=0
    R_0 = wrist_rotations[0]  # (3, 3)
    query_rot_6d = rotation_matrix_to_6d(R_0)  # (6,)
    
    # Rotation displacements: relative rotation R_t @ R_0.T -> 6D
    rot_displacements = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        rot_displacements[t] = compute_relative_rotation_6d(wrist_rotations[t], R_0)
    
    return query_uvd, uvd_displacements, query_rot_6d, rot_displacements


def get_valid_hand_poses(hand_data, window_frames):
    """
    Get valid hand poses for a window. Returns poses for each hand if valid for ALL frames.
    
    Args:
        hand_data: Dict with 'left_valid', 'right_valid', 'left_wrist_pose', 'right_wrist_pose'
        window_frames: Array of frame indices for this window
    
    Returns:
        Dict with 'left' and 'right' keys. Each value is (T, 4, 4) poses or None if invalid.
    """
    left_valid = hand_data['left_valid']
    right_valid = hand_data['right_valid']
    
    # Check validity for all frames in window using numpy
    left_valid_window = left_valid[window_frames].all()
    right_valid_window = right_valid[window_frames].all()
    
    result = {'left': None, 'right': None}
    
    if left_valid_window:
        result['left'] = hand_data['left_wrist_pose'][window_frames]
    
    if right_valid_window:
        result['right'] = hand_data['right_wrist_pose'][window_frames]
    
    return result


# ==============================================================================
# IMAGE PROCESSING FUNCTIONS
# ==============================================================================

def center_crop_and_resize(frame, target_size, crop_params=None, interpolation=cv2.INTER_LANCZOS4):
    """
    Center crop frame to square, then resize to target_size x target_size.
    
    Args:
        frame: Input frame (H, W) or (H, W, C)
        target_size: Target dimension (int)
        crop_params: Optional (crop_x, crop_y, crop_size) tuple. If None, computes from frame.
        interpolation: cv2 interpolation method (INTER_LANCZOS4 for RGB, INTER_NEAREST for depth)
    
    Returns:
        If crop_params is None:
            processed_frame, crop_offset_x, crop_offset_y, crop_size
        If crop_params is provided:
            processed_frame only
    """
    if crop_params is None:
        # Compute crop parameters
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        crop_offset_x = (w - crop_size) // 2
        crop_offset_y = (h - crop_size) // 2
        return_params = True
    else:
        crop_offset_x, crop_offset_y, crop_size = crop_params
        return_params = False
    
    # Perform center crop
    cropped = frame[crop_offset_y:crop_offset_y + crop_size, 
                   crop_offset_x:crop_offset_x + crop_size]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)
    
    if return_params:
        return resized, crop_offset_x, crop_offset_y, crop_size
    return resized


def transform_intrinsics(intrinsics, crop_offset_x, crop_offset_y, crop_size, target_size):
    """
    Transform camera intrinsics for crop + resize.
    
    Args:
        intrinsics: (3, 3) camera intrinsics matrix
        crop_offset_x, crop_offset_y: Crop offsets
        crop_size: Size of square crop
        target_size: Target size after resize
    
    Returns:
        transformed intrinsics (3, 3)
    """
    K = intrinsics.copy()
    # Apply crop offset to principal point
    K[0, 2] -= crop_offset_x  # cx
    K[1, 2] -= crop_offset_y  # cy
    # Apply resize scale
    scale = target_size / crop_size
    K[0, :] *= scale  # fx, s, cx
    K[1, :] *= scale  # fy, cy
    return K




# ==============================================================================
# 2D TRACK FUNCTIONS
# ==============================================================================

def transform_track_coordinates_2d(tracks, orig_width, orig_height, 
                                crop_offset_x, crop_offset_y, crop_size, target_size):
    """
    Transform 2D track coordinates from original image space to normalized [0, 1] space
    after center crop and resize.
    
    Args:
        tracks: (N, 2) array in absolute pixel coordinates (x, y)
        orig_width, orig_height: Original image dimensions
        crop_offset_x, crop_offset_y: Crop offsets
        crop_size: Size of square crop
        target_size: Target image size after resize
    
    Returns:
        transformed_tracks: (N', 2) array in normalized [0, 1] coordinates, N' <= N
    """
    if tracks.shape[0] == 0:
        return tracks  # Empty array, return as-is
    
    # Apply crop offset
    tracks_cropped = tracks.copy()
    tracks_cropped[:, 0] = tracks[:, 0] - crop_offset_x  # x
    tracks_cropped[:, 1] = tracks[:, 1] - crop_offset_y  # y
    
    # Filter out-of-bounds points (outside crop region)
    valid_mask = (tracks_cropped[:, 0] >= 0) & (tracks_cropped[:, 0] < crop_size) & \
                 (tracks_cropped[:, 1] >= 0) & (tracks_cropped[:, 1] < crop_size)
    tracks_cropped = tracks_cropped[valid_mask]
    
    if tracks_cropped.shape[0] == 0:
        return tracks_cropped  # All points filtered out
    
    # Apply resize scaling
    scale = target_size / crop_size
    tracks_resized = tracks_cropped * scale
    
    # Normalize to [0, 1]
    tracks_normalized = tracks_resized / target_size
    
    return tracks_normalized


def convert_sliding_windows_to_future_tracks_2d(all_tracks, windows, num_track_ts, total_frames):
    """
    Convert 2D sliding window tracks to per-frame future trajectories.
    
    Args:
        all_tracks: List of track tensors [(1, T_window, N_i, 2), ...]
        windows: List of (start, end) tuples indicating window boundaries
        num_track_ts: Number of future timesteps to extract
        total_frames: Total number of frames in video
    
    Returns:
        per_frame_tracks: List of arrays, shape (num_track_ts, N_i, 2)
    """
    # Create mapping from window start frame to tracks (only if long enough)
    window_map = {}
    for window_idx, (start, end) in enumerate(windows):
        window_tracks = all_tracks[window_idx][0]  # Remove batch dim: (T_window, N, 2)
        if window_tracks.shape[0] >= num_track_ts:
            window_map[start] = window_tracks[:num_track_ts]
    
    # Create per-frame future tracks
    per_frame_tracks = []
    
    for frame_idx in range(total_frames):
        if frame_idx in window_map:
            future_tracks = window_map[frame_idx]
            if isinstance(future_tracks, torch.Tensor):
                future_tracks = future_tracks.numpy()
            per_frame_tracks.append(future_tracks)
        else:
            per_frame_tracks.append(None)
    
    return per_frame_tracks


# ==============================================================================
# 3D TRACK FUNCTIONS
# ==============================================================================

def convert_sliding_windows_to_future_tracks_3d(all_tracks, all_window_poses, windows, 
                                                  num_track_ts, total_frames, intrinsics, img_size,
                                                  hand_data=None):
    """
    Convert 3D sliding window tracks to per-frame future (u, v, d) coordinates.
    
    Tracks are already in camera-0's coordinate system for each window.
    Projects to (u, v, d) and computes displacements relative to t=0.
    Filters out points that go outside image bounds or behind camera at any timestep.
    Skips windows that are shorter than num_track_ts.
    
    If hand_data is provided, processes both left and right hand poses for each window
    (storing whichever hands are valid for all frames in the window).
    """
    # Create mapping from window start frame to data (only if long enough)
    window_map = {}
    for window_idx, (start, end) in enumerate(windows):
        window_tracks = all_tracks[window_idx][0]  # (T_window, N, 3)
        window_poses = all_window_poses[window_idx]  # (T_window, 4, 4)
        if window_tracks.shape[0] >= num_track_ts:
            window_map[start] = (window_tracks[:num_track_ts], window_poses[:num_track_ts])
    
    per_frame_query_coords = []
    per_frame_displacements = []
    per_frame_poses_9d = []
    per_frame_hand_data = []  # Hand pose data per frame
    
    for frame_idx in range(total_frames):
        if frame_idx in window_map:
            window_tracks, window_poses = window_map[frame_idx]
            window_frames = np.arange(frame_idx, frame_idx + num_track_ts)
            
            if isinstance(window_tracks, torch.Tensor):
                window_tracks = window_tracks.numpy()
            
            N = window_tracks.shape[1]
            
            # Project ALL timesteps to (u, v, d) using camera-0 intrinsics
            uvd_all = np.zeros((num_track_ts, N, 3), dtype=np.float32)
            
            for t in range(num_track_ts):
                u, v, d = project_to_camera_frame(window_tracks[t], intrinsics, img_size)
                uvd_all[t, :, 0] = u
                uvd_all[t, :, 1] = v
                uvd_all[t, :, 2] = d
            
            # Filter: keep only points that are valid at ALL timesteps
            # Valid means: u in [0, 1], v in [0, 1], d > 0
            valid_per_timestep = (
                (uvd_all[:, :, 0] >= 0) & (uvd_all[:, :, 0] <= 1) &  # u in bounds
                (uvd_all[:, :, 1] >= 0) & (uvd_all[:, :, 1] <= 1) &  # v in bounds
                (uvd_all[:, :, 2] > 0)                                # d positive
            )  # (T, N)
            valid_all_timesteps = valid_per_timestep.all(axis=0)  # (N,)
            
            # Apply filter
            uvd_all = uvd_all[:, valid_all_timesteps, :]  # (T, N_valid, 3)
            
            # Query coords at t=0: (N_valid, 3) - (u_0, v_0, d_0)
            query_coords = uvd_all[0]
            
            # Displacements: (du, dv, dd) relative to t=0
            displacements = uvd_all - uvd_all[0:1]  # (T, N_valid, 3)
            
            # Convert poses to 9D
            poses_9d = poses_to_9d(window_poses)  # (T, 9)
            
            # Process hand poses if available - always try both hands
            hand_pose_data = {}
            if hand_data is not None:
                valid_hands = get_valid_hand_poses(hand_data, window_frames)
                
                # Process left hand if valid
                if valid_hands['left'] is not None:
                    left_query_uvd, left_uvd_disp, left_query_rot, left_rot_disp = \
                        process_hand_poses_for_window(valid_hands['left'], intrinsics, img_size, num_track_ts)
                    hand_pose_data['left_wrist_query_uvd'] = left_query_uvd
                    hand_pose_data['left_wrist_uvd_displacements'] = left_uvd_disp
                    hand_pose_data['left_wrist_query_rot_6d'] = left_query_rot
                    hand_pose_data['left_wrist_rot_displacements'] = left_rot_disp
                
                # Process right hand if valid
                if valid_hands['right'] is not None:
                    right_query_uvd, right_uvd_disp, right_query_rot, right_rot_disp = \
                        process_hand_poses_for_window(valid_hands['right'], intrinsics, img_size, num_track_ts)
                    hand_pose_data['right_wrist_query_uvd'] = right_query_uvd
                    hand_pose_data['right_wrist_uvd_displacements'] = right_uvd_disp
                    hand_pose_data['right_wrist_query_rot_6d'] = right_query_rot
                    hand_pose_data['right_wrist_rot_displacements'] = right_rot_disp
            
            per_frame_query_coords.append(query_coords)
            per_frame_displacements.append(displacements)
            per_frame_poses_9d.append(poses_9d)
            per_frame_hand_data.append(hand_pose_data if hand_pose_data else None)
        else:
            # Frame not a window start - no tracks
            per_frame_query_coords.append(None)
            per_frame_displacements.append(None)
            per_frame_poses_9d.append(None)
            per_frame_hand_data.append(None)
    
    return per_frame_query_coords, per_frame_displacements, per_frame_poses_9d, per_frame_hand_data


# ==============================================================================
# VIDEO PROCESSING FUNCTIONS
# ==============================================================================

def process_video(video_folder, tracks_file, output_h5_path, config, track_type, vipe_file=None, hand_poses_file=None, text_description=""):
    """
    Process a single video with 2D or 3D tracks.
    
    For 2D: transforms track coords to normalized [0,1] space
    For 3D: projects to (u,v,d), computes displacements and 9D poses
    
    If hand_poses_file is provided, also processes both left and right hand poses
    (storing whichever hands are valid for each window).
    
    Args:
        text_description: Optional text description of the video clip for language conditioning.
    """
    target_size = config['target_size']
    num_track_ts = config['num_track_ts']
    depth_min = config.get('depth_min', None)
    depth_max = config.get('depth_max', None)
    
    # Load all frames
    frame_files = sorted(video_folder.glob("*.png"))
    if len(frame_files) == 0:
        print(f"  No frames found in {video_folder}")
        return None
    
    total_frames = len(frame_files)
    print(f"  Loading {total_frames} frames...")
    
    # Load first frame to get original dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = first_frame.shape[:2]
    
    # Process all frames
    processed_frames = []
    crop_params = None
    
    for frame_file in tqdm(frame_files, desc="  Processing frames", leave=False):
        frame = cv2.imread(str(frame_file))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed, crop_x, crop_y, crop_size = center_crop_and_resize(frame, target_size)
        processed_frames.append(processed)
        
        if crop_params is None:
            crop_params = (crop_x, crop_y, crop_size)
    
    crop_x, crop_y, crop_size = crop_params
    
    # Stack frames: (T, H, W, C) -> (T, C, H, W)
    video_array = np.stack(processed_frames, axis=0)
    video_array = np.transpose(video_array, (0, 3, 1, 2))
    print(f"  Processed frames to shape: {video_array.shape}")
    
    # Load tracks
    if not tracks_file.exists():
        print(f"  Tracks file not found: {tracks_file}")
        return None
    
    print(f"  Loading tracks from {tracks_file.name}...")
    track_data = torch.load(tracks_file, map_location='cpu', weights_only=False)
    all_tracks = track_data['tracks']
    windows = track_data['windows']
    print(f"  Loaded {len(all_tracks)} windows")
    
    # Initialize return values
    all_depth_values = []
    all_poses_9d = []
    depth_array = None
    intrinsics_transformed = None
    
    if track_type == '2d':
        # 2D: Convert and transform track coordinates
        per_frame_tracks = convert_sliding_windows_to_future_tracks_2d(
            all_tracks, windows, num_track_ts, total_frames
        )
        
        per_frame_query_coords = []
        per_frame_displacements = []
        track_counts = []
        
        for tracks_t in per_frame_tracks:
            if tracks_t is None:
                per_frame_query_coords.append(None)
                per_frame_displacements.append(None)
                track_counts.append(0)
            else:
                transformed_timesteps = []
                for ts in range(num_track_ts):
                    transformed = transform_track_coordinates_2d(
                        tracks_t[ts], orig_width, orig_height,
                        crop_x, crop_y, crop_size, target_size
                    )
                    transformed_timesteps.append(transformed)
                
                min_points = min(t.shape[0] for t in transformed_timesteps)
                
                if min_points == 0:
                    per_frame_query_coords.append(None)
                    per_frame_displacements.append(None)
                    track_counts.append(0)
                else:
                    truncated = [t[:min_points] for t in transformed_timesteps]
                    tracks_stacked = np.stack(truncated, axis=0).astype(np.float32)  # (T, N, 2)
                    
                    # Extract query_coords (position at t=0) and displacements
                    query_coords = tracks_stacked[0]  # (N, 2)
                    displacements = tracks_stacked - tracks_stacked[0:1]  # (T, N, 2)
                    
                    per_frame_query_coords.append(query_coords)
                    per_frame_displacements.append(displacements)
                    track_counts.append(min_points)
    
        # Collect displacements for statistics
        all_displacements = [d for d in per_frame_displacements if d is not None]
        
    else:  # 3D
        all_window_poses = track_data['window_poses']
        intrinsics_orig = track_data['intrinsics']
        
        # Transform intrinsics
        intrinsics_transformed = transform_intrinsics(
            intrinsics_orig, crop_x, crop_y, crop_size, target_size
        )
        print(f"  Transformed intrinsics for {target_size}x{target_size} images")
        
        # Load depth (always required for 3D)
        if vipe_file and vipe_file.exists():
            print(f"  Loading depth from {vipe_file.name}...")
            vipe_data = np.load(vipe_file)
            depths_full = vipe_data['depths'].astype(np.float32)  # Convert float16 to float32
            
            processed_depths = []
            for t in range(min(total_frames, depths_full.shape[0])):
                depth_t = center_crop_and_resize(
                    depths_full[t], target_size, 
                    crop_params=(crop_x, crop_y, crop_size),
                    interpolation=cv2.INTER_NEAREST
                )
                # Apply depth clipping
                depth_t = clip_depth(depth_t, depth_min, depth_max)
                processed_depths.append(depth_t)
                valid_d = depth_t[depth_t > 0]
                if len(valid_d) > 0:
                    all_depth_values.append(valid_d)
            
            depth_array = np.stack(processed_depths, axis=0).astype(np.float32)
            print(f"  Processed depth to shape: {depth_array.shape}")
        else:
            raise ValueError(f"3D mode requires ViPE depth file but not found: {vipe_file}")
        
        # Load hand pose data if available
        hand_data = None
        if hand_poses_file and hand_poses_file.exists():
            print(f"  Loading hand poses from {hand_poses_file.name}...")
            hand_data = torch.load(hand_poses_file, map_location='cpu', weights_only=False)
        
        # Convert to (u, v, d), displacements, 9D poses, and hand data
        per_frame_query_coords, per_frame_displacements, per_frame_poses_9d, per_frame_hand_data = \
            convert_sliding_windows_to_future_tracks_3d(
                all_tracks, all_window_poses, windows, num_track_ts, 
                total_frames, intrinsics_transformed, img_size=(target_size, target_size),
                hand_data=hand_data
            )
        
        # Apply depth clipping to track depths and collect statistics
        for query_coords in per_frame_query_coords:
            if query_coords is not None:
                # Clip depth values in query_coords (3rd dimension)
                query_coords[:, 2] = clip_depth(query_coords[:, 2], depth_min, depth_max)
                valid_d = query_coords[:, 2][query_coords[:, 2] > 0]
                if len(valid_d) > 0:
                    all_depth_values.append(valid_d)
        
        for poses_9d in per_frame_poses_9d:
            if poses_9d is not None:
                all_poses_9d.append(poses_9d)
        
        track_counts = [q.shape[0] if q is not None else 0 for q in per_frame_query_coords]
        all_displacements = [d for d in per_frame_displacements if d is not None]
    
    # Statistics
    frames_with_tracks = sum(1 for c in track_counts if c > 0)
    avg_tracks = np.mean([c for c in track_counts if c > 0]) if frames_with_tracks > 0 else 0
    
    print(f"  Track statistics:")
    print(f"    Frames with tracks: {frames_with_tracks}/{total_frames}")
    print(f"    Avg tracks per frame: {avg_tracks:.1f}")
    
    # Collect hand pose data for statistics (3D only)
    all_hand_uvd_displacements = []
    all_hand_rot_displacements = []
    frames_with_left_hand = 0
    frames_with_right_hand = 0
    
    if track_type == '3d' and hand_data is not None:
        for hand_pose_data in per_frame_hand_data:
            if hand_pose_data is not None:
                # Collect left hand if present
                if 'left_wrist_uvd_displacements' in hand_pose_data:
                    frames_with_left_hand += 1
                    all_hand_uvd_displacements.append(hand_pose_data['left_wrist_uvd_displacements'])
                    all_hand_rot_displacements.append(hand_pose_data['left_wrist_rot_displacements'])
                
                # Collect right hand if present
                if 'right_wrist_uvd_displacements' in hand_pose_data:
                    frames_with_right_hand += 1
                    all_hand_uvd_displacements.append(hand_pose_data['right_wrist_uvd_displacements'])
                    all_hand_rot_displacements.append(hand_pose_data['right_wrist_rot_displacements'])
        
        print(f"  Hand statistics:")
        print(f"    Frames with valid left hand: {frames_with_left_hand}/{total_frames}")
        print(f"    Frames with valid right hand: {frames_with_right_hand}/{total_frames}")
    
    # Save to HDF5
    print(f"  Saving to HDF5...")
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('root/video', data=video_array, dtype='uint8')
        
        if track_type == '2d':
            tracks_group = f.create_group('root/tracks')
            for t in range(total_frames):
                if per_frame_query_coords[t] is not None:
                    frame_group = tracks_group.create_group(f'frame_{t:04d}')
                    frame_group.create_dataset('query_coords', data=per_frame_query_coords[t].astype(np.float32))
                    frame_group.create_dataset('displacements', data=per_frame_displacements[t].astype(np.float32))
        else:  # 3D
            if depth_array is not None:
                f.create_dataset('root/depth', data=depth_array, dtype='float32')
            f.create_dataset('root/intrinsics', data=intrinsics_transformed.astype(np.float32), dtype='float32')
            
            tracks_group = f.create_group('root/tracks')
            for t in range(total_frames):
                if per_frame_query_coords[t] is not None:
                    frame_group = tracks_group.create_group(f'frame_{t:04d}')
                    frame_group.create_dataset('query_coords', data=per_frame_query_coords[t].astype(np.float32))
                    frame_group.create_dataset('displacements', data=per_frame_displacements[t].astype(np.float32))
                    frame_group.create_dataset('poses', data=per_frame_poses_9d[t].astype(np.float32))
                    
                    # Save hand pose data if available
                    if per_frame_hand_data[t] is not None:
                        hand_data_t = per_frame_hand_data[t]
                        for key, value in hand_data_t.items():
                            frame_group.create_dataset(key, data=value.astype(np.float32))
        
        f.create_dataset('root/num_frames', data=total_frames)
        f.create_dataset('root/num_track_ts', data=num_track_ts)
        f.create_dataset('root/track_type', data=track_type)
        
        # Store text description (variable-length string)
        if text_description:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('root/text', data=text_description, dtype=dt)
    
    print(f"  Saved to {output_h5_path}")
    
    result = {
        'total_frames': total_frames,
        'frames_with_tracks': frames_with_tracks,
        'avg_tracks': avg_tracks,
        'displacements': all_displacements,
        'track_dim': 2 if track_type == '2d' else 3
    }
    if track_type == '3d':
        result['depth_values'] = all_depth_values
        result['poses_9d'] = all_poses_9d
        if hand_data is not None:
            result['hand_uvd_displacements'] = all_hand_uvd_displacements
            result['hand_rot_displacements'] = all_hand_rot_displacements
            result['frames_with_left_hand'] = frames_with_left_hand
            result['frames_with_right_hand'] = frames_with_right_hand
    
    return result


# ==============================================================================
# STATISTICS FUNCTIONS
# ==============================================================================

def compute_displacement_statistics(all_displacements, track_dim):
    """Compute mean and std of displacements."""
    flattened = []
    for disp in all_displacements:
        if disp is None:
            continue
        if isinstance(disp, list):
            for d in disp:
                if d is not None and d.shape[1] > 0:
                    flattened.append(d.reshape(-1, track_dim))
        elif disp.shape[1] > 0:
            flattened.append(disp.reshape(-1, track_dim))
    
    if len(flattened) == 0:
        print("  Warning: No displacement data for computing statistics!")
        return {
            'displacement_mean': [0.0] * track_dim,
            'displacement_std': [1.0] * track_dim,
            'num_samples': 0
        }
    
    all_disp = np.concatenate(flattened, axis=0)
    
    return {
        'displacement_mean': all_disp.mean(axis=0).tolist(),
        'displacement_std': all_disp.std(axis=0).tolist(),
        'num_samples': int(len(all_disp))
    }


def compute_depth_statistics(all_depth_values):
    """
    Compute mean and std of depth values.
    
    Args:
        all_depth_values: List of depth value arrays
    
    Returns:
        dict with 'depth_mean' and 'depth_std'
    """
    if len(all_depth_values) == 0:
        print("  Warning: No depth data for computing statistics!")
        return {
            'depth_mean': 0.0,
            'depth_std': 1.0,
            'num_samples': 0
        }
    
    all_depths = np.concatenate([d.flatten() for d in all_depth_values])
    
    return {
        'depth_mean': float(all_depths.mean()),
        'depth_std': float(all_depths.std()),
        'num_samples': int(len(all_depths))
    }


def compute_pose_statistics(all_poses_9d):
    """
    Compute mean and std of 9D poses.
    
    Args:
        all_poses_9d: List of (T, 9) pose arrays
    
    Returns:
        dict with 'pose_mean' and 'pose_std'
    """
    if len(all_poses_9d) == 0:
        print("  Warning: No pose data for computing statistics!")
        return {
            'pose_mean': [0.0] * 9,
            'pose_std': [1.0] * 9,
            'num_samples': 0
        }
    
    all_poses = np.concatenate([p.reshape(-1, 9) for p in all_poses_9d], axis=0)
    
    return {
        'pose_mean': all_poses.mean(axis=0).tolist(),
        'pose_std': all_poses.std(axis=0).tolist(),
        'num_samples': int(len(all_poses))
    }


def compute_hand_pose_statistics(all_uvd_displacements, all_rot_displacements):
    """
    Compute mean and std of hand pose displacements.
    
    Args:
        all_uvd_displacements: List of (T, 3) wrist uvd displacement arrays
        all_rot_displacements: List of (T, 6) wrist rotation displacement arrays
    
    Returns:
        dict with hand pose statistics
    """
    result = {
        'hand_uvd_disp_mean': [0.0] * 3,
        'hand_uvd_disp_std': [1.0] * 3,
        'hand_rot_disp_mean': [0.0] * 6,
        'hand_rot_disp_std': [1.0] * 6,
        'num_samples': 0
    }
    
    if len(all_uvd_displacements) == 0:
        print("  Warning: No hand pose data for computing statistics!")
        return result
    
    # UVD displacement statistics
    all_uvd = np.concatenate([d.reshape(-1, 3) for d in all_uvd_displacements], axis=0)
    result['hand_uvd_disp_mean'] = all_uvd.mean(axis=0).tolist()
    result['hand_uvd_disp_std'] = all_uvd.std(axis=0).tolist()
    
    # Rotation displacement statistics
    if len(all_rot_displacements) > 0:
        all_rot = np.concatenate([d.reshape(-1, 6) for d in all_rot_displacements], axis=0)
        result['hand_rot_disp_mean'] = all_rot.mean(axis=0).tolist()
        result['hand_rot_disp_std'] = all_rot.std(axis=0).tolist()
    
    result['num_samples'] = int(len(all_uvd))
    
    return result


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    # Load configuration
    config = load_config("config.yaml")
    h5_config = config['create_h5_dataset']
    
    # Extract configuration
    target_size = h5_config['target_size']
    num_track_ts = h5_config['num_track_ts']
    track_type = h5_config.get('track_type', '2d')
    input_images_dir = Path(h5_config['input_images_dir'])
    output_h5_dir = Path(h5_config['output_h5_dir'])
    num_videos_to_process = h5_config['num_videos_to_process']
    continue_mode = h5_config.get('continue', False)
    
    # Track type specific paths
    if track_type == '2d':
        input_tracks_dir = Path(h5_config['input_tracks_dir'])
        input_vipe_dir = None
        input_hand_poses_dir = None
        track_dim = 2
    else:  # 3d
        input_tracks_dir = Path(h5_config['input_tracks_3d_dir'])
        input_vipe_dir = Path(h5_config['input_vipe_dir'])
        input_hand_poses_dir = Path(h5_config['input_hand_poses_dir']) if h5_config.get('input_hand_poses_dir') else None
        track_dim = 3
    
    # Load metadata for text descriptions (only if include_text is enabled)
    clip_to_text = {}
    include_text = h5_config.get('include_text', False)
    if include_text:
        metadata_file = h5_config.get('metadata_file')
        if not metadata_file:
            raise ValueError("include_text=True but metadata_file not specified in config")
        
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"include_text=True but metadata file not found: {metadata_path}")
        
        print(f"Loading metadata from: {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        
        if 'text' not in metadata_df.columns:
            raise ValueError(f"include_text=True but 'text' column not found in {metadata_path}")
        
        for _, row in metadata_df.iterrows():
            clip_id = int(row['clip_id'])
            text = str(row['text']) if pd.notna(row['text']) else ""
            clip_to_text[clip_id] = text
        print(f"✓ Loaded {len(clip_to_text)} text descriptions")
    
    # Delete previous output directory if not continuing
    if not continue_mode and output_h5_dir.exists():
        print(f"\nDeleting previous output directory: {output_h5_dir}")
        shutil.rmtree(output_h5_dir)
    
    # Create output directory
    output_h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video folders
    video_folders = sorted([d for d in input_images_dir.iterdir() if d.is_dir()])
    
    # Limit to num_videos_to_process if specified
    if num_videos_to_process is not None:
        video_folders = video_folders[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"H5 DATASET CREATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Track type: {track_type.upper()}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Future track timesteps: {num_track_ts}")
    print(f"Input images directory: {input_images_dir}")
    print(f"Input tracks directory: {input_tracks_dir}")
    if track_type == '3d':
        print(f"Input ViPE directory: {input_vipe_dir}")
        depth_min = h5_config.get('depth_min')
        depth_max = h5_config.get('depth_max')
        print(f"Depth clipping: min={depth_min}, max={depth_max}")
        if input_hand_poses_dir:
            print(f"Input hand poses directory: {input_hand_poses_dir}")
    print(f"Output H5 directory: {output_h5_dir}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Include text: {include_text}" + (f" ({len(clip_to_text)} descriptions loaded)" if include_text else ""))
    print(f"Continue mode: {continue_mode}")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # PASS 1: Process all videos and collect statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PASS 1: PROCESSING VIDEOS AND COLLECTING STATISTICS")
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    start_idx = 0
    
    # For statistics collection
    all_displacements = []
    all_depth_values = []
    all_poses_9d = []
    all_hand_uvd_displacements = []
    all_hand_rot_displacements = []
    all_stats = []
    
    # Efficient resume
    if continue_mode and output_h5_dir.exists():
        existing_h5 = sorted([f for f in output_h5_dir.iterdir() if f.suffix == '.hdf5'])
        
        if existing_h5:
            last_h5 = existing_h5[-1]
            video_name = last_h5.stem
            
            print(f"Last HDF5 file {video_name}.hdf5 exists")
            
            for i, vf in enumerate(video_folders):
                if vf.name == video_name:
                    start_idx = i + 1
                    skipped_videos = i + 1
                    break
            
            if start_idx > 0:
                print(f"Resuming from video {start_idx + 1}/{len(video_folders)}")
                print(f"Skipping {start_idx} already processed videos\n")
    
    # Process each video folder
    for idx in range(start_idx, len(video_folders)):
        video_folder = video_folders[idx]
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_folder.name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Find corresponding tracks file
        tracks_file = input_tracks_dir / f"{video_folder.name}.pt"
        
        # Output path
        output_h5_path = output_h5_dir / f"{video_folder.name}.hdf5"
        
        vipe_file = input_vipe_dir / f"{video_folder.name}.npz" if input_vipe_dir else None
        hand_poses_file = input_hand_poses_dir / f"{video_folder.name}.pt" if input_hand_poses_dir else None
        
        # Get text description for this clip
        clip_id = int(video_folder.name)  # Folder name is clip_id (e.g., "00042" -> 42)
        text_description = clip_to_text.get(clip_id, "")
        
        stats = process_video(video_folder, tracks_file, output_h5_path, h5_config, track_type, vipe_file, hand_poses_file, text_description)
        
        if stats is not None:
            all_stats.append(stats)
            total_videos_processed += 1
            
            # Collect data for statistics
            if 'displacements' in stats:
                all_displacements.extend(stats['displacements'])
            if 'depth_values' in stats:
                all_depth_values.extend(stats['depth_values'])
            if 'poses_9d' in stats:
                all_poses_9d.extend(stats['poses_9d'])
            if 'hand_uvd_displacements' in stats:
                all_hand_uvd_displacements.extend(stats['hand_uvd_displacements'])
            if 'hand_rot_displacements' in stats:
                all_hand_rot_displacements.extend(stats['hand_rot_displacements'])
            
            video_elapsed = time.time() - video_start_time
            print(f"\n  Completed in {video_elapsed:.2f}s")
        else:
            print(f"\n  Skipped due to missing data")
        
    
    # =========================================================================
    # COMPUTE STATISTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"COMPUTING NORMALIZATION STATISTICS")
    print(f"{'='*60}")
    
    # Displacement statistics
    disp_stats = compute_displacement_statistics(all_displacements, track_dim)
    print(f"\nDisplacement statistics:")
    print(f"  Mean: {disp_stats['displacement_mean']}")
    print(f"  Std: {disp_stats['displacement_std']}")
    print(f"  Samples: {disp_stats['num_samples']}")
    
    # Depth statistics (3D only)
    depth_stats = {'depth_mean': 0.0, 'depth_std': 1.0, 'num_samples': 0}
    if track_type == '3d':
        depth_stats = compute_depth_statistics(all_depth_values)
        print(f"\nDepth statistics:")
        print(f"  Mean: {depth_stats['depth_mean']:.4f}")
        print(f"  Std: {depth_stats['depth_std']:.4f}")
        print(f"  Samples: {depth_stats['num_samples']}")
    
    # Pose statistics (3D only)
    pose_stats = {'pose_mean': [0.0] * 9, 'pose_std': [1.0] * 9, 'num_samples': 0}
    if track_type == '3d':
        pose_stats = compute_pose_statistics(all_poses_9d)
        print(f"\nPose statistics:")
        print(f"  Mean: {[f'{m:.4f}' for m in pose_stats['pose_mean']]}")
        print(f"  Std: {[f'{s:.4f}' for s in pose_stats['pose_std']]}")
        print(f"  Samples: {pose_stats['num_samples']}")
    
    # Hand pose statistics (3D only, if hand data available)
    hand_pose_stats = None
    if track_type == '3d' and len(all_hand_uvd_displacements) > 0:
        hand_pose_stats = compute_hand_pose_statistics(all_hand_uvd_displacements, all_hand_rot_displacements)
        print(f"\nHand pose statistics:")
        print(f"  UVD disp mean: {[f'{m:.4f}' for m in hand_pose_stats['hand_uvd_disp_mean']]}")
        print(f"  UVD disp std: {[f'{s:.4f}' for s in hand_pose_stats['hand_uvd_disp_std']]}")
        print(f"  Rot disp mean: {[f'{m:.4f}' for m in hand_pose_stats['hand_rot_disp_mean']]}")
        print(f"  Rot disp std: {[f'{s:.4f}' for s in hand_pose_stats['hand_rot_disp_std']]}")
        print(f"  Samples: {hand_pose_stats['num_samples']}")
    
    # =========================================================================
    # SAVE NORMALIZATION STATISTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"SAVING NORMALIZATION STATISTICS")
    print(f"{'='*60}")
    
    norm_stats = {
        'imagenet_mean': IMAGENET_MEAN.tolist(),
        'imagenet_std': IMAGENET_STD.tolist(),
        'track_type': track_type,
        'track_dim': track_dim,
        'displacement_mean': disp_stats['displacement_mean'],
        'displacement_std': disp_stats['displacement_std'],
        'displacement_num_samples': disp_stats['num_samples'],
    }
    
    if track_type == '3d':
        norm_stats.update({
            'depth_mean': depth_stats['depth_mean'],
            'depth_std': depth_stats['depth_std'],
            'depth_num_samples': depth_stats['num_samples'],
            'pose_mean': pose_stats['pose_mean'],
            'pose_std': pose_stats['pose_std'],
            'pose_num_samples': pose_stats['num_samples'],
        })
        
        # Add hand pose statistics if available
        if hand_pose_stats is not None:
            norm_stats.update({
                'hand_uvd_disp_mean': hand_pose_stats['hand_uvd_disp_mean'],
                'hand_uvd_disp_std': hand_pose_stats['hand_uvd_disp_std'],
                'hand_rot_disp_mean': hand_pose_stats['hand_rot_disp_mean'],
                'hand_rot_disp_std': hand_pose_stats['hand_rot_disp_std'],
                'hand_pose_num_samples': hand_pose_stats['num_samples'],
            })
    
    stats_path = output_h5_dir / 'normalization_stats.yaml'
    with open(stats_path, 'w') as f:
        yaml.dump(norm_stats, f, default_flow_style=False)
    
    print(f"Saved normalization statistics to: {stats_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"ALL VIDEOS PROCESSED!")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_folders)}")
    if continue_mode and skipped_videos > 0:
        print(f"Videos processed: {total_videos_processed}")
        print(f"Videos skipped: {skipped_videos}")
    else:
        print(f"Videos processed: {total_videos_processed}")
    
    if all_stats:
        total_frames = sum(s['total_frames'] for s in all_stats)
        total_frames_with_tracks = sum(s['frames_with_tracks'] for s in all_stats)
        avg_tracks_global = np.mean([s['avg_tracks'] for s in all_stats if s['avg_tracks'] > 0])
        
        print(f"\nAggregate statistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames with tracks: {total_frames_with_tracks} ({100*total_frames_with_tracks/total_frames:.1f}%)")
        print(f"  Avg tracks per frame (non-zero): {avg_tracks_global:.1f}")
    
    print(f"\nTotal time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
    print(f"H5 files saved to: {output_h5_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
