#!/usr/bin/env python3
"""
Create HDF5 dataset from processed frames and tracks.
Converts clean frames to center-cropped & resized images and transforms track coordinates.
"""
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import h5py


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def center_crop_and_resize(frame, target_size):
    """
    Center crop frame to square, then resize to target_size x target_size.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: Target dimension (int)
    
    Returns:
        processed_frame: (target_size, target_size, C)
        crop_offset_x: X offset of crop
        crop_offset_y: Y offset of crop
        crop_size: Size of the square crop
    """
    h, w = frame.shape[:2]
    
    # Determine crop size (minimum dimension)
    crop_size = min(h, w)
    
    # Calculate crop offsets (center crop)
    crop_offset_x = (w - crop_size) // 2
    crop_offset_y = (h - crop_size) // 2
    
    # Perform center crop
    cropped = frame[crop_offset_y:crop_offset_y + crop_size, 
                   crop_offset_x:crop_offset_x + crop_size]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    return resized, crop_offset_x, crop_offset_y, crop_size


def transform_track_coordinates(tracks, orig_width, orig_height, 
                                crop_offset_x, crop_offset_y, crop_size, target_size):
    """
    Transform track coordinates from original image space to normalized [0, 1] space
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


def convert_sliding_windows_to_future_tracks(all_tracks, windows, num_track_ts, total_frames):
    """
    Convert sliding window tracks to per-frame future trajectories.
    
    Args:
        all_tracks: List of track tensors [(1, T_window, N_i, 2), ...]
        windows: List of (start, end) tuples indicating window boundaries
        num_track_ts: Number of future timesteps to extract
        total_frames: Total number of frames in video
    
    Returns:
        List of arrays, one per frame, shape (num_track_ts, N_i, 2), variable N_i
    """
    # Create mapping from window start frame to tracks
    window_map = {}
    for window_idx, (start, end) in enumerate(windows):
        window_tracks = all_tracks[window_idx]  # (1, T_window, N, 2)
        window_map[start] = window_tracks[0]  # Remove batch dimension -> (T_window, N, 2)
    
    # Create per-frame future tracks
    per_frame_tracks = []
    for frame_idx in range(total_frames):
        if frame_idx in window_map:
            # Extract future trajectory
            window_tracks = window_map[frame_idx]  # (T_window, N, 2)
            
            # Take first num_track_ts frames as future trajectory
            if window_tracks.shape[0] >= num_track_ts:
                future_tracks = window_tracks[:num_track_ts]  # (num_track_ts, N, 2)
            else:
                # Pad with last position if window is shorter than num_track_ts
                pad_length = num_track_ts - window_tracks.shape[0]
                last_position = window_tracks[-1:].repeat(pad_length, 1, 1)  # (pad_length, N, 2)
                future_tracks = torch.cat([window_tracks, last_position], dim=0)  # (num_track_ts, N, 2)
            
            per_frame_tracks.append(future_tracks.numpy())
        else:
            # No tracks for this frame - create empty array
            empty_tracks = np.zeros((num_track_ts, 0, 2), dtype=np.float32)
            per_frame_tracks.append(empty_tracks)
    
    return per_frame_tracks


def process_video(video_folder, tracks_file, output_h5_path, config):
    """
    Process a single video: load frames and tracks, transform coordinates, save to HDF5.
    
    Args:
        video_folder: Path to folder containing frame images
        tracks_file: Path to .pt file containing tracks
        output_h5_path: Path to save output HDF5 file
        config: Configuration dictionary
    
    Returns:
        track_stats: Dictionary with statistics (total_frames, frames_with_tracks, etc.)
    """
    target_size = config['target_size']
    num_track_ts = config['num_track_ts']
    
    # Load all frames
    frame_files = sorted(video_folder.glob("*.png"))
    if len(frame_files) == 0:
        print(f"⚠ No frames found in {video_folder}")
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
        
        # Center crop and resize
        processed, crop_x, crop_y, crop_size = center_crop_and_resize(frame, target_size)
        processed_frames.append(processed)
        
        # Store crop params (same for all frames)
        if crop_params is None:
            crop_params = (crop_x, crop_y, crop_size)
    
    # Stack frames: (T, H, W, C) -> (T, C, H, W)
    video_array = np.stack(processed_frames, axis=0)  # (T, H, W, C)
    video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
    
    print(f"  ✓ Processed frames to shape: {video_array.shape}")
    
    # Load tracks
    if not tracks_file.exists():
        print(f"⚠ Tracks file not found: {tracks_file}")
        return None
    
    print(f"  Loading tracks from {tracks_file.name}...")
    track_data = torch.load(tracks_file, map_location='cpu')
    all_tracks = track_data['tracks']  # List of [(1, T_window, N_i, 2), ...]
    windows = track_data['windows']  # List of (start, end) tuples
    
    print(f"  ✓ Loaded {len(all_tracks)} windows")
    
    # Convert to per-frame future tracks
    print(f"  Converting sliding windows to per-frame tracks...")
    per_frame_tracks = convert_sliding_windows_to_future_tracks(
        all_tracks, windows, num_track_ts, total_frames
    )
    
    # Transform track coordinates
    print(f"  Transforming track coordinates...")
    crop_x, crop_y, crop_size = crop_params
    per_frame_tracks_transformed = []
    track_counts = []
    
    for frame_idx, tracks_t in enumerate(per_frame_tracks):
        # tracks_t shape: (num_track_ts, N, 2)
        if tracks_t.shape[1] == 0:
            # No tracks, keep empty
            per_frame_tracks_transformed.append(tracks_t.astype(np.float32))
            track_counts.append(0)
        else:
            # Transform each timestep's tracks
            transformed_timesteps = []
            for ts in range(num_track_ts):
                tracks_ts = tracks_t[ts]  # (N, 2)
                transformed = transform_track_coordinates(
                    tracks_ts, orig_width, orig_height,
                    crop_x, crop_y, crop_size, target_size
                )
                transformed_timesteps.append(transformed)
            
            # Find minimum point count across timesteps (due to filtering)
            min_points = min(t.shape[0] for t in transformed_timesteps)
            
            if min_points == 0:
                # All points filtered out
                empty_tracks = np.zeros((num_track_ts, 0, 2), dtype=np.float32)
                per_frame_tracks_transformed.append(empty_tracks)
                track_counts.append(0)
            else:
                # Truncate to consistent point count across timesteps
                # (some points may be filtered out in future frames if they go out of bounds)
                truncated_timesteps = [t[:min_points] for t in transformed_timesteps]
                stacked = np.stack(truncated_timesteps, axis=0)  # (num_track_ts, min_points, 2)
                per_frame_tracks_transformed.append(stacked.astype(np.float32))
                track_counts.append(min_points)
    
    print(f"  ✓ Transformed tracks")
    
    # Statistics
    frames_with_tracks = sum(1 for c in track_counts if c > 0)
    avg_tracks = np.mean([c for c in track_counts if c > 0]) if frames_with_tracks > 0 else 0
    max_tracks = max(track_counts) if track_counts else 0
    min_tracks_nonzero = min([c for c in track_counts if c > 0]) if frames_with_tracks > 0 else 0
    
    print(f"  Track statistics:")
    print(f"    Frames with tracks: {frames_with_tracks}/{total_frames}")
    print(f"    Avg tracks per frame (non-zero): {avg_tracks:.1f}")
    print(f"    Min tracks (non-zero): {min_tracks_nonzero}")
    print(f"    Max tracks: {max_tracks}")
    
    # Save to HDF5
    print(f"  Saving to HDF5...")
    with h5py.File(output_h5_path, 'w') as f:
        # Store video
        f.create_dataset('root/video', data=video_array, dtype='uint8')
        
        # Store tracks as separate datasets per frame
        tracks_group = f.create_group('root/tracks')
        for t, tracks_t in enumerate(per_frame_tracks_transformed):
            tracks_group.create_dataset(f'frame_{t:04d}', data=tracks_t, dtype='float32')
        
        # Metadata
        f.create_dataset('root/num_frames', data=total_frames)
        f.create_dataset('root/num_track_ts', data=num_track_ts)
    
    print(f"  ✓ Saved to {output_h5_path}")
    
    return {
        'total_frames': total_frames,
        'frames_with_tracks': frames_with_tracks,
        'avg_tracks': avg_tracks,
        'min_tracks_nonzero': min_tracks_nonzero,
        'max_tracks': max_tracks,
        'tracks': per_frame_tracks_transformed  # For displacement statistics
    }


def compute_displacement_statistics(all_displacements):
    """
    Compute mean and std of displacements.
    
    Args:
        all_displacements: List of displacement arrays, each (num_track_ts, N, 2)
    
    Returns:
        dict with 'displacement_mean' and 'displacement_std'
    """
    # Flatten all displacements to (N_total, 2)
    flattened = []
    for disp in all_displacements:
        # disp shape: (num_track_ts, N, 2)
        if disp.shape[1] > 0:  # Skip empty tracks
            flattened.append(disp.reshape(-1, 2))
    
    if len(flattened) == 0:
        print("⚠ Warning: No tracks found for computing statistics!")
        return {
            'displacement_mean': [0.0, 0.0],
            'displacement_std': [1.0, 1.0],
            'num_samples': 0
        }
    
    all_disp = np.concatenate(flattened, axis=0)  # (N_total, 2)
    
    # Compute statistics
    disp_mean = all_disp.mean(axis=0).tolist()  # [mean_x, mean_y]
    disp_std = all_disp.std(axis=0).tolist()    # [std_x, std_y]
    
    return {
        'displacement_mean': disp_mean,
        'displacement_std': disp_std,
        'num_samples': int(len(all_disp))
    }


def main():
    # Load configuration
    config = load_config("config.yaml")
    h5_config = config['create_h5_dataset']
    
    # Extract configuration
    target_size = h5_config['target_size']
    num_track_ts = h5_config['num_track_ts']
    input_images_dir = Path(h5_config['input_images_dir'])
    input_tracks_dir = Path(h5_config['input_tracks_dir'])
    output_h5_dir = Path(h5_config['output_h5_dir'])
    num_videos_to_process = h5_config['num_videos_to_process']
    continue_mode = h5_config.get('continue', False)
    
    # For displacement statistics computation
    all_displacements = []
    
    # Delete previous output directory if not continuing
    if not continue_mode and output_h5_dir.exists():
        print(f"\n🗑️  Deleting previous output directory: {output_h5_dir}")
        shutil.rmtree(output_h5_dir)
        print(f"✓ Previous output deleted\n")
    
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
    print(f"Target size: {target_size}x{target_size}")
    print(f"Future track timesteps: {num_track_ts}")
    print(f"Input images directory: {input_images_dir}")
    print(f"Input tracks directory: {input_tracks_dir}")
    print(f"Output H5 directory: {output_h5_dir}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Continue mode: {continue_mode}")
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    start_idx = 0
    
    # Efficient resume: Check only the last output .hdf5 file
    if continue_mode and output_h5_dir.exists():
        existing_h5 = sorted([f for f in output_h5_dir.iterdir() if f.suffix == '.hdf5'])
        
        if existing_h5:
            last_h5 = existing_h5[-1]
            video_name = last_h5.stem  # e.g., "00039.hdf5" -> "00039"
            
            print(f"\n✓ Last HDF5 file {video_name}.hdf5 exists")
            
            # Find index in video_folders list and start from next
            for i, vf in enumerate(video_folders):
                if vf.name == video_name:
                    start_idx = i + 1
                    skipped_videos = i + 1
                    break
            
            if start_idx > 0:
                print(f"🔄 Resuming from video {start_idx + 1}/{len(video_folders)}")
                print(f"⏭️  Skipping {start_idx} already processed videos\n")
    
    # Process each video folder
    all_stats = []
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
        
        try:
            stats = process_video(video_folder, tracks_file, output_h5_path, h5_config)
            
            if stats is not None:
                all_stats.append(stats)
                total_videos_processed += 1
                
                # Collect tracks for displacement statistics
                # Convert positions to displacements (pos[t] - pos[0])
                for tracks_t in stats['tracks']:
                    # tracks_t shape: (num_track_ts, N, 2)
                    if tracks_t.shape[1] > 0:  # Skip empty tracks
                        initial_pos = tracks_t[0]  # (N, 2)
                        displacements = tracks_t - initial_pos[None, :, :]  # (num_track_ts, N, 2)
                        all_displacements.append(displacements)
                
                video_elapsed = time.time() - video_start_time
                print(f"\n✓ Completed in {video_elapsed:.2f}s")
            else:
                print(f"\n⚠ Skipped due to missing data")
        
        except Exception as e:
            print(f"❌ Error processing {video_folder.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ ALL VIDEOS PROCESSED!")
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
    
    # Compute and save displacement statistics
    if all_displacements:
        print(f"\n{'='*60}")
        print(f"COMPUTING DISPLACEMENT STATISTICS")
        print(f"{'='*60}")
        print(f"Number of track samples: {len(all_displacements)}")
        
        disp_stats = compute_displacement_statistics(all_displacements)
        
        print(f"Displacement mean: {disp_stats['displacement_mean']}")
        print(f"Displacement std: {disp_stats['displacement_std']}")
        print(f"Total samples: {disp_stats['num_samples']}")
        
        # Save to YAML file in H5 directory
        stats_path = output_h5_dir / 'normalization_stats.yaml'
        with open(stats_path, 'w') as f:
            yaml.dump(disp_stats, f)
        
        print(f"✓ Saved displacement statistics to: {stats_path}")
        print(f"{'='*60}")
    else:
        print(f"\n⚠ Warning: No displacement data collected!")
    
    print(f"\nTotal time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
    print(f"H5 files saved to: {output_h5_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

