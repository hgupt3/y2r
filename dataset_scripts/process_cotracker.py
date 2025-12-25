import os
import sys
import torch
from omegaconf import OmegaConf
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import imageio

# Add thirdparty to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty"))

from cotracker3.tap import cotracker


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    return OmegaConf.load(config_path)


def load_frames_from_directory(frames_dir):
    """
    Load all frames from a directory into a tensor.
    
    Args:
        frames_dir: Path to directory containing frame images (00000.png, 00001.png, etc.)
    
    Returns:
        frames_tensor: Tensor of shape (T, C, H, W)
        frame_files: List of frame file paths (for reference)
    """
    frame_files = sorted(list(frames_dir.glob("*.png")))
    
    if len(frame_files) == 0:
        return None, None
    
    # Load all frames
    frames_list = []
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames_list.append(frame)
    
    # Stack into numpy array and convert to tensor
    frames_array = np.stack(frames_list, axis=0)  # (T, H, W, C)
    frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
    
    return frames_tensor, frame_files


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


def process_windows_sequentially(model, frames, combined_mask, windows, grid_size, 
                                  visibility_threshold, device, vis_flag, vis_path, video_name):
    """
    Process sliding windows sequentially.
    
    Args:
        model: Pre-loaded CoTracker model
        frames: Full video frames tensor (T, C, H, W)
        combined_mask: Combined mask across all objects (T, H, W)
        windows: List of (start, end) tuples
        grid_size: Grid density for tracking
        visibility_threshold: Minimum visibility to keep tracks (0-1)
        device: Device to run on
        vis_flag: Whether to save visualizations
        vis_path: Path to save visualizations
        video_name: Name of the video for visualization filenames
    
    Returns:
        all_tracks: List of track tensors, one per window
                   Each element has shape (1, T, N, 2) where:
                   - 1 = batch dimension
                   - T = window length (number of frames in window)
                   - N = number of tracked points (varies per window after filtering)
                   - 2 = (x, y) coordinates in pixels
    """
    all_tracks = []
    
    # Process windows sequentially with progress bar
    num_windows = len(windows)
    
    with tqdm(total=num_windows, desc="  Processing windows", unit="window", leave=False) as pbar:
        for window_idx, (start_idx, end_idx) in enumerate(windows):
            # Extract window frames
            window_frames = frames[start_idx:end_idx]  # (T, C, H, W)
            
            # Extract mask at start frame
            window_mask = combined_mask[start_idx]  # (H, W)
            
            # Run CoTracker on single window (visualization disabled in cotracker() now)
            with torch.no_grad():
                pred_tracks, pred_visibility = cotracker(
                    model=model,
                    frames=window_frames,
                    segm_mask=window_mask,
                    grid_size=grid_size,
                    visibility_threshold=visibility_threshold,
                    query_frame_idx=0,
                    save_dir=None,  # Visualization now handled separately
                    device=device
                )
                
                all_tracks.append(pred_tracks.cpu())
            
            # Update progress bar
            pbar.update(1)
            
            # Clear GPU cache periodically
            if (window_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    return all_tracks


def create_trajectory_summary_video(frames, all_tracks, windows, vis_path, video_name, vis_fps, pad_value=0, linewidth=1):
    """
    Create a trajectory summary video where each frame shows a window's first frame
    with the full trajectory drawn on it.
    
    Args:
        frames: Full video frames tensor (T, C, H, W)
        all_tracks: List of track tensors, one per window [(1, T_window, N_i, 2), ...]
        windows: List of (start, end) tuples indicating window boundaries
        vis_path: Path to save visualization video
        video_name: Name of the video
        vis_fps: FPS for the output video
        pad_value: Padding value for visualization
        linewidth: Line width for drawing tracks
    """
    from cotracker3.cotracker.utils.visualizer import Visualizer
    import imageio
    
    print(f"\nCreating trajectory summary video...")
    
    # Create visualizer
    vis = Visualizer(save_dir=str(vis_path), pad_value=pad_value, linewidth=linewidth)
    
    # Collect rendered frames
    rendered_frames = []
    
    for window_idx, ((start_idx, end_idx), window_tracks) in enumerate(zip(windows, all_tracks)):
        # Skip windows with no tracks
        if window_tracks.shape[2] == 0:
            print(f"  Window {window_idx}: No tracks (skipping)")
            continue
        
        # Extract first frame of this window
        first_frame = frames[start_idx]  # (C, H, W)
        
        # Darken for better visibility (like in original visualization)
        first_frame_darkened = first_frame * 0.5
        
        # Render trajectory on this frame
        rendered_frame = vis.visualize_trajectory_on_frame(
            frame=first_frame_darkened,
            tracks=window_tracks,  # (1, T_window, N, 2)
            visibility=None,  # Already filtered by visibility in cotracker()
            segm_mask=None,
            query_frame=0,
            opacity=1.0,
        )
        
        rendered_frames.append(rendered_frame)
    
    if len(rendered_frames) == 0:
        print(f"⚠ No frames to visualize (all windows had no tracks)")
        return
    
    # Save as video
    os.makedirs(vis_path, exist_ok=True)
    output_path = vis_path / f"{video_name}_trajectory_summary.mp4"
    
    video_writer = imageio.get_writer(output_path, fps=vis_fps)
    for frame in rendered_frames:
        video_writer.append_data(frame)
    video_writer.close()
    
    print(f"✓ Trajectory summary video saved: {output_path}")
    print(f"  Total frames in summary: {len(rendered_frames)}")


def main():
    # Load configuration
    config = load_config("config.yaml")
    cotracker_config = config['cotracker']
    
    # Extract configuration
    window_length = cotracker_config['window_length']
    stride = cotracker_config['stride']
    grid_size = cotracker_config['grid_size']
    visibility_threshold = cotracker_config['visibility_threshold']
    mask_erosion_pixels = cotracker_config['mask_erosion_pixels']
    subtract_human_masks = cotracker_config.get('subtract_human_masks', True)
    human_mask_dilation_pixels = cotracker_config['human_mask_dilation_pixels']
    input_images_dir = Path(cotracker_config['input_images_dir'])
    input_masks_dir = Path(cotracker_config['input_masks_dir'])
    input_human_masks_dir = Path(cotracker_config['input_human_masks_dir'])
    output_tracks_dir = Path(cotracker_config['output_tracks_dir'])
    device = cotracker_config['device']
    num_videos_to_process = cotracker_config['num_videos_to_process']
    vis_flag = cotracker_config['vis_flag']
    vis_path = Path(cotracker_config['vis_path']) if vis_flag else None
    vis_fps = cotracker_config['vis_fps']
    continue_mode = cotracker_config.get('continue', False)
    
    # Delete previous output directories if not continuing
    if not continue_mode:
        if output_tracks_dir.exists():
            print(f"\n🗑️  Deleting previous tracks directory: {output_tracks_dir}")
            shutil.rmtree(output_tracks_dir)
            print(f"✓ Previous tracks deleted")
        if vis_flag and vis_path and vis_path.exists():
            print(f"🗑️  Deleting previous visualizations directory: {vis_path}")
            shutil.rmtree(vis_path)
            print(f"✓ Previous visualizations deleted\n")
    
    # Create output directories
    output_tracks_dir.mkdir(parents=True, exist_ok=True)
    if vis_flag:
        vis_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video folders (00000, 00001, etc.)
    video_folders = sorted([d for d in input_images_dir.iterdir() if d.is_dir()])
    
    # Limit to num_videos_to_process if specified
    if num_videos_to_process is not None:
        video_folders = video_folders[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"COTRACKER PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Window length: {window_length} frames")
    print(f"Stride: {stride} frames")
    print(f"Grid size: {grid_size}")
    print(f"Mask erosion: {mask_erosion_pixels} pixels")
    print(f"Subtract human masks: {'Enabled' if subtract_human_masks else 'Disabled'}")
    if subtract_human_masks:
        print(f"Human mask dilation: {human_mask_dilation_pixels} pixels")
    print(f"Input images directory: {input_images_dir}")
    print(f"Input masks directory: {input_masks_dir}")
    if subtract_human_masks:
        print(f"Input human masks directory: {input_human_masks_dir}")
    print(f"Output tracks directory: {output_tracks_dir}")
    print(f"Device: {device}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Visualization: {'Enabled' if vis_flag else 'Disabled'}")
    if vis_flag:
        print(f"Visualization path: {vis_path}")
    print(f"{'='*60}\n")
    
    # Load CoTracker model once
    print(f"{'='*60}")
    print("LOADING COTRACKER MODEL (one-time initialization)")
    print(f"{'='*60}")
    print("Loading CoTracker3 Offline model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    print("✅ CoTracker3 Offline loaded")
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    total_windows_processed = 0
    skipped_videos = 0
    start_time = time.time()
    videos_to_process = []
    
    # Comprehensive resume: Check ALL output .pt files for completeness
    if continue_mode and output_tracks_dir.exists():
        print(f"\n{'='*60}")
        print(f"CHECKING ALL EXISTING OUTPUTS FOR COMPLETENESS")
        print(f"{'='*60}\n")
        
        for idx, video_folder in enumerate(video_folders):
            tracks_file = output_tracks_dir / f"{video_folder.name}.pt"
            
            # Check if .pt file exists (1a - just existence check)
            if not tracks_file.exists():
                videos_to_process.append(idx)
                continue
            
            # File exists, so it's complete (torch.save is atomic)
            print(f"✓ Tracks file {video_folder.name}.pt exists")
            skipped_videos += 1
        
        # Check for and delete orphaned visualization files (no corresponding .pt file)
        if vis_flag and vis_path and vis_path.exists():
            for vis_file in vis_path.glob("*_trajectory_summary.mp4"):
                # Extract video name from filename (e.g., "00039_trajectory_summary.mp4" -> "00039")
                video_name = vis_file.stem.replace("_trajectory_summary", "")
                tracks_file = output_tracks_dir / f"{video_name}.pt"
                if not tracks_file.exists():
                    print(f"🗑️  Deleting orphaned visualization: {vis_file}")
                    vis_file.unlink()
        
        if videos_to_process:
            print(f"\n🔄 Found {len(videos_to_process)} videos to process")
            print(f"⏭️  Skipping {skipped_videos} already complete videos\n")
        else:
            print(f"\n✅ All {skipped_videos} videos are already complete!\n")
    else:
        # Not continuing, process all videos
        videos_to_process = list(range(len(video_folders)))
    
    # Process each video folder with progress bar, starting from videos_to_process
    for idx in videos_to_process:
        video_folder = video_folders[idx]
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_folder.name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Load frames
        print(f"Loading frames from {video_folder}...")
        frames, frame_files = load_frames_from_directory(video_folder)
        
        if frames is None:
            print(f"⚠ No frames found in {video_folder.name}")
            continue
        
        num_frames = frames.shape[0]
        print(f"✓ Loaded {num_frames} frames")
        
        # Load masks
        mask_file = input_masks_dir / f"{video_folder.name}.pt"
        if not mask_file.exists():
            print(f"⚠ Mask file not found: {mask_file}")
            continue
        
        print(f"Loading masks from {mask_file}...")
        mask_data = torch.load(mask_file, map_location='cpu')
        masks = mask_data['masks']  # Shape: (T, O, H, W) from gsam_video
        
        print(f"✓ Loaded masks with shape: {masks.shape}")
        
        # gsam_video outputs masks with shape (T, O, H, W)
        # where T=frames, O=objects, H=height, W=width
        # Combine across all objects to get (T, H, W)
        if masks.dim() == 4:  # (T, O, H, W) - expected format
            # Combine all objects: any pixel that's True in any object mask
            combined_mask = torch.any(masks > 0.5, dim=1).float()  # (T, H, W)
        elif masks.dim() == 3:  # (T, H, W) - single object already combined
            combined_mask = masks.float()
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}. Expected (T, O, H, W) or (T, H, W)")
        
        print(f"✓ Combined mask shape: {combined_mask.shape}")
        
        # Load human masks and subtract from object masks (if enabled)
        if subtract_human_masks:
            human_mask_file = input_human_masks_dir / f"{video_folder.name}.pt"
            if not human_mask_file.exists():
                print(f"⚠ Human mask file not found: {human_mask_file}")
                print(f"⚠ Skipping human mask subtraction for this video")
                combined_human_mask = torch.zeros_like(combined_mask)
            else:
                print(f"Loading human masks from {human_mask_file}...")
                human_mask_data = torch.load(human_mask_file, map_location='cpu')
                human_masks = human_mask_data['masks']  # Shape: (T, O, H, W) from gsam_video
                
                print(f"✓ Loaded human masks with shape: {human_masks.shape}")
                
                # Combine human masks across all objects to get (T, H, W)
                if human_masks.dim() == 4:  # (T, O, H, W) - expected format
                    # Combine all humans: any pixel that's True in any human mask
                    combined_human_mask = torch.any(human_masks > 0.5, dim=1).float()  # (T, H, W)
                elif human_masks.dim() == 3:  # (T, H, W) - single object already combined
                    combined_human_mask = human_masks.float()
                else:
                    print(f"⚠ Unexpected human mask shape: {human_masks.shape}. Skipping subtraction")
                    combined_human_mask = torch.zeros_like(combined_mask)
                
                print(f"✓ Combined human mask shape: {combined_human_mask.shape}")
            
            # Dilate human mask to create buffer zone around humans
            if human_mask_dilation_pixels > 0 and combined_human_mask.sum() > 0:
                print(f"Dilating human mask by {human_mask_dilation_pixels} pixels...")
                kernel = np.ones((human_mask_dilation_pixels * 2 + 1, human_mask_dilation_pixels * 2 + 1), np.uint8)
                dilated_human_mask_list = []
                for t in range(combined_human_mask.shape[0]):
                    mask_np = combined_human_mask[t].numpy().astype(np.uint8)
                    dilated = cv2.dilate(mask_np, kernel, iterations=1)
                    dilated_human_mask_list.append(torch.from_numpy(dilated).float())
                combined_human_mask = torch.stack(dilated_human_mask_list, dim=0)
                print(f"✓ Human mask dilated")
            
            # Subtract human mask from object mask (clamp to [0, 1])
            print(f"Subtracting human masks from object masks...")
            combined_mask = torch.clamp(combined_mask - combined_human_mask, 0, 1)
            print(f"✓ Human masks subtracted")
        else:
            print(f"Human mask subtraction disabled (subtract_human_masks=False)")
        
        # Erode mask to avoid tracking unreliable edge points
        if mask_erosion_pixels > 0:
            print(f"Eroding mask by {mask_erosion_pixels} pixels...")
            kernel = np.ones((mask_erosion_pixels * 2 + 1, mask_erosion_pixels * 2 + 1), np.uint8)
            eroded_mask_list = []
            for t in range(combined_mask.shape[0]):
                mask_np = combined_mask[t].numpy().astype(np.uint8)
                eroded = cv2.erode(mask_np, kernel, iterations=1)
                eroded_mask_list.append(torch.from_numpy(eroded).float())
            combined_mask = torch.stack(eroded_mask_list, dim=0)
            print(f"✓ Mask eroded")
        
        # Generate sliding windows
        windows = generate_sliding_windows(num_frames, window_length, stride)
        
        if len(windows) == 0:
            print(f"⚠ Video too short ({num_frames} frames) for window length {window_length}")
            continue
        
        print(f"Generated {len(windows)} sliding windows")
        print(f"Window examples: {windows[:3]}{'...' if len(windows) > 3 else ''}")
        
        # Process windows sequentially
        print(f"\nProcessing {len(windows)} windows...")
        all_tracks = process_windows_sequentially(
            model=model,
            frames=frames,
            combined_mask=combined_mask,
            windows=windows,
            grid_size=grid_size,
            visibility_threshold=visibility_threshold,
            device=device,
            vis_flag=vis_flag,
            vis_path=vis_path,
            video_name=video_folder.name
        )
        
        # Create trajectory summary visualization if enabled
        if vis_flag:
            create_trajectory_summary_video(
                frames=frames,
                all_tracks=all_tracks,
                windows=windows,
                vis_path=vis_path,
                video_name=video_folder.name,
                vis_fps=vis_fps,
                pad_value=0,
                linewidth=1
            )
        
        # Save tracks
        # NOTE: tracks is a Python list, not a stacked tensor
        # tracks[i] contains the i-th window's tracks with shape (1, T, N_i, 2)
        # where N_i can differ per window due to occlusion filtering
        tracks_filename = output_tracks_dir / f"{video_folder.name}.pt"
        torch.save({
            'tracks': all_tracks,
            'windows': windows,
            'video_name': video_folder.name,
            'window_length': window_length,
            'stride': stride,
            'grid_size': grid_size
        }, tracks_filename)
        
        video_elapsed = time.time() - video_start_time
        print(f"\n✓ Completed in {video_elapsed:.2f}s")
        print(f"  Number of windows: {len(windows)}")
        print(f"  Tracks saved to: {tracks_filename}")
        if vis_flag:
            print(f"  Visualizations saved to: {vis_path}")
        
        total_videos_processed += 1
        total_windows_processed += len(windows)
        
        # Clear GPU cache between videos
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    
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
    print(f"Total windows: {total_windows_processed}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
        print(f"Average: {elapsed_time/total_windows_processed:.2f}s per window")
    print(f"Tracks saved to: {output_tracks_dir}")
    if vis_flag:
        print(f"Visualizations saved to: {vis_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

