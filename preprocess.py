from tkinter import NONE
import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import yaml
import shutil

def resize_and_minimal_crop(frame, target_width, target_height):
    """
    Resize frame to minimize cropping, then center crop to exact target dimensions.
    This preserves aspect ratio better than square cropping.
    
    Args:
        frame: Input frame
        target_width: Target width in pixels
        target_height: Target height in pixels
    
    Returns:
        Processed frame of exact size target_width x target_height
    """
    height, width = frame.shape[:2]
    
    # Calculate scale factors for both dimensions
    width_scale = target_width / width
    height_scale = target_height / height
    
    # Use the larger scale factor to ensure we cover target dimensions
    # (minimizes how much we need to crop)
    scale = max(width_scale, height_scale)
    
    # Resize with the chosen scale
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Center crop to exact target dimensions
    start_x = (new_width - target_width) // 2
    start_y = (new_height - target_height) // 2
    
    cropped = resized[start_y:start_y + target_height, start_x:start_x + target_width]
    
    return cropped

def process_video(video_path, output_folder, video_index, target_width=512, target_height=384, target_fps=None):
    """
    Process a single video: extract frames, resize with minimal crop
    Args:
        target_width: Target width in pixels
        target_height: Target height in pixels
        target_fps: Target FPS for frame extraction. If None, extracts all frames.
    """
    start_time = time.time()
    
    # Create output folder for this video (e.g., 00000, 00001, etc.)
    video_output_dir = output_folder / f"{video_index:05d}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video information
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    # Calculate frame sampling interval
    if target_fps is not None and target_fps > 0:
        frame_interval = original_fps / target_fps
        expected_frames = int(duration * target_fps)
    else:
        frame_interval = 1.0
        expected_frames = total_frames
    
    print(f"\n{'='*60}")
    print(f"Video: {video_path.name}")
    print(f"Original size: {width}x{height} | FPS: {original_fps:.2f} | Duration: {duration:.2f}s")
    if target_fps is not None:
        print(f"Original frames: {total_frames} | Target FPS: {target_fps} | Expected frames: {expected_frames}")
    else:
        print(f"Total frames: {total_frames} | Extracting all frames")
    print(f"Target size: {target_width}x{target_height}")
    print(f"Output: {video_output_dir}")
    print(f"{'='*60}")
    
    frame_count = 0
    saved_frame_count = 0
    next_frame_to_save = 0.0
    
    # Process frames with progress bar
    with tqdm(total=total_frames, desc=f"Processing frames", unit="frame", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check if we should save this frame (based on target FPS)
            if frame_count >= next_frame_to_save:
                # Process frame: resize and minimal crop
                processed_frame = resize_and_minimal_crop(frame, target_width, target_height)
                
                # Save frame with name like 00000.png, 00001.png, etc.
                frame_filename = video_output_dir / f"{saved_frame_count:05d}.png"
                cv2.imwrite(str(frame_filename), processed_frame)
                
                saved_frame_count += 1
                next_frame_to_save += frame_interval
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    elapsed_time = time.time() - start_time
    fps_processed = saved_frame_count / elapsed_time if elapsed_time > 0 else 0
    
    if target_fps is not None:
        print(f"\n✓ Completed in {elapsed_time:.2f}s ({fps_processed:.2f} frames/sec)")
        print(f"  Read {frame_count} frames, saved {saved_frame_count} frames ({saved_frame_count/duration:.1f} fps) to {video_output_dir}\n")
    else:
        print(f"\n✓ Completed in {elapsed_time:.2f}s ({fps_processed:.2f} frames/sec)")
        print(f"  Saved {saved_frame_count} frames to {video_output_dir}\n")
    
    return saved_frame_count

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration from config file
    config = load_config("config.yaml")
    preprocess_config = config['preprocess']
    
    TARGET_WIDTH = preprocess_config['target_width']
    TARGET_HEIGHT = preprocess_config['target_height']
    TARGET_FPS = preprocess_config['target_fps']
    NUM_VIDEOS_TO_PROCESS = preprocess_config['num_videos_to_process']
    CONTINUE = preprocess_config.get('continue', False)
    
    start_time = time.time()
    
    # Set up paths from config
    raw_videos_dir = Path(preprocess_config['raw_videos_dir'])
    output_dir = Path(preprocess_config['output_dir'])
    
    # Delete previous output directory if not continuing
    if not CONTINUE and output_dir.exists():
        print(f"\n🗑️  Deleting previous output directory: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"✓ Previous output deleted\n")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all video files based on extensions from config, sorted
    video_files = []
    for ext in preprocess_config['video_extensions']:
        video_files.extend(raw_videos_dir.glob(ext))
    video_files = sorted(video_files)
    
    # Limit to NUM_VIDEOS_TO_PROCESS if specified
    if NUM_VIDEOS_TO_PROCESS is not None:
        video_files = video_files[:NUM_VIDEOS_TO_PROCESS]
        print(f"\n{'='*60}")
        print(f"VIDEO PROCESSING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Mode: Testing ({len(video_files)} video)")
        print(f"Target size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"Target FPS: {TARGET_FPS if TARGET_FPS else 'All frames'}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"VIDEO PROCESSING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Mode: Full processing ({len(video_files)} videos)")
        print(f"Target size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"Target FPS: {TARGET_FPS if TARGET_FPS else 'All frames'}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    
    total_frames_processed = 0
    skipped_videos = 0
    start_idx = 0
    
    # Efficient resume: Check only the last output folder
    if CONTINUE and output_dir.exists():
        existing_outputs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if existing_outputs:
            last_output = existing_outputs[-1]
            video_idx = int(last_output.name)  # e.g., "00039" -> 39
            
            if video_idx < len(video_files):
                video_path = video_files[video_idx]
                
                # Get video metadata to calculate expected frames
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / original_fps if original_fps > 0 else 0
                cap.release()
                
                # Calculate expected output frames
                if TARGET_FPS is not None and TARGET_FPS > 0:
                    expected_output_frames = int(duration * TARGET_FPS)
                else:
                    expected_output_frames = total_frames
                
                # Check if last frame exists
                if expected_output_frames > 0:
                    last_frame_path = last_output / f"{expected_output_frames - 1:05d}.png"
                    
                    if last_frame_path.exists():
                        print(f"\n✓ Last output {last_output.name} is complete ({expected_output_frames} frames)")
                        start_idx = video_idx + 1  # Resume from next video
                        skipped_videos = video_idx + 1  # Count all previous as skipped
                    else:
                        print(f"\n⚠️  Last output {last_output.name} is incomplete (expected {expected_output_frames} frames)")
                        print(f"🗑️  Deleting incomplete output: {last_output}")
                        shutil.rmtree(last_output)
                        start_idx = video_idx  # Reprocess this video
                        skipped_videos = video_idx  # Count all previous as skipped
                else:
                    # Can't determine expected frames, start from this video
                    start_idx = video_idx
                    skipped_videos = video_idx
            
            if start_idx > 0:
                print(f"🔄 Resuming from video {start_idx + 1}/{len(video_files)}")
                print(f"⏭️  Skipping {start_idx} already processed videos\n")
    
    # Process each video with overall progress, starting from start_idx
    for idx in range(start_idx, len(video_files)):
        video_path = video_files[idx]
        print(f"📹 Video {idx + 1}/{len(video_files)}")
        
        frames = process_video(video_path, output_dir, idx, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, target_fps=TARGET_FPS)
        if frames:
            total_frames_processed += frames
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ ALL VIDEOS PROCESSED!")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    if CONTINUE and skipped_videos > 0:
        print(f"Videos processed: {len(video_files) - skipped_videos}")
        print(f"Videos skipped: {skipped_videos}")
    print(f"Total frames: {total_frames_processed}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_frames_processed > 0:
        print(f"Average: {total_frames_processed/elapsed_time:.2f} frames/sec")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

