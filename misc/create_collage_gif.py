import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import imageio

def load_video_frames(video_folder):
    """Load all frames from a video folder"""
    frames = []
    frame_files = sorted(video_folder.glob("*.png"))
    
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is not None:
            # Convert BGR to RGB for PIL
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    
    return frames

def create_grid_frame(video_frames_list, frame_idx, grid_size=7):
    """
    Create a single frame with a grid of videos
    Args:
        video_frames_list: List of video frame lists (one per video)
        frame_idx: Current frame index
        grid_size: Size of the grid (7x7 = 49 videos)
    """
    if not video_frames_list or len(video_frames_list[0]) == 0:
        return None
    
    # Get frame size from first video
    frame_height, frame_width = video_frames_list[0][0].shape[:2]
    
    # Create empty canvas for the grid
    grid_height = frame_height * grid_size
    grid_width = frame_width * grid_size
    grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place each video in the grid
    for i, video_frames in enumerate(video_frames_list):
        if i >= grid_size * grid_size:
            break
        
        # Calculate grid position
        row = i // grid_size
        col = i % grid_size
        
        # Get the frame (loop if necessary)
        video_frame_idx = frame_idx % len(video_frames)
        frame = video_frames[video_frame_idx]
        
        # Place frame in grid
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        grid_frame[y_start:y_end, x_start:x_end] = frame
    
    return grid_frame

def create_collage_gif(processed_videos_dir, output_file, grid_size=7, speed_multiplier=3, 
                       target_duration=18, output_resolution=(1080, 1080), original_fps=12):
    """
    Create a collage GIF from processed videos
    Args:
        processed_videos_dir: Directory containing processed video folders
        output_file: Output GIF file path
        grid_size: Size of the grid (default 7x7)
        speed_multiplier: Speed up factor (3x means skip 2 out of 3 frames)
        target_duration: Target duration in seconds for the output GIF
        output_resolution: Target resolution (width, height) for the final GIF
        original_fps: FPS of the original processed videos
    """
    print(f"\n{'='*60}")
    print(f"VIDEO COLLAGE GIF GENERATOR")
    print(f"{'='*60}")
    print(f"Grid size: {grid_size}x{grid_size} ({grid_size*grid_size} videos)")
    print(f"Speed multiplier: {speed_multiplier}x")
    print(f"Target duration: {target_duration}s")
    print(f"Output resolution: {output_resolution[0]}x{output_resolution[1]}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Get all video folders
    video_folders = sorted([f for f in Path(processed_videos_dir).iterdir() if f.is_dir()])
    
    if len(video_folders) < grid_size * grid_size:
        print(f"⚠️  Warning: Found {len(video_folders)} videos, but need {grid_size*grid_size} for a {grid_size}x{grid_size} grid.")
        print(f"   Will duplicate videos to fill the grid.\n")
    
    # Randomly select and order videos
    required_videos = grid_size * grid_size
    if len(video_folders) >= required_videos:
        selected_folders = random.sample(video_folders, required_videos)
    else:
        # Duplicate videos if we don't have enough
        selected_folders = []
        while len(selected_folders) < required_videos:
            selected_folders.extend(video_folders)
        selected_folders = selected_folders[:required_videos]
        random.shuffle(selected_folders)
    
    print(f"📁 Loading {len(selected_folders)} videos...\n")
    
    # Load all video frames
    video_frames_list = []
    for i, folder in enumerate(tqdm(selected_folders, desc="Loading videos", unit="video")):
        frames = load_video_frames(folder)
        if frames:
            video_frames_list.append(frames)
            if i == 0:
                print(f"   Frame size: {frames[0].shape[1]}x{frames[0].shape[0]}")
        else:
            print(f"   Warning: No frames found in {folder}")
    
    if not video_frames_list:
        print("❌ Error: No video frames loaded!")
        return
    
    # Calculate number of frames needed for target duration
    # For target_duration seconds at original_fps with speed_multiplier:
    # We need: target_duration * original_fps output frames
    # Which comes from: target_duration * original_fps * speed_multiplier source frames
    output_frames_needed = int(target_duration * original_fps)
    source_frames_needed = output_frames_needed * speed_multiplier
    
    min_frames = min(len(frames) for frames in video_frames_list)
    max_video_frames = max(len(frames) for frames in video_frames_list)
    
    # Use the calculated number of frames, but don't exceed available frames
    num_frames = min(source_frames_needed, max_video_frames)
    
    # Apply speed multiplier (skip frames)
    frame_indices = list(range(0, num_frames, speed_multiplier))
    
    # Ensure we have exactly the right number of output frames (or close to it)
    if len(frame_indices) > output_frames_needed:
        frame_indices = frame_indices[:output_frames_needed]
    
    print(f"\n📊 Statistics:")
    print(f"   Shortest video: {min_frames} frames")
    print(f"   Longest video: {max_video_frames} frames")
    print(f"   Source frames needed: {source_frames_needed}")
    print(f"   Source frames to use: {num_frames}")
    print(f"   Output frames (after {speed_multiplier}x speed): {len(frame_indices)} frames")
    print(f"   Output duration: {len(frame_indices)/original_fps:.2f}s at {original_fps} fps")
    
    # Get dimensions for final output
    sample_frame = video_frames_list[0][0]
    frame_height, frame_width = sample_frame.shape[:2]
    grid_height = frame_height * grid_size
    grid_width = frame_width * grid_size
    print(f"   Grid dimensions: {grid_width}x{grid_height}")
    print(f"   Final output: {output_resolution[0]}x{output_resolution[1]}\n")
    
    # Generate grid frames
    print(f"🎬 Generating collage frames...\n")
    grid_frames = []
    
    for frame_idx in tqdm(frame_indices, desc="Creating grid frames", unit="frame"):
        grid_frame = create_grid_frame(video_frames_list, frame_idx, grid_size)
        if grid_frame is not None:
            # Resize to target resolution
            if output_resolution is not None:
                grid_frame = cv2.resize(grid_frame, output_resolution, interpolation=cv2.INTER_AREA)
            grid_frames.append(grid_frame)
    
    if not grid_frames:
        print("❌ Error: No frames generated!")
        return
    
    # Save as GIF
    print(f"\n💾 Saving GIF...\n")
    
    # The GIF plays at original_fps (not sped up in playback)
    # The speed effect comes from skipping frames during creation
    gif_fps = original_fps
    
    # Save with imageio for better control
    imageio.mimsave(
        output_file,
        grid_frames,
        fps=gif_fps,
        loop=0  # 0 means infinite loop
    )
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    actual_duration = len(grid_frames) / gif_fps
    
    print(f"\n{'='*60}")
    print(f"✅ COLLAGE GIF CREATED!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Dimensions: {output_resolution[0]}x{output_resolution[1]}")
    print(f"Frames: {len(grid_frames)}")
    print(f"FPS: {gif_fps}")
    print(f"Duration: {actual_duration:.2f} seconds (target: {target_duration}s)")
    print(f"Effective speed: {speed_multiplier}x")
    print(f"{'='*60}\n")

def main():
    # CONFIGURATION
    PROCESSED_VIDEOS_DIR = "/Users/harshgupta/Desktop/process/processed_videos"
    OUTPUT_FILE = "/Users/harshgupta/Desktop/process/collage.gif"
    GRID_SIZE = 7  # 7x7 grid
    SPEED_MULTIPLIER = 3  # 3x speed
    TARGET_DURATION = 18  # Duration in seconds
    OUTPUT_RESOLUTION = (1080, 1080)  # Final GIF resolution
    ORIGINAL_FPS = 12  # Should match TARGET_FPS from process_videos.py
    
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    create_collage_gif(
        PROCESSED_VIDEOS_DIR,
        OUTPUT_FILE,
        grid_size=GRID_SIZE,
        speed_multiplier=SPEED_MULTIPLIER,
        target_duration=TARGET_DURATION,
        output_resolution=OUTPUT_RESOLUTION,
        original_fps=ORIGINAL_FPS
    )

if __name__ == "__main__":
    main()

