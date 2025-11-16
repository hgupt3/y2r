import os
import sys
import yaml
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import tempfile

# Add DiffuEraser to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffuEraser'))

from diffuEraser.diffueraser.diffueraser import DiffuEraser
from diffuEraser.propainter.inference import Propainter, get_device


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def convert_pt_masks_to_frames(mask_pt_path, output_dir, target_size=None):
    """
    Convert .pt mask file to frame sequence
    Args:
        mask_pt_path: Path to .pt file containing masks
        output_dir: Directory to save mask frames
        target_size: Optional (width, height) tuple to resize masks
    """
    # Load .pt mask data
    mask_data = torch.load(mask_pt_path, map_location='cpu')
    masks = mask_data['masks']  # Shape: (T, O, H, W)
    
    # Combine all objects into single binary mask per frame
    if masks.dim() == 4:  # (T, O, H, W)
        combined_mask = torch.any(masks > 0.5, dim=1)  # (T, H, W)
    elif masks.dim() == 3:  # (T, H, W)
        combined_mask = masks > 0.5
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame as PNG
    num_frames = combined_mask.shape[0]
    for t in range(num_frames):
        mask_np = combined_mask[t].numpy().astype(np.uint8) * 255
        
        # Resize if target size specified
        if target_size is not None:
            mask_np = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_NEAREST)
        
        frame_path = os.path.join(output_dir, f"{t:05d}.png")
        cv2.imwrite(frame_path, mask_np)
    
    return num_frames


def get_frame_size(frames_dir):
    """Get the size of frames in a directory"""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        return None
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    img = cv2.imread(first_frame_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)  # Return as (width, height)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process videos with DiffuEraser to erase humans")
    args = parser.parse_args()
    
    # Load configuration from config file
    config = load_config("config.yaml")
    diffueraser_config = config['diffueraser']
    preprocess_config = config['preprocess']
    
    # Extract configuration
    input_images_dir = Path(diffueraser_config['input_images_dir'])
    fps = preprocess_config['target_fps']  # Get FPS from preprocess config
    input_masks_dir = Path(diffueraser_config['input_masks_dir'])
    output_frames_dir = Path(diffueraser_config['output_frames_dir'])
    device = diffueraser_config['device']
    num_videos_to_process = diffueraser_config['num_videos_to_process']
    max_img_size = diffueraser_config['max_img_size']
    mask_dilation_iter = diffueraser_config['mask_dilation_iter']
    
    # Propainter params
    ref_stride = diffueraser_config['ref_stride']
    neighbor_length = diffueraser_config['neighbor_length']
    subvideo_length = diffueraser_config['subvideo_length']
    
    continue_mode = diffueraser_config.get('continue', False)
    
    # Model paths - set internally using relative paths
    script_dir = Path(__file__).parent
    diffueraser_dir = script_dir / "diffuEraser"
    base_model_path = str(diffueraser_dir / "weights" / "stable-diffusion-v1-5")
    vae_path = str(diffueraser_dir / "weights" / "sd-vae-ft-mse")
    diffueraser_path = str(diffueraser_dir / "weights" / "diffuEraser")
    propainter_model_dir = str(diffueraser_dir / "weights" / "propainter")
    
    # Delete previous output directory if not continuing
    if not continue_mode and output_frames_dir.exists():
        print(f"\n🗑️  Deleting previous output directory: {output_frames_dir}")
        shutil.rmtree(output_frames_dir)
        print(f"✓ Previous output deleted\n")
    
    # Create output directory
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video folders (00000, 00001, etc.)
    video_folders = sorted([d for d in input_images_dir.iterdir() if d.is_dir()])
    
    # Limit to num_videos_to_process if specified
    if num_videos_to_process is not None:
        video_folders = video_folders[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"DIFFUERASER VIDEO PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input frames directory: {input_images_dir}")
    print(f"Input masks directory: {input_masks_dir}")
    print(f"Output frames directory: {output_frames_dir}")
    print(f"Device: {device}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Max image size: {max_img_size}")
    print(f"Mask dilation iterations: {mask_dilation_iter}")
    print(f"FPS: {fps}")
    print(f"{'='*60}\n")
    
    # Load models once before processing all videos
    print(f"{'='*60}")
    print("LOADING MODELS (one-time initialization)")
    print(f"{'='*60}")
    print("Loading DiffuEraser model...")
    ckpt = "2-Step"
    diffueraser = DiffuEraser(device, base_model_path, vae_path, diffueraser_path, ckpt=ckpt)
    print("✅ DiffuEraser loaded")
    
    print("Loading Propainter model...")
    propainter = Propainter(propainter_model_dir, device=device)
    print("✅ Propainter loaded")
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    start_idx = 0
    
    # Efficient resume: Check only the last output folder
    if continue_mode and output_frames_dir.exists():
        existing_outputs = sorted([d for d in output_frames_dir.iterdir() if d.is_dir()])
        
        if existing_outputs:
            last_output = existing_outputs[-1]
            video_name = last_output.name
            
            # Find corresponding input folder
            input_folder = input_images_dir / video_name
            
            if input_folder.exists():
                # Get last frame from input
                input_frames = sorted(input_folder.glob("*.png"))
                
                if input_frames:
                    last_frame_name = input_frames[-1].name
                    last_output_frame = last_output / last_frame_name
                    
                    if last_output_frame.exists():
                        print(f"\n✓ Last output {video_name} is complete (last frame: {last_frame_name})")
                        # Find index in video_folders list and start from next
                        for i, vf in enumerate(video_folders):
                            if vf.name == video_name:
                                start_idx = i + 1
                                skipped_videos = i + 1
                                break
                    else:
                        print(f"\n⚠️  Last output {video_name} is incomplete (missing last frame: {last_frame_name})")
                        print(f"🗑️  Deleting incomplete output: {last_output}")
                        shutil.rmtree(last_output)
                        # Find index and reprocess this video
                        for i, vf in enumerate(video_folders):
                            if vf.name == video_name:
                                start_idx = i
                                skipped_videos = i
                                break
            
            if start_idx > 0:
                print(f"🔄 Resuming from video {start_idx + 1}/{len(video_folders)}")
                print(f"⏭️  Skipping {start_idx} already processed videos\n")
    
    # Process each video folder, starting from start_idx
    for idx in range(start_idx, len(video_folders)):
        video_folder = video_folders[idx]
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_folder.name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Check if mask file exists
        mask_file = input_masks_dir / f"{video_folder.name}.pt"
        if not mask_file.exists():
            print(f"⚠ Mask file not found: {mask_file}")
            continue
        
        # Get frame size from input frames
        frame_size = get_frame_size(str(video_folder))
        if frame_size is None:
            print(f"⚠ Could not read frames from {video_folder}")
            continue
        
        print(f"Frame size: {frame_size[0]}x{frame_size[1]}")
        
        # Create temporary directories for mask frames and priori frames
        temp_base_dir = tempfile.mkdtemp(prefix=f"diffueraser_{video_folder.name}_")
        temp_mask_dir = os.path.join(temp_base_dir, "masks")
        temp_priori_dir = os.path.join(temp_base_dir, "priori")
        
        try:
            # Convert .pt masks to frame sequence
            print(f"Converting masks from {mask_file}...")
            num_mask_frames = convert_pt_masks_to_frames(mask_file, temp_mask_dir, target_size=frame_size)
            print(f"✓ Converted {num_mask_frames} mask frames")
            
            # Run Propainter to generate priori
            print(f"Running Propainter to generate priori...")
            propainter.forward(
                video=str(video_folder),
                mask=temp_mask_dir,
                output_path=temp_priori_dir,
                video_length=1000,  # Not used for frame directories
                ref_stride=ref_stride,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
                mask_dilation=mask_dilation_iter,
                save_fps=fps,
                output_as_frames=True
            )
            print(f"✓ Priori generation complete")
            
            # Run DiffuEraser
            print(f"Running DiffuEraser inference...")
            output_video_dir = output_frames_dir / video_folder.name
            diffueraser.forward(
                validation_image=str(video_folder),
                validation_mask=temp_mask_dir,
                priori=temp_priori_dir,
                output_path=str(output_video_dir),
                max_img_size=max_img_size,
                video_length=1000,  # Not used for frame directories
                mask_dilation_iter=mask_dilation_iter,
                fps=fps,
                output_as_frames=True
            )
            print(f"✓ DiffuEraser inference complete")
            
            video_elapsed = time.time() - video_start_time
            print(f"\n✓ Completed in {video_elapsed:.2f}s")
            print(f"  Saved cleaned frames to: {output_video_dir}")
            
            total_videos_processed += 1
            
        except Exception as e:
            print(f"❌ Error processing {video_folder.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up temporary directories
            if os.path.exists(temp_base_dir):
                shutil.rmtree(temp_base_dir)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
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
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
    print(f"Cleaned frames saved to: {output_frames_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

