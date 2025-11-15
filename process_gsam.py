import os
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import shutil
from sam2.grounded_sam2 import gsam_video


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process videos with GSAM for object detection and segmentation")
    parser.add_argument('--mode', type=str, default='world', choices=['world', 'human'],
                        help='Select configuration mode: "world" (default) or "human"')
    args = parser.parse_args()
    
    # Load configuration from config file
    config = load_config("config.yaml")
    
    # Select the appropriate config based on mode
    config_key = 'gsam' if args.mode == 'world' else 'gsam_human'
    gsam_config = config[config_key]
    
    # Extract configuration
    text_prompt = gsam_config['text_prompt']
    key_frame_idx = gsam_config.get('key_frame_idx', 0)  # Default to 0 if not specified
    input_images_dir = Path(gsam_config['input_images_dir'])
    device = gsam_config['device']
    output_masks_dir = Path(gsam_config['output_masks_dir'])
    num_videos_to_process = gsam_config['num_videos_to_process']
    vis_flag = gsam_config['vis_flag']
    vis_path = Path(gsam_config['vis_path']) if vis_flag else None
    vis_fps = gsam_config['vis_fps']
    delete_previous_output = gsam_config.get('delete_previous_output', False)
    
    # Delete previous output directories if requested
    if delete_previous_output:
        if output_masks_dir.exists():
            print(f"\n🗑️  Deleting previous masks directory: {output_masks_dir}")
            shutil.rmtree(output_masks_dir)
            print(f"✓ Previous masks deleted")
        if vis_flag and vis_path and vis_path.exists():
            print(f"🗑️  Deleting previous visualizations directory: {vis_path}")
            shutil.rmtree(vis_path)
            print(f"✓ Previous visualizations deleted\n")
    
    # Create output directories
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    if vis_flag:
        vis_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video folders (00000, 00001, etc.)
    video_folders = sorted([d for d in input_images_dir.iterdir() if d.is_dir()])
    
    # Limit to num_videos_to_process if specified
    if num_videos_to_process is not None:
        video_folders = video_folders[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"GSAM VIDEO PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Text prompt: {text_prompt}")
    print(f"Key frame index: {key_frame_idx} ({'middle frame' if key_frame_idx == -1 else f'frame {key_frame_idx}'})")
    print(f"Input directory: {input_images_dir}")
    print(f"Output masks directory: {output_masks_dir}")
    print(f"Device: {device}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Visualization: {'Enabled' if vis_flag else 'Disabled'}")
    if vis_flag:
        print(f"Visualization path: {vis_path}")
    print(f"{'='*60}\n")
    
    # Load models once before processing all videos
    print(f"{'='*60}")
    print("LOADING MODELS (one-time initialization)")
    print(f"{'='*60}")
    print("Loading Grounding DINO model...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    print("✅ Grounding DINO loaded")
    
    print("Loading SAM2 video predictor...")
    from sam2.build_sam import build_sam2_video_predictor
    sam2_predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
        device=device
    )
    print("✅ SAM2 loaded")
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    start_time = time.time()
    
    # Process each video folder
    for idx, video_folder in enumerate(video_folders):
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_folder.name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Prepare save path for visualization if enabled
        save_path = str(vis_path) if vis_flag else None
        video_name = video_folder.name  # Get the video name (e.g., 00000, 00001, etc.)
        
        # Run GSAM on the video folder
        print(f"Running GSAM detection and segmentation...")
        masks_4d, obj_dict = gsam_video(
            frames_dir=str(video_folder),
            text_prompt=text_prompt,
            key_frame_idx=key_frame_idx,  # Pass the key frame index
            save_path=save_path,
            video_name=video_name,  # Pass video name for unique filenames
            device=device,
            fps=vis_fps,
            grounding_model=grounding_model,  # Pass pre-loaded model
            processor=processor,  # Pass pre-loaded processor
            sam2_predictor=sam2_predictor  # Pass pre-loaded SAM2 predictor
        )
        
        if masks_4d is None:
            print(f"⚠ Failed to process {video_folder.name}")
            continue
        
        # Save masks as PyTorch file
        mask_filename = output_masks_dir / f"{video_folder.name}.pt"
        torch.save({
            'masks': torch.from_numpy(masks_4d),
            'object_dict': obj_dict,
            'video_name': video_folder.name
        }, mask_filename)
        
        video_elapsed = time.time() - video_start_time
        print(f"\n✓ Completed in {video_elapsed:.2f}s")
        print(f"  Masks shape: {masks_4d.shape}")
        print(f"  Objects detected: {len(obj_dict)}")
        print(f"  Object IDs: {obj_dict}")
        print(f"  Saved masks to: {mask_filename}")
        if vis_flag:
            print(f"  Saved visualization to: {vis_path / f'{video_folder.name}.mp4'}")
        
        total_videos_processed += 1
        
        # Clear GPU cache and force garbage collection to prevent memory accumulation
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ ALL VIDEOS PROCESSED!")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos_processed}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
    print(f"Masks saved to: {output_masks_dir}")
    if vis_flag:
        print(f"Visualizations saved to: {vis_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

