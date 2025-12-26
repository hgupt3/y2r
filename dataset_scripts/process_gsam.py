import os
import sys
import torch
from omegaconf import OmegaConf
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import pandas as pd

# Add thirdparty to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty"))

from sam2.grounded_sam2 import gsam_video, sam3_video

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    return OmegaConf.load(config_path)


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
    detection_model = gsam_config.get('detection_model', 'grounding-dino')  # Default to grounding-dino
    key_frame_idx = gsam_config.get('key_frame_idx', 0)  # Default to 0 if not specified
    
    # Get text_prompt from config (None means use metadata CSV)
    config_text_prompt = gsam_config.get('text_prompt', None)
    use_metadata_prompts = config_text_prompt is None
    
    # Load metadata CSV for dynamic text prompts (only for 'world' mode when text_prompt is null)
    clip_to_noun = {}
    if args.mode == 'world' and use_metadata_prompts:
        metadata_path = Path(config['common']['metadata_file'])
        if metadata_path.exists():
            print(f"Loading metadata from: {metadata_path}")
            metadata_df = pd.read_csv(metadata_path)
            # Create mapping from clip_id to transformed noun
            for _, row in metadata_df.iterrows():
                clip_id = int(row['clip_id'])
                noun = str(row['noun'])
                clip_to_noun[clip_id] = noun
            print(f"✓ Loaded {len(clip_to_noun)} clip-to-noun mappings")
        else:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Required when text_prompt is null.")
    input_images_dir = Path(gsam_config['input_images_dir'])
    device = gsam_config['device']
    output_masks_dir = Path(gsam_config['output_masks_dir'])
    num_videos_to_process = gsam_config['num_videos_to_process']
    vis_flag = gsam_config['vis_flag']
    vis_path = Path(gsam_config['vis_path']) if vis_flag else None
    vis_fps = gsam_config['vis_fps']
    continue_mode = gsam_config.get('continue', False)
    
    # Delete previous output directories if not continuing
    if not continue_mode:
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
    print(f"Detection model: {detection_model}")
    if use_metadata_prompts:
        print(f"Text prompt: Dynamic from metadata.csv (noun column)")
    else:
        print(f"Text prompt: {config_text_prompt}")
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
    
    # Initialize variables for detection models
    grounding_model = None
    processor = None
    florence2_model = None
    florence2_processor = None
    sam2_predictor = None
    sam3_predictor = None
    
    # Load detection model based on configuration
    if detection_model == "grounding-dino":
        print("Loading Grounding DINO model...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        print("✅ Grounding DINO loaded")
        
        print("Loading SAM2 video predictor...")
        from sam2.build_sam import build_sam2_video_predictor
        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",  # Hydra expects relative path to sam2 package
            str(PROJECT_ROOT / "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"),
            device=device
        )
        print("✅ SAM2 loaded")
        
    elif detection_model == "florence-2":
        print(f"Loading Florence-2 model (microsoft/Florence-2-large)...")
        from transformers import AutoProcessor, AutoModelForCausalLM
        florence2_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        florence2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            attn_implementation="eager"  # Use eager attention to avoid compatibility issues
        ).eval().to(device)
        print("✅ Florence-2 loaded")
        
        print("Loading SAM2 video predictor...")
        from sam2.build_sam import build_sam2_video_predictor
        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",  # Hydra expects relative path to sam2 package
            str(PROJECT_ROOT / "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"),
            device=device
        )
        print("✅ SAM2 loaded")
        
    elif detection_model == "sam3":
        print("Loading SAM3 video predictor (unified detection + segmentation)...")
        # Add sam3 to path
        sys.path.insert(0, str(PROJECT_ROOT / "thirdparty/sam3"))
        from sam3.model_builder import build_sam3_video_predictor
        sam3_predictor = build_sam3_video_predictor()
        print("✅ SAM3 loaded")
        
    else:
        raise ValueError(f"Unknown detection_model: {detection_model}. Must be 'grounding-dino', 'florence-2', or 'sam3'.")
    
    print(f"{'='*60}\n")
    
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    videos_to_process = []
    
    # Comprehensive resume: Check ALL output .pt files for completeness
    if continue_mode and output_masks_dir.exists():
        print(f"\n{'='*60}")
        print(f"CHECKING ALL EXISTING OUTPUTS FOR COMPLETENESS")
        print(f"{'='*60}\n")
        
        for idx, video_folder in enumerate(video_folders):
            mask_file = output_masks_dir / f"{video_folder.name}.pt"
            
            # Check if .pt file exists (1a - just existence check)
            if not mask_file.exists():
                videos_to_process.append(idx)
                continue
            
            # File exists, so it's complete (torch.save is atomic)
            print(f"✓ Mask file {video_folder.name}.pt exists")
            skipped_videos += 1
        
        # Check for and delete orphaned visualization files (no corresponding .pt file)
        if vis_flag and vis_path and vis_path.exists():
            for vis_file in vis_path.glob("*.mp4"):
                video_name = vis_file.stem  # e.g., "00039.mp4" -> "00039"
                mask_file = output_masks_dir / f"{video_name}.pt"
                if not mask_file.exists():
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
    
    # Process each video folder that needs processing
    for idx in videos_to_process:
        video_folder = video_folders[idx]
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_folder.name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Prepare save path for visualization if enabled
        save_path = str(vis_path) if vis_flag else None
        video_name = video_folder.name  # Get the video name (e.g., 00000, 00001, etc.)
        
        # Get text prompt: use config if provided, otherwise get from metadata
        clip_id = int(video_name)  # Convert folder name to int (e.g., "00042" -> 42)
        if config_text_prompt is not None:
            text_prompt = config_text_prompt
        elif clip_id in clip_to_noun:
            text_prompt = clip_to_noun[clip_id] + "."  # Add period for GSAM format
        else:
            print(f"⚠ No noun found in metadata for clip_id {clip_id}, skipping {video_folder.name}")
            continue
        
        # Run detection and segmentation on the video folder
        current_key_frame = key_frame_idx
        
        if detection_model == "sam3":
            # Use SAM3's unified detection + segmentation
            print(f"Running SAM3 detection and segmentation...")
            print(f"  Text prompt: \"{text_prompt}\"")
            
            masks_4d, obj_dict = sam3_video(
                frames_dir=str(video_folder),
                text_prompt=text_prompt,
                key_frame_idx=current_key_frame,
                save_path=save_path,
                video_name=video_name,
                device=device,
                fps=vis_fps,
                sam3_predictor=sam3_predictor
            )
            
            # If detection failed and we weren't already using middle frame, retry with middle frame
            if (masks_4d is None or obj_dict is None or len(obj_dict) == 0) and current_key_frame != -1:
                print(f"  ⚠ No object detected at frame {current_key_frame}, retrying with middle frame...")
                masks_4d, obj_dict = sam3_video(
                    frames_dir=str(video_folder),
                    text_prompt=text_prompt,
                    key_frame_idx=-1,  # Middle frame
                    save_path=save_path,
                    video_name=video_name,
                    device=device,
                    fps=vis_fps,
                    sam3_predictor=sam3_predictor
                )
        else:
            # Use GSAM (Grounding DINO/Florence-2 + SAM2)
            print(f"Running GSAM detection and segmentation...")
            print(f"  Text prompt: \"{text_prompt}\"")
            
            masks_4d, obj_dict = gsam_video(
                frames_dir=str(video_folder),
                text_prompt=text_prompt,
                key_frame_idx=current_key_frame,
                save_path=save_path,
                video_name=video_name,
                device=device,
                fps=vis_fps,
                grounding_model=grounding_model,
                processor=processor,
                sam2_predictor=sam2_predictor,
                detection_model=detection_model,
                florence2_model=florence2_model,
                florence2_processor=florence2_processor
            )
            
            # If detection failed and we weren't already using middle frame, retry with middle frame
            if (masks_4d is None or obj_dict is None or len(obj_dict) == 0) and current_key_frame != -1:
                print(f"  ⚠ No object detected at frame {current_key_frame}, retrying with middle frame...")
                masks_4d, obj_dict = gsam_video(
                    frames_dir=str(video_folder),
                    text_prompt=text_prompt,
                    key_frame_idx=-1,  # Middle frame
                    save_path=save_path,
                    video_name=video_name,
                    device=device,
                    fps=vis_fps,
                    grounding_model=grounding_model,
                    processor=processor,
                    sam2_predictor=sam2_predictor,
                    detection_model=detection_model,
                    florence2_model=florence2_model,
                    florence2_processor=florence2_processor
                )
        
        # If still no detection, skip this clip
        if masks_4d is None or obj_dict is None or len(obj_dict) == 0:
            print(f"⚠ No object detected in {video_folder.name} after retrying, skipping clip")
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
    print(f"Total videos: {len(video_folders)}")
    if continue_mode and skipped_videos > 0:
        print(f"Videos processed: {total_videos_processed}")
        print(f"Videos skipped: {skipped_videos}")
    else:
        print(f"Videos processed: {total_videos_processed}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    if total_videos_processed > 0:
        print(f"Average: {elapsed_time/total_videos_processed:.2f}s per video")
    print(f"Masks saved to: {output_masks_dir}")
    if vis_flag:
        print(f"Visualizations saved to: {vis_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

