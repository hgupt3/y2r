"""
ViPE Processing Script
Runs NVIDIA ViPE on videos to extract camera poses and dense depth maps.
These outputs can then be used by SpaTracker for 3D point tracking.

Usage:
    python process_vipe.py
"""

import os
import sys
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Load config and set CUDA device BEFORE any torch imports
def _load_config_early():
    with open(SCRIPT_DIR / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    vipe_config = config.get('vipe', {})
    spatracker_config = config.get('spatracker', {})
    device = vipe_config.get('device', spatracker_config.get('device', 'cuda'))
    if device.startswith("cuda:"):
        gpu_id = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
    return config, device

_config_early, _device_early = _load_config_early()

# NOW import torch and other heavy dependencies
import torch
import numpy as np
from tqdm import tqdm
import time
import shutil
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "vipe"))
from vipe import get_config_path, make_pipeline
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.base import ProcessedVideoStream, FrameAttribute
from vipe.utils.logging import configure_logging


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_vipe_on_video(video_path, output_dir, pipeline_name="default", save_viz=False, vis_path=None):
    """
    Run ViPE pipeline on a single video and save results as .pt file.
    
    Args:
        video_path: Path to video file or directory of frames
        output_dir: Directory to save outputs (.pt files)
        pipeline_name: ViPE pipeline config name ("default", "no_vda", "wide_angle")
        save_viz: Whether to save visualization video
        vis_path: Directory to save visualization videos (if different from output_dir)
    
    Returns:
        Dictionary with:
        - depth: Tensor (T, H, W) metric depth maps
        - poses: Tensor (T, 4, 4) cam2world SE3 matrices  
        - intrinsics: Tensor (4,) [fx, fy, cx, cy]
        - num_frames: int
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if vis_path:
        vis_path = Path(vis_path)
    
    # Configure hydra overrides
    # When save_viz is enabled, we also need save_artifacts for the interactive viser visualizer
    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={vis_path if save_viz and vis_path else output_dir}",
        f"pipeline.output.save_artifacts={'true' if save_viz else 'false'}",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
    ]
    
    # Initialize hydra with ViPE config
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)
    
    # Create pipeline and enable output streams return
    vipe_pipeline = make_pipeline(args.pipeline)
    vipe_pipeline.return_output_streams = True  # Get processed output with poses & depths
    
    # Determine if input is video or frame directory
    if video_path.is_file():
        video_stream = ProcessedVideoStream(
            RawMp4Stream(video_path), []
        ).cache(desc="Reading video stream")
        video_name = video_path.stem
    else:
        video_stream = ProcessedVideoStream(
            FrameDirStream(video_path), []
        ).cache(desc="Reading image frames")
        video_name = video_path.name
    
    # Run pipeline
    result = vipe_pipeline.run(video_stream)
    
    # Extract data from output streams
    output_stream = result.output_streams[0]
    
    # Collect poses, depths, and intrinsics
    poses_list = []
    depths_list = []
    intrinsics = None
    
    pose_list = output_stream.get_stream_attribute(FrameAttribute.POSE)
    depth_list = output_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
    intr_list = output_stream.get_stream_attribute(FrameAttribute.INTRINSICS)
    
    for i, (pose, depth, intr) in enumerate(zip(pose_list, depth_list, intr_list)):
        if pose is not None:
            poses_list.append(pose.matrix().cpu())
        if depth is not None:
            depths_list.append(depth.cpu())
        if intr is not None and intrinsics is None:
            intrinsics = intr.cpu()
    
    # Stack into tensors
    poses = torch.stack(poses_list, dim=0) if poses_list else None
    depths = torch.stack(depths_list, dim=0) if depths_list else None
    
    result = {
        'poses': poses,           # (T, 4, 4) cam2world
        'depth': depths,          # (T, H, W) metric depth
        'intrinsics': intrinsics, # (4,) [fx, fy, cx, cy]
        'num_frames': len(poses_list),
        'video_name': video_name,
    }
    
    # Save .pt file to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{video_name}.pt"
    torch.save(result, pt_path)
    result['pt_path'] = str(pt_path)
    
    # Note: when save_viz=True, artifacts are saved to vis_path for interactive viser visualizer
    # The viser visualizer can be run with: vipe visualize <vis_path> --port 20540
    if save_viz and vis_path:
        viz_video = vis_path / "vipe" / f"{video_name}_vis.mp4"
        if viz_video.exists():
            result['viz_video_path'] = str(viz_video)
        result['viser_path'] = str(vis_path)  # Path to use with `vipe visualize`
    
    return result


def process_frame_directory_with_vipe(frames_dir, output_dir, config):
    """
    Process a directory of frames with ViPE and save as .pt file.
    
    Args:
        frames_dir: Path to directory containing frames (00000.png, 00001.png, etc.)
        output_dir: Directory to save outputs
        config: ViPE config dictionary
    
    Returns:
        Path to saved .pt file or None if failed
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Check if frames exist
    frame_files = sorted(list(frames_dir.glob("*.png")))
    if len(frame_files) == 0:
        print(f"No frames found in {frames_dir}")
        return None
    
    print(f"Processing {len(frame_files)} frames from {frames_dir.name}")
    
    try:
        result = run_vipe_on_video(
            video_path=frames_dir,
            output_dir=output_dir,
            pipeline_name=config.get('pipeline', 'default'),
            save_viz=config.get('vis_flag', False),
            vis_path=config.get('vis_path'),
        )
        
        # .pt file is already saved by run_vipe_on_video
        return Path(result.get('pt_path'))
        
    except Exception as e:
        print(f"Error processing {frames_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run ViPE on all videos."""
    print("=" * 60)
    print("ViPE Processing Pipeline")
    print("Extracting camera poses and depth maps for SpaTracker")
    print("=" * 60)
    
    # Use the config that was already loaded at module init
    config = _config_early
    device = _device_early
    
    # Get ViPE-specific config
    vipe_config = config.get('vipe', {})
    spatracker_config = config.get('spatracker', {})
    
    # Use spatracker's input directory as source
    input_dir = Path(vipe_config.get('input_images_dir', spatracker_config.get('input_images_dir')))
    output_dir = Path(vipe_config.get('output_dir', str(input_dir.parent / "vipe_results")))
    
    # Other settings
    num_videos = vipe_config.get('num_videos_to_process', spatracker_config.get('num_videos_to_process'))
    continue_flag = vipe_config.get('continue', spatracker_config.get('continue', False))
    vis_flag = vipe_config.get('vis_flag', False)
    vis_path = vipe_config.get('vis_path')
    
    print(f"Device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if vis_flag and vis_path:
        print(f"Visualization directory: {vis_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of video directories
    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if num_videos is not None:
        video_dirs = video_dirs[:num_videos]
    
    print(f"Found {len(video_dirs)} video directories to process")
    
    # Initialize ViPE logger
    logger = configure_logging()
    
    # Track processed videos
    processed = []
    failed = []
    
    for i, video_dir in enumerate(tqdm(video_dirs, desc="Processing videos")):
        video_name = video_dir.name
        
        # Check if already processed
        expected_pt_path = output_dir / f"{video_name}.pt"
        if continue_flag and expected_pt_path.exists():
            print(f"Skipping {video_name} (already processed)")
            processed.append(video_name)
            continue
        
        print(f"\n[{i+1}/{len(video_dirs)}] Processing: {video_name}")
        start_time = time.time()
        
        pt_path = process_frame_directory_with_vipe(video_dir, output_dir, vipe_config)
        
        elapsed = time.time() - start_time
        
        if pt_path is not None:
            processed.append(video_name)
            print(f"  ✓ Completed in {elapsed:.1f}s")
            print(f"  Saved: {pt_path}")
        else:
            failed.append(video_name)
            print(f"  ✗ Failed after {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"  Processed: {len(processed)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed videos: {failed}")
    if vis_flag and vis_path:
        print(f"\n  Interactive 3D Visualization:")
        print(f"    Run: vipe visualize {vis_path} --port 20540")
        print(f"    Open: http://localhost:20540 in browser")
    print("=" * 60)


if __name__ == "__main__":
    main()

