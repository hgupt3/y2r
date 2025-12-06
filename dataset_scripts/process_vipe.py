"""
ViPE Processing Script
Runs NVIDIA ViPE on videos to extract camera poses and dense depth maps.
Outputs directly in TAPIP3D-ready .npz format.

Usage:
    python process_vipe.py
"""

import os
import sys
import yaml
import logging
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
    
    # Suppress huggingface tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Disable all internal tqdm progress bars (must be set before tqdm import)
    os.environ["TQDM_DISABLE"] = "1"
    
    return config, device

_config_early, _device_early = _load_config_early()

# NOW import torch and other heavy dependencies
import warnings
warnings.filterwarnings("ignore", message=".*deprecated/invalid arguments.*")

import torch
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf


sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "vipe"))
from vipe import get_config_path, make_pipeline
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.base import ProcessedVideoStream, FrameAttribute


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_frames(frames_dir: Path, num_frames: int = None) -> np.ndarray:
    """
    Load PNG frames from a directory into a video array.
    
    Args:
        frames_dir: Path to directory containing PNG frames (00000.png, 00001.png, etc.)
        num_frames: Expected number of frames (for validation)
    
    Returns:
        video: (T, H, W, 3) uint8 array
    """
    frame_files = sorted(frames_dir.glob("*.png"))
    
    if num_frames is not None and len(frame_files) != num_frames:
        print(f"Warning: Expected {num_frames} frames but found {len(frame_files)}")
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        frames.append(np.array(img))
    
    return np.stack(frames, axis=0).astype(np.uint8)


def convert_intrinsics_to_matrix(intrinsics_vec: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Convert ViPE intrinsics from [fx, fy, cx, cy] to (T, 3, 3) camera matrix.
    
    Args:
        intrinsics_vec: (4,) array [fx, fy, cx, cy]
        num_frames: Number of frames T
    
    Returns:
        intrinsics: (T, 3, 3) camera matrix array
    """
    fx, fy, cx, cy = intrinsics_vec
    
    # Create 3x3 camera matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Broadcast to (T, 3, 3)
    return np.tile(K[np.newaxis, :, :], (num_frames, 1, 1))


def convert_poses_to_extrinsics(poses: np.ndarray) -> np.ndarray:
    """
    Convert cam2world poses to world2cam extrinsics.
    
    Args:
        poses: (T, 4, 4) cam2world matrices
    
    Returns:
        extrinsics: (T, 4, 4) world2cam matrices
    """
    return np.linalg.inv(poses).astype(np.float32)


def run_vipe_on_video(
    video_path, 
    frames_dir, 
    output_dir, 
    pipeline_name="default", 
    save_viz=False, 
    vis_path=None, 
    fixed_fov_degrees=None,
    fov_min_degrees=None,
    fov_max_degrees=None,
    optimize_intrinsics=True,
):
    """
    Run ViPE pipeline on a single video and save results as TAPIP3D-ready .npz file.
    
    Args:
        video_path: Path to video file or directory of frames (for ViPE processing)
        frames_dir: Path to original frames directory (for loading RGB into output)
        output_dir: Directory to save outputs (.npz files)
        pipeline_name: ViPE pipeline config name ("default", "no_vda", "wide_angle", "lyra")
        save_viz: Whether to save visualization video
        vis_path: Directory to save visualization videos (if different from output_dir)
        fixed_fov_degrees: If set, bypass GeoCalib entirely with a fixed FOV
        fov_min_degrees: If set, clamp GeoCalib's estimate to this minimum
        fov_max_degrees: If set, clamp GeoCalib's estimate to this maximum
        optimize_intrinsics: Whether SLAM can refine intrinsics during bundle adjustment
    
    Returns:
        Dictionary with TAPIP3D-ready data:
        - video: (T, H, W, 3) uint8 RGB frames
        - depths: (T, H, W) float32 metric depth maps
        - intrinsics: (T, 3, 3) float32 camera matrices
        - extrinsics: (T, 4, 4) float32 world2cam matrices
    """
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    if vis_path:
        vis_path = Path(vis_path)
    
    # Configure hydra overrides
    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={vis_path if save_viz and vis_path else output_dir}",
        f"pipeline.output.save_artifacts={'true' if save_viz else 'false'}",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
    ]
    
    # Add FOV overrides for initial estimation (GeoCalib clamping)
    if fixed_fov_degrees is not None:
        # Bypass GeoCalib entirely with a fixed FOV
        overrides.append(f"pipeline.init.fixed_fov_degrees={fixed_fov_degrees}")
    else:
        # Use GeoCalib with optional min/max clamping
        if fov_min_degrees is not None:
            overrides.append(f"pipeline.init.fov_min_degrees={fov_min_degrees}")
        if fov_max_degrees is not None:
            overrides.append(f"pipeline.init.fov_max_degrees={fov_max_degrees}")
    
    # Pass FOV bounds to SLAM for bounded intrinsic optimization (use + to add new keys)
    if fov_min_degrees is not None:
        overrides.append(f"+pipeline.slam.fov_min_degrees={fov_min_degrees}")
    if fov_max_degrees is not None:
        overrides.append(f"+pipeline.slam.fov_max_degrees={fov_max_degrees}")
    
    # Control whether SLAM can refine intrinsics during bundle adjustment
    overrides.append(f"pipeline.slam.optimize_intrinsics={'true' if optimize_intrinsics else 'false'}")
    
    # Initialize hydra with ViPE config
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)
    
    # Create pipeline and enable output streams return
    vipe_pipeline = make_pipeline(args.pipeline)
    vipe_pipeline.return_output_streams = True
    
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
    intrinsics_vec = None
    
    pose_list = output_stream.get_stream_attribute(FrameAttribute.POSE)
    depth_list = output_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
    intr_list = output_stream.get_stream_attribute(FrameAttribute.INTRINSICS)
    
    for i, (pose, depth, intr) in enumerate(zip(pose_list, depth_list, intr_list)):
        if pose is not None:
            poses_list.append(pose.matrix().cpu().numpy())
        if depth is not None:
            depths_list.append(depth.cpu().numpy())
        if intr is not None and intrinsics_vec is None:
            intrinsics_vec = intr.cpu().numpy()
    
    # Stack into arrays
    poses = np.stack(poses_list, axis=0) if poses_list else None  # (T, 4, 4) cam2world
    depths = np.stack(depths_list, axis=0) if depths_list else None  # (T, H, W)
    num_frames = len(poses_list)
    
    # Load original RGB frames
    video = load_frames(frames_dir, num_frames)
    
    # Validate dimensions
    assert video.shape[0] == depths.shape[0] == poses.shape[0], \
        f"Frame count mismatch: video={video.shape[0]}, depths={depths.shape[0]}, poses={poses.shape[0]}"
    assert video.shape[1:3] == depths.shape[1:3], \
        f"Resolution mismatch: video={video.shape[1:3]}, depths={depths.shape[1:3]}"
    
    # Convert to TAPIP3D format
    
    # Convert intrinsics from [fx, fy, cx, cy] to (T, 3, 3) camera matrix
    intrinsics = convert_intrinsics_to_matrix(intrinsics_vec, num_frames)
    
    # Convert poses (cam2world) to extrinsics (world2cam)
    extrinsics = convert_poses_to_extrinsics(poses)
    
    # Prepare TAPIP3D-ready output (NO video field)
    tapip3d_data = {
        'depths': depths.astype(np.float16),     # (T, H, W) float16 (50% smaller)
        'intrinsics': intrinsics,                # (T, 3, 3) float32
        'extrinsics': extrinsics,                # (T, 4, 4) float32
    }
    
    # Save .npz file with compression
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{video_name}.npz"
    np.savez_compressed(npz_path, **tapip3d_data)
    
    result = {
        'npz_path': str(npz_path),
        'video_name': video_name,
        'num_frames': num_frames,
    }
    
    # Handle visualization artifacts
    if save_viz and vis_path:
        viz_video = vis_path / "vipe" / f"{video_name}_vis.mp4"
        if viz_video.exists():
            result['viz_video_path'] = str(viz_video)
        result['viser_path'] = str(vis_path)
    
    return result


def process_frame_directory_with_vipe(frames_dir, output_dir, config):
    """
    Process a directory of frames with ViPE and save as TAPIP3D-ready .npz file.
    
    Args:
        frames_dir: Path to directory containing frames (00000.png, 00001.png, etc.)
        output_dir: Directory to save outputs
        config: ViPE config dictionary
    
    Returns:
        Path to saved .npz file or None if failed
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Check if frames exist
    frame_files = sorted(list(frames_dir.glob("*.png")))
    if len(frame_files) == 0:
        tqdm.write(f"  No frames found in {frames_dir}")
        return None
    
    result = run_vipe_on_video(
        video_path=frames_dir,
        frames_dir=frames_dir,  # Same directory for RGB frames
        output_dir=output_dir,
        pipeline_name=config.get('pipeline', 'default'),
        save_viz=config.get('vis_flag', False),
        vis_path=config.get('vis_path'),
        fixed_fov_degrees=config.get('fixed_fov_degrees'),
        fov_min_degrees=config.get('fov_min_degrees'),
        fov_max_degrees=config.get('fov_max_degrees'),
        optimize_intrinsics=config.get('optimize_intrinsics', True),
    )
    
    return Path(result.get('npz_path'))


def main():
    """Main function to run ViPE on all videos."""
    print("=" * 60)
    print("ViPE Processing Pipeline")
    print("Extracting camera poses and depth maps (TAPIP3D-ready format)")
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
    fixed_fov_degrees = vipe_config.get('fixed_fov_degrees')
    fov_min_degrees = vipe_config.get('fov_min_degrees')
    fov_max_degrees = vipe_config.get('fov_max_degrees')
    optimize_intrinsics = vipe_config.get('optimize_intrinsics', True)
    
    print(f"Device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: TAPIP3D-ready .npz")
    if fixed_fov_degrees is not None:
        opt_str = " + SLAM bounded optimization" if optimize_intrinsics else ", fixed"
        print(f"Intrinsics: Fixed FOV {fixed_fov_degrees}°{opt_str}")
    elif fov_min_degrees is not None or fov_max_degrees is not None:
        opt_str = " + SLAM bounded optimization" if optimize_intrinsics else ", no SLAM optimization"
        print(f"Intrinsics: GeoCalib clamped to [{fov_min_degrees or '?'}, {fov_max_degrees or '?'}]°{opt_str}")
    else:
        opt_str = " + SLAM optimization" if optimize_intrinsics else ""
        print(f"Intrinsics: GeoCalib (automatic){opt_str}")
    if vis_flag and vis_path:
        print(f"Visualization directory: {vis_path}")
    
    # Delete previous output if not continuing
    if not continue_flag:
        if output_dir.exists():
            print(f"Deleting previous output directory: {output_dir}")
            shutil.rmtree(output_dir)
        if vis_flag and vis_path:
            vis_path_obj = Path(vis_path)
            if vis_path_obj.exists():
                print(f"Deleting previous visualization directory: {vis_path}")
                shutil.rmtree(vis_path_obj)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of video directories
    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if num_videos is not None:
        video_dirs = video_dirs[:num_videos]
    
    print(f"Found {len(video_dirs)} video directories to process")
    
    # Suppress verbose ViPE logging (BA iterations, etc.)
    logging.getLogger("vipe").setLevel(logging.WARNING)
    logging.getLogger("vipe.slam").setLevel(logging.WARNING)
    logging.getLogger("vipe.pipeline").setLevel(logging.WARNING)
    
    # Track processed videos
    processed = []
    failed = []
    
    # Explicitly enable our progress bar (overrides TQDM_DISABLE env var)
    pbar = tqdm(video_dirs, desc="Episodes", position=0, leave=True, disable=False)
    for video_dir in pbar:
        video_name = video_dir.name
        pbar.set_postfix_str(video_name)
        
        # Check if already processed (now .npz instead of .pt)
        expected_npz_path = output_dir / f"{video_name}.npz"
        if continue_flag and expected_npz_path.exists():
            processed.append(video_name)
            continue
        
        npz_path = process_frame_directory_with_vipe(video_dir, output_dir, vipe_config)
        
        if npz_path is not None:
            processed.append(video_name)
        else:
            failed.append(video_name)
            tqdm.write(f"  ✗ {video_name} failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"  Processed: {len(processed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output format: TAPIP3D-ready .npz files")
    if failed:
        print(f"  Failed videos: {failed}")
    if vis_flag and vis_path:
        print(f"\n  Interactive 3D Visualization:")
        print(f"    Run: vipe visualize {vis_path} --port 20540")
        print(f"    Open: http://localhost:20540 in browser")
    print("\n  To run TAPIP3D inference:")
    print(f"    python thirdparty/TAPIP3D/inference.py --input_path {output_dir}/<video>.npz --checkpoint thirdparty/TAPIP3D/checkpoints/tapip3d_final.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
