"""
ViPE Processing Script
Runs NVIDIA ViPE on videos to extract camera poses and dense depth maps.
Outputs directly in TAPIP3D-ready .npz format.

Supports parallel processing via num_workers config option.

Usage:
    python process_vipe.py
"""

import os
import sys
import logging
import traceback
import multiprocessing as mp
from pathlib import Path
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _setup_env_for_worker():
    """Setup environment variables for a worker process (called before torch import)."""
    # Suppress huggingface tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Disable all internal tqdm progress bars
    os.environ["TQDM_DISABLE"] = "1"


def _load_config_early():
    """Load config and set CUDA device BEFORE any torch imports."""
    config = OmegaConf.load(SCRIPT_DIR / "config.yaml")
    
    vipe_config = config.get('vipe', {})
    spatracker_config = config.get('spatracker', {})
    device = vipe_config.get('device', spatracker_config.get('device', 'cuda'))
    if device.startswith("cuda:"):
        gpu_id = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
    
    _setup_env_for_worker()
    
    return config, device


# Only load config in main process at module level
# Workers will load it themselves after spawn
if mp.current_process().name == 'MainProcess':
    _config_early, _device_early = _load_config_early()
else:
    _config_early, _device_early = None, None

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
from vipe.model_cache import VipeModelCache


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    return OmegaConf.load(config_path)


def load_frames(frames_dir: Path, num_frames: int = None) -> np.ndarray:
    """
    Load PNG frames from a directory into a video array.
    
    Args:
        frames_dir: Path to directory containing PNG frames (00000.png, 00001.png, etc.)
        num_frames: Expected number of frames (for validation)
    
    Returns:
        video: (T, H, W, 3) uint8 array
    """
    frame_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
    
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


def is_depth_flat(
    depths: np.ndarray,
    flat_std_threshold: float = 0.1,
    max_flat_frame_pct: float = 20.0,
) -> tuple[bool, float, float]:
    """
    Check if too many frames have flat depth (indicating ViPE failure).
    
    Args:
        depths: (T, H, W) float depth maps
        flat_std_threshold: A frame is "flat" if its std < this value (default 0.1)
        max_flat_frame_pct: Discard if more than this % of frames are flat (default 20%)
    
    Returns:
        Tuple of (should_discard, pct_flat_frames, overall_std)
    """
    T = depths.shape[0]
    frame_stds = np.std(depths, axis=(1, 2))  # (T,) std per frame
    num_flat = np.sum(frame_stds < flat_std_threshold)
    pct_flat = (num_flat / T) * 100
    overall_std = float(np.std(depths))
    
    should_discard = pct_flat > max_flat_frame_pct
    return should_discard, pct_flat, overall_std


def create_vipe_pipeline(
    pipeline_name="default",
    output_dir=None,
    save_viz=False,
    vis_path=None,
    fixed_fov_degrees=None,
    fov_min_degrees=None,
    fov_max_degrees=None,
    optimize_intrinsics=True,
):
    """
    Create and initialize ViPE pipeline with cached models for reuse across multiple videos.
    
    Args:
        pipeline_name: ViPE pipeline config name ("default", "no_vda", "wide_angle", "lyra")
        output_dir: Directory for outputs
        save_viz: Whether to save visualization video
        vis_path: Directory to save visualization videos
        fixed_fov_degrees: If set, bypass GeoCalib entirely with a fixed FOV
        fov_min_degrees: If set, clamp GeoCalib's estimate to this minimum
        fov_max_degrees: If set, clamp GeoCalib's estimate to this maximum
        optimize_intrinsics: Whether SLAM can refine intrinsics during bundle adjustment
    
    Returns:
        Tuple of (vipe_pipeline, model_cache) - pipeline and cache for reuse
    """
    # Configure hydra overrides
    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={vis_path if save_viz and vis_path else output_dir}",
        f"pipeline.output.save_artifacts={'true' if save_viz else 'false'}",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
    ]
    
    # Add FOV overrides for initial estimation (GeoCalib clamping)
    if fixed_fov_degrees is not None:
        overrides.append(f"pipeline.init.fixed_fov_degrees={fixed_fov_degrees}")
    else:
        if fov_min_degrees is not None:
            overrides.append(f"pipeline.init.fov_min_degrees={fov_min_degrees}")
        if fov_max_degrees is not None:
            overrides.append(f"pipeline.init.fov_max_degrees={fov_max_degrees}")
    
    # Pass FOV bounds to SLAM for bounded intrinsic optimization
    if fov_min_degrees is not None:
        overrides.append(f"+pipeline.slam.fov_min_degrees={fov_min_degrees}")
    if fov_max_degrees is not None:
        overrides.append(f"+pipeline.slam.fov_max_degrees={fov_max_degrees}")
    
    # Control whether SLAM can refine intrinsics during bundle adjustment
    overrides.append(f"pipeline.slam.optimize_intrinsics={'true' if optimize_intrinsics else 'false'}")
    
    # Initialize hydra with ViPE config
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)
    
    # Create model cache ONCE - this loads all expensive models
    print("  Loading models into cache...")
    model_cache = VipeModelCache.create(
        slam_cfg=args.pipeline.slam,
        init_cfg=args.pipeline.init,
        post_cfg=args.pipeline.post,
        device=torch.device("cuda"),
    )
    
    # Create pipeline with cached models
    vipe_pipeline = make_pipeline(args.pipeline, model_cache=model_cache)
    vipe_pipeline.return_output_streams = True
    
    return vipe_pipeline, model_cache


def run_vipe_on_video(
    vipe_pipeline,
    video_path, 
    frames_dir, 
    output_dir, 
    save_viz=False, 
    vis_path=None,
    flat_std_threshold=0.1,
    max_flat_frame_pct=20.0,
):
    """
    Run ViPE pipeline on a single video and save results as TAPIP3D-ready .npz file.
    
    Args:
        vipe_pipeline: Pre-initialized ViPE pipeline (reused across videos)
        video_path: Path to video file or directory of frames (for ViPE processing)
        frames_dir: Path to original frames directory (for loading RGB into output)
        output_dir: Directory to save outputs (.npz files)
        save_viz: Whether to save visualization video
        vis_path: Directory to save visualization videos (if different from output_dir)
        flat_std_threshold: A frame is "flat" if per-frame std < this (default 0.1)
        max_flat_frame_pct: Skip if more than this % of frames are flat (default 20%)
    
    Returns:
        Dictionary with results:
        - npz_path: Path to saved .npz file (None if skipped due to flat depth)
        - video_name: Name of the video
        - num_frames: Number of frames processed
        - pct_flat_frames: % of frames that are flat
        - skipped: True if skipped due to flat depth
    """
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    if vis_path:
        vis_path = Path(vis_path)
    
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
    
    # Run pipeline (reusing pre-loaded models)
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
    
    # Check for flat depth (indicates ViPE failure)
    should_discard, pct_flat, overall_std = is_depth_flat(
        depths,
        flat_std_threshold=flat_std_threshold,
        max_flat_frame_pct=max_flat_frame_pct,
    )
    
    if should_discard:
        # Skip saving - too many flat frames
        return {
            'npz_path': None,
            'video_name': video_name,
            'num_frames': num_frames,
            'pct_flat_frames': pct_flat,
            'skipped': True,
        }
    
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
        'pct_flat_frames': pct_flat,
        'skipped': False,
    }
    
    # Handle visualization artifacts
    if save_viz and vis_path:
        viz_video = vis_path / "vipe" / f"{video_name}_vis.mp4"
        if viz_video.exists():
            result['viz_video_path'] = str(viz_video)
        result['viser_path'] = str(vis_path)
    
    return result


def process_frame_directory_with_vipe(vipe_pipeline, frames_dir, output_dir, config):
    """
    Process a directory of frames with ViPE and save as TAPIP3D-ready .npz file.
    
    Args:
        vipe_pipeline: Pre-initialized ViPE pipeline (reused across videos)
        frames_dir: Path to directory containing frames (00000.png, 00001.png, etc.)
        output_dir: Directory to save outputs
        config: ViPE config dictionary
    
    Returns:
        Dict with processing results, or None if no frames found
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Check if frames exist
    frame_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
    if len(frame_files) == 0:
        tqdm.write(f"  No frames found in {frames_dir}")
        return None
    
    result = run_vipe_on_video(
        vipe_pipeline=vipe_pipeline,
        video_path=frames_dir,
        frames_dir=frames_dir,  # Same directory for RGB frames
        output_dir=output_dir,
        save_viz=config.get('vis_flag', False),
        vis_path=config.get('vis_path'),
        flat_std_threshold=config.get('flat_std_threshold', 0.1),
        max_flat_frame_pct=config.get('max_flat_frame_pct', 20.0),
    )
    
    return result


def _worker_process(worker_id, video_dirs, output_dir, vipe_config, progress_queue):
    """
    Worker process that initializes its own ViPE pipeline and processes a shard of videos.
    
    Args:
        worker_id: Worker index (for logging)
        video_dirs: List of video directories to process (this worker's shard)
        output_dir: Output directory for .npz files
        vipe_config: ViPE config dictionary (resolved, no OmegaConf interpolation)
        progress_queue: Queue to report progress back to main process
    """
    # Setup environment for this worker
    _setup_env_for_worker()
    
    # Suppress verbose logging
    logging.getLogger("vipe").setLevel(logging.WARNING)
    logging.getLogger("vipe.slam").setLevel(logging.WARNING)
    logging.getLogger("vipe.pipeline").setLevel(logging.WARNING)
    
    # Initialize ViPE pipeline for this worker (with cached models)
    try:
        vipe_pipeline, model_cache = create_vipe_pipeline(
            pipeline_name=vipe_config.get('pipeline', 'default'),
            output_dir=output_dir,
            save_viz=vipe_config.get('vis_flag', False),
            vis_path=vipe_config.get('vis_path'),
            fixed_fov_degrees=vipe_config.get('fixed_fov_degrees'),
            fov_min_degrees=vipe_config.get('fov_min_degrees'),
            fov_max_degrees=vipe_config.get('fov_max_degrees'),
            optimize_intrinsics=vipe_config.get('optimize_intrinsics', True),
        )
        progress_queue.put(('init', worker_id, None))
    except Exception as e:
        progress_queue.put(('error', worker_id, f"Pipeline init failed: {e}"))
        return
    
    # Process each video in this worker's shard
    for video_dir in video_dirs:
        video_name = video_dir.name
        try:
            result = process_frame_directory_with_vipe(
                vipe_pipeline, video_dir, output_dir, vipe_config
            )
            if result is None:
                progress_queue.put(('fail', worker_id, f"{video_name}: no frames"))
            elif result.get('skipped'):
                # Skipped due to flat depth
                pct_flat = result.get('pct_flat_frames', 0)
                progress_queue.put(('skip', worker_id, f"{video_name} ({pct_flat:.1f}% flat)"))
            else:
                progress_queue.put(('done', worker_id, video_name))
        except Exception as e:
            progress_queue.put(('fail', worker_id, f"{video_name}: {e}\n{traceback.format_exc()}"))
    
    # Signal worker is finished
    progress_queue.put(('finished', worker_id, None))


def main():
    """Main function to run ViPE on all videos with parallel workers."""
    print("=" * 60)
    print("ViPE Processing Pipeline (Parallel)")
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
    num_workers = vipe_config.get('num_workers', 2)
    flat_std_threshold = vipe_config.get('flat_std_threshold', 0.1)
    max_flat_frame_pct = vipe_config.get('max_flat_frame_pct', 20.0)
    
    print(f"Device: {device}")
    print(f"Workers: {num_workers}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: TAPIP3D-ready .npz")
    print(f"Flat depth filter: skip if >{max_flat_frame_pct}% of frames have std<{flat_std_threshold}")
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
    
    # Filter out already-processed videos if continuing
    if continue_flag:
        videos_to_process = []
        skipped = 0
        for video_dir in video_dirs:
            expected_npz_path = output_dir / f"{video_dir.name}.npz"
            if expected_npz_path.exists():
                skipped += 1
            else:
                videos_to_process.append(video_dir)
        if skipped > 0:
            print(f"Skipping {skipped} already-processed videos")
        video_dirs = videos_to_process
    
    total_videos = len(video_dirs)
    print(f"Processing {total_videos} video directories with {num_workers} workers")
    
    if total_videos == 0:
        print("No videos to process!")
        return
    
    # Convert vipe_config to plain dict (OmegaConf objects can't be pickled for multiprocessing)
    vipe_config_dict = OmegaConf.to_container(vipe_config, resolve=True)
    
    # Split videos into shards for workers (round-robin distribution for balance)
    shards = [[] for _ in range(num_workers)]
    for i, video_dir in enumerate(video_dirs):
        shards[i % num_workers].append(video_dir)
    
    # Use spawn context for CUDA safety
    ctx = mp.get_context('spawn')
    progress_queue = ctx.Queue()
    
    # Start worker processes
    print(f"\nSpawning {num_workers} worker processes...")
    workers = []
    for worker_id in range(num_workers):
        if len(shards[worker_id]) == 0:
            continue  # Skip empty shards
        p = ctx.Process(
            target=_worker_process,
            args=(worker_id, shards[worker_id], output_dir, vipe_config_dict, progress_queue)
        )
        p.start()
        workers.append((worker_id, p))
    
    # Track progress
    processed = []
    skipped = []  # Skipped due to flat depth
    failed = []
    workers_initialized = 0
    workers_finished = 0
    
    # Progress bar
    pbar = tqdm(total=total_videos, desc="Videos", position=0, leave=True, disable=False)
    
    # Collect results from workers
    while workers_finished < len(workers):
        msg = progress_queue.get()
        msg_type, worker_id, data = msg
        
        if msg_type == 'init':
            workers_initialized += 1
            tqdm.write(f"  Worker {worker_id} initialized ({workers_initialized}/{len(workers)})")
        elif msg_type == 'done':
            processed.append(data)
            pbar.update(1)
            pbar.set_postfix_str(f"W{worker_id}: {data}")
        elif msg_type == 'skip':
            skipped.append(data)
            pbar.update(1)
            tqdm.write(f"  Worker {worker_id} SKIP (flat depth): {data}")
        elif msg_type == 'fail':
            failed.append(data)
            pbar.update(1)
            tqdm.write(f"  Worker {worker_id} FAIL: {data}")
        elif msg_type == 'error':
            tqdm.write(f"  Worker {worker_id} ERROR: {data}")
            workers_finished += 1
        elif msg_type == 'finished':
            workers_finished += 1
            tqdm.write(f"  Worker {worker_id} finished")
    
    pbar.close()
    
    # Wait for all workers to exit
    for worker_id, p in workers:
        p.join()
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"  Saved: {len(processed)}")
    print(f"  Skipped (flat depth): {len(skipped)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output format: TAPIP3D-ready .npz files")
    if skipped:
        print(f"\n  Skipped videos (>{max_flat_frame_pct}% frames with std<{flat_std_threshold}):")
        for s in skipped[:10]:
            print(f"    - {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
    if failed:
        print(f"\n  Failed videos: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    if vis_flag and vis_path:
        print(f"\n  Interactive 3D Visualization:")
        print(f"    Run: vipe visualize {vis_path} --port 20540")
        print(f"    Open: http://localhost:20540 in browser")
    print("\n  To run TAPIP3D inference:")
    print(f"    python thirdparty/TAPIP3D/inference.py --input_path {output_dir}/<video>.npz --checkpoint thirdparty/TAPIP3D/checkpoints/tapip3d_final.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
