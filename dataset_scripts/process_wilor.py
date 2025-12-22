"""
WiLoR Hand Pose Processing Script

Extracts hand poses from video frames using WiLoR, anchors wrist position 
using ViPE depth maps, and stores results in a clean array format.

Uses Left/Right classification for hand tracking across frames.

Usage:
    python process_wilor.py
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
import imageio

# Add thirdparty to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "WiLoR"))

# Set PyOpenGL platform before importing pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_frames_from_directory(frames_dir: Path) -> np.ndarray:
    """Load PNG frames into (T, H, W, 3) uint8 array (RGB)."""
    frame_files = sorted(frames_dir.glob("*.png"))
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0).astype(np.uint8)


def load_vipe_data(npz_path: Path):
    """
    Load ViPE output data from .npz file.
    
    Returns:
        depths: (T, H, W) float32 depth maps
        intrinsics: (3, 3) float32 camera matrix (use first frame's intrinsics)
    """
    data = np.load(npz_path)
    depths = data['depths'].astype(np.float32)
    intrinsics = data['intrinsics'][0]  # Use first frame's intrinsics (3, 3)
    return depths, intrinsics


def backproject_point(u, v, depth, intrinsics):
    """
    Backproject a 2D point to 3D using depth and intrinsics.
    
    Args:
        u, v: pixel coordinates
        depth: depth value in meters
        intrinsics: (3, 3) camera intrinsics matrix
    
    Returns:
        (3,) 3D point in camera frame
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z], dtype=np.float32)


def project_points_to_2d(points_3d, intrinsics):
    """
    Project 3D points to 2D using intrinsics.
    
    Args:
        points_3d: (N, 3) 3D points in camera frame
        intrinsics: (3, 3) camera intrinsics matrix
    
    Returns:
        (N, 2) 2D points in pixel coordinates
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    
    return np.stack([u, v], axis=-1)


def get_best_detection(detections, is_right_target):
    """
    Get the best detection for a given hand side (left or right).
    
    Args:
        detections: YOLO detection results
        is_right_target: True for right hand, False for left hand
    
    Returns:
        Best detection dict or None if no matching detection
    """
    best_det = None
    best_conf = -1
    
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        conf = det.boxes.conf.cpu().detach().squeeze().item()
        is_right = det.boxes.cls.cpu().detach().squeeze().item()
        
        # Check if this detection matches the target side
        if bool(is_right) == is_right_target:
            if conf > best_conf:
                best_conf = conf
                best_det = {
                    'bbox': bbox[:4],
                    'confidence': conf,
                    'is_right': is_right
                }
    
    return best_det


def draw_wrist_pose(image, wrist_pos_3d, wrist_rot, intrinsics, axis_length=0.05):
    """
    Draw wrist position and orientation axes on image.
    
    Args:
        image: (H, W, 3) image to draw on (RGB)
        wrist_pos_3d: (3,) wrist position in camera frame
        wrist_rot: (3, 3) rotation matrix
        intrinsics: (3, 3) camera intrinsics
        axis_length: length of axis arrows in meters
    """
    # Project wrist point to 2D
    wrist_2d = project_points_to_2d(wrist_pos_3d.reshape(1, 3), intrinsics)[0]
    wrist_pt = tuple(wrist_2d.astype(int))
    
    # Draw wrist point - white filled circle with black outline
    cv2.circle(image, wrist_pt, 8, (0, 0, 0), -1)  # Black outline
    cv2.circle(image, wrist_pt, 6, (255, 255, 255), -1)  # White fill
    
    # Draw coordinate axes
    # X axis = Red, Y axis = Green, Z axis = Blue
    axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB
    
    for i, color in enumerate(axis_colors):
        # Get axis direction in camera frame
        axis_dir = wrist_rot[:, i]  # Column i of rotation matrix
        
        # End point of axis in 3D
        axis_end_3d = wrist_pos_3d + axis_length * axis_dir
        
        # Project to 2D
        axis_end_2d = project_points_to_2d(axis_end_3d.reshape(1, 3), intrinsics)[0]
        axis_end_pt = tuple(axis_end_2d.astype(int))
        
        # Draw axis line with arrow
        cv2.arrowedLine(image, wrist_pt, axis_end_pt, color, 2, tipLength=0.3)


def process_video(
    video_name,
    frames_dir,
    vipe_file,
    output_dir,
    model,
    model_cfg,
    detector,
    renderer,
    device,
    config,
    vis_flag=False,
    vis_path=None,
    vis_fps=12,
):
    """
    Process a single video to extract hand poses.
    Uses batched inference for efficiency.
    
    Returns:
        dict with results or None if failed
    """
    from wilor.utils import recursive_to
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils.renderer import cam_crop_to_full
    
    # Load frames and ViPE data
    frames = load_frames_from_directory(frames_dir)  # (T, H, W, 3) RGB
    depths, intrinsics = load_vipe_data(vipe_file)
    
    num_frames = len(frames)
    H, W = frames.shape[1], frames.shape[2]
    
    # Initialize output arrays
    result = {
        'video_name': video_name,
        'num_frames': num_frames,
        'intrinsics': intrinsics,
        
        # Left hand
        'left_valid': np.zeros(num_frames, dtype=bool),
        'left_wrist_pose': np.full((num_frames, 4, 4), np.nan, dtype=np.float32),
        'left_hand_pose': np.full((num_frames, 15, 3, 3), np.nan, dtype=np.float32),
        'left_joints_3d': np.full((num_frames, 21, 3), np.nan, dtype=np.float32),
        
        # Right hand
        'right_valid': np.zeros(num_frames, dtype=bool),
        'right_wrist_pose': np.full((num_frames, 4, 4), np.nan, dtype=np.float32),
        'right_hand_pose': np.full((num_frames, 15, 3, 3), np.nan, dtype=np.float32),
        'right_joints_3d': np.full((num_frames, 21, 3), np.nan, dtype=np.float32),
    }
    
    detection_confidence = config.get('detection_confidence', 0.3)
    wrist_depth_offset = config.get('wrist_depth_offset', 0.02)
    batch_size = config.get('batch_size', 256)
    rescale_factor = 2.0  # Default WiLoR value - don't change
    
    # Convert frames to BGR for YOLO (as a list, not stacked array)
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    
    # ========== PHASE 1: Batch YOLO detection ==========
    print(f"  Running YOLO detection on {num_frames} frames...")
    all_detections = detector(frames_bgr, conf=detection_confidence, verbose=False)
    
    # Collect per-frame detections (best left and right for each frame)
    frame_detections = []  # List of (frame_idx, side, bbox, is_right)
    for frame_idx, dets in enumerate(all_detections):
        left_det = get_best_detection(dets, is_right_target=False)
        right_det = get_best_detection(dets, is_right_target=True)
        
        if left_det is not None:
            frame_detections.append((frame_idx, 'left', left_det['bbox'], left_det['is_right']))
        if right_det is not None:
            frame_detections.append((frame_idx, 'right', right_det['bbox'], right_det['is_right']))
    
    if len(frame_detections) == 0:
        print(f"  No hands detected in any frame")
        return result
    
    print(f"  Found {len(frame_detections)} hand detections across {num_frames} frames")
    
    # ========== PHASE 2: Prepare all crops ==========
    # Create dataset items for all detections
    all_batch_items = []
    all_metadata = []  # (frame_idx, side, is_right)
    
    for frame_idx, side, bbox, is_right in frame_detections:
        frame_bgr = frames_bgr[frame_idx]
        boxes = np.array([bbox])
        right_arr = np.array([is_right])
        
        dataset = ViTDetDataset(
            model_cfg, 
            frame_bgr, 
            boxes, 
            right_arr, 
            rescale_factor=rescale_factor
        )
        
        # Get the single item from this dataset
        item = dataset[0]
        all_batch_items.append(item)
        all_metadata.append((frame_idx, side, is_right))
    
    # ========== PHASE 3: Batch WiLoR inference ==========
    print(f"  Running WiLoR inference in batches of {batch_size}...")
    
    # Storage for WiLoR outputs
    all_wilor_outputs = []
    
    num_detections = len(all_batch_items)
    for batch_start in tqdm(range(0, num_detections, batch_size), desc=f"  {video_name}", leave=False):
        batch_end = min(batch_start + batch_size, num_detections)
        batch_items = all_batch_items[batch_start:batch_end]
        
        # Collate batch items
        batch = {}
        for key in batch_items[0].keys():
            if isinstance(batch_items[0][key], np.ndarray):
                batch[key] = torch.from_numpy(np.stack([item[key] for item in batch_items]))
            else:
                batch[key] = torch.tensor([item[key] for item in batch_items])
        
        batch = recursive_to(batch, device)
        
        with torch.no_grad():
            out = model(batch)
        
        # Store outputs for this batch
        batch_size_actual = len(batch_items)
        for i in range(batch_size_actual):
            all_wilor_outputs.append({
                'global_orient': out['pred_mano_params']['global_orient'][i].cpu().numpy(),
                'hand_pose': out['pred_mano_params']['hand_pose'][i].cpu().numpy(),
                'betas': out['pred_mano_params']['betas'][i].cpu().numpy(),
                'pred_keypoints_3d': out['pred_keypoints_3d'][i].cpu().numpy(),
                'pred_vertices': out['pred_vertices'][i].cpu().numpy(),
                'pred_cam': out['pred_cam'][i].cpu().numpy(),
                'box_center': batch['box_center'][i].cpu().numpy(),
                'box_size': batch['box_size'][i].cpu().numpy(),
                'img_size': batch['img_size'][i].cpu().numpy(),
                'right': batch['right'][i].cpu().numpy(),
            })
    
    # ========== PHASE 4: Process outputs and compute 3D positions ==========
    print(f"  Computing 3D positions with depth anchoring...")
    
    left_betas_list = []
    right_betas_list = []
    
    # For visualization - store per-frame data
    vis_data = {i: {'left': None, 'right': None} for i in range(num_frames)} if vis_flag else None
    
    for det_idx, ((frame_idx, side, is_right), wilor_out) in enumerate(zip(all_metadata, all_wilor_outputs)):
        depth_map = depths[frame_idx]
        
        global_orient = wilor_out['global_orient']
        hand_pose = wilor_out['hand_pose']
        betas = wilor_out['betas']
        pred_keypoints_3d = wilor_out['pred_keypoints_3d']
        verts = wilor_out['pred_vertices']
        pred_cam = wilor_out['pred_cam']
        box_center = wilor_out['box_center']
        box_size = wilor_out['box_size']
        img_size = wilor_out['img_size']
        is_right_val = wilor_out['right']
        
        # Compute camera translation
        multiplier = (2 * is_right_val - 1)
        pred_cam_adjusted = pred_cam.copy()
        pred_cam_adjusted[1] = multiplier * pred_cam_adjusted[1]
        
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        
        # cam_crop_to_full expects tensors
        pred_cam_t = torch.tensor(pred_cam_adjusted).unsqueeze(0)
        box_center_t = torch.tensor(box_center).unsqueeze(0)
        box_size_t = torch.tensor(box_size).unsqueeze(0)
        img_size_t = torch.tensor(img_size).unsqueeze(0)
        
        pred_cam_t_full = cam_crop_to_full(
            pred_cam_t, box_center_t, box_size_t, img_size_t, scaled_focal_length
        ).numpy()[0]
        
        # Mirror vertices/joints for left hand
        joints_3d_wilor = pred_keypoints_3d.copy()
        verts[:, 0] = (2 * is_right_val - 1) * verts[:, 0]
        joints_3d_wilor[:, 0] = (2 * is_right_val - 1) * joints_3d_wilor[:, 0]
        
        # Project WiLoR 3D joints to get 2D keypoints in full image
        joints_3d_cam_wilor = joints_3d_wilor + pred_cam_t_full
        wrist_2d = project_points_to_2d(
            joints_3d_cam_wilor[:1], 
            np.array([[scaled_focal_length, 0, W/2],
                      [0, scaled_focal_length, H/2],
                      [0, 0, 1]])
        )[0]
        
        # Clamp wrist 2D to image bounds
        wrist_u = int(np.clip(wrist_2d[0], 0, W - 1))
        wrist_v = int(np.clip(wrist_2d[1], 0, H - 1))
        
        # Get wrist depth from ViPE + offset
        wrist_depth = depth_map[wrist_v, wrist_u]
        if wrist_depth <= 0 or not np.isfinite(wrist_depth):
            # Try to sample from nearby pixels
            for offset in range(1, 10):
                for dv, du in [(-offset, 0), (offset, 0), (0, -offset), (0, offset)]:
                    nv, nu = wrist_v + dv, wrist_u + du
                    if 0 <= nv < H and 0 <= nu < W:
                        d = depth_map[nv, nu]
                        if d > 0 and np.isfinite(d):
                            wrist_depth = d
                            break
                if wrist_depth > 0:
                    break
        
        if wrist_depth <= 0 or not np.isfinite(wrist_depth):
            # Skip this detection if no valid depth
            continue
        
        wrist_depth += wrist_depth_offset
        
        # Backproject wrist to camera frame
        wrist_pos = backproject_point(wrist_u, wrist_v, wrist_depth, intrinsics)
        
        # Build 4x4 wrist transform
        wrist_rot = global_orient.squeeze()  # (3, 3)
        wrist_pose_4x4 = np.eye(4, dtype=np.float32)
        wrist_pose_4x4[:3, :3] = wrist_rot
        wrist_pose_4x4[:3, 3] = wrist_pos
        
        # Compute joints in camera frame
        joints_local = joints_3d_wilor - joints_3d_wilor[0:1]
        joints_3d_camera = (wrist_rot @ joints_local.T).T + wrist_pos
        
        # Store results
        result[f'{side}_valid'][frame_idx] = True
        result[f'{side}_wrist_pose'][frame_idx] = wrist_pose_4x4
        result[f'{side}_hand_pose'][frame_idx] = hand_pose
        result[f'{side}_joints_3d'][frame_idx] = joints_3d_camera
        
        if side == 'left':
            left_betas_list.append(betas)
        else:
            right_betas_list.append(betas)
        
        # Store data for visualization - reposition to metric depth
        # NOTE: MANO vertices already have global_orient baked in, so we DON'T rotate again
        if vis_flag:
            # Center vertices on wrist joint, then translate to metric wrist position
            wrist_joint_wilor = joints_3d_wilor[0]  # Wrist position in WiLoR MANO space
            verts_centered = verts - wrist_joint_wilor  # Center on wrist
            verts_repositioned = verts_centered + wrist_pos  # Move to metric position
            
            vis_data[frame_idx][side] = {
                'verts_repositioned': verts_repositioned,  # Vertices repositioned to metric depth
                'wrist_pos': wrist_pos,
                'wrist_rot': wrist_rot,
                'is_right': is_right_val,
            }
    
    # Compute median betas
    result['left_betas'] = np.median(np.stack(left_betas_list), axis=0) if left_betas_list else np.zeros(10, dtype=np.float32)
    result['right_betas'] = np.median(np.stack(right_betas_list), axis=0) if right_betas_list else np.zeros(10, dtype=np.float32)
    
    # ========== PHASE 5: Generate visualization ==========
    if vis_flag and vis_path:
        print(f"  Generating visualization...")
        vis_frames = []
        
        # Use ViPE intrinsics for metric visualization
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        
        for frame_idx in range(num_frames):
            vis_frame = frames[frame_idx].copy()
            
            # Collect all hands in this frame for batch rendering
            all_verts = []
            all_cam_t = []
            all_is_right = []
            all_mesh_colors = []
            wrist_data_list = []
            
            for side in ['left', 'right']:
                data = vis_data[frame_idx][side]
                if data is None:
                    continue
                
                mesh_color = (0.2, 0.4, 0.9) if side == 'left' else (0.9, 0.4, 0.2)
                
                # Use vertices repositioned to metric depth (no extra rotation applied)
                all_verts.append(data['verts_repositioned'])
                all_cam_t.append(np.zeros(3))  # Already in camera frame
                all_is_right.append(data['is_right'])
                all_mesh_colors.append(mesh_color)
                wrist_data_list.append({
                    'wrist_pos': data['wrist_pos'],
                    'wrist_rot': data['wrist_rot'],
                })
            
            # Render all hands in this frame (metric camera frame)
            if all_verts:
                # Use first mesh color (could blend if needed)
                mesh_overlay = renderer.render_rgba_multiple(
                    all_verts,
                    cam_t=all_cam_t,
                    mesh_base_color=all_mesh_colors[0] if len(all_mesh_colors) == 1 else (0.6, 0.4, 0.5),
                    scene_bg_color=(0, 0, 0),
                    render_res=[W, H],
                    focal_length=fx,  # Use ViPE focal length
                    is_right=all_is_right,
                )
                
                alpha = mesh_overlay[:, :, 3:4]
                vis_frame = (vis_frame / 255.0 * (1 - alpha) + mesh_overlay[:, :, :3] * alpha)
                vis_frame = (vis_frame * 255).astype(np.uint8)
                
                # Draw wrist poses using ViPE intrinsics
                for wrist_data in wrist_data_list:
                    draw_wrist_pose(vis_frame, wrist_data['wrist_pos'], wrist_data['wrist_rot'], 
                                   intrinsics, axis_length=0.05)
            
            vis_frames.append(vis_frame)
        
        vis_path.mkdir(parents=True, exist_ok=True)
        video_path = vis_path / f"{video_name}.mp4"
        writer = imageio.get_writer(video_path, fps=vis_fps)
        for frame in vis_frames:
            writer.append_data(frame)
        writer.close()
    
    return result


def main():
    # Load configuration
    config = load_config("config.yaml")
    wilor_config = config['wilor']
    
    # Extract configuration
    input_frames_dir = Path(wilor_config['input_frames_dir'])
    input_vipe_dir = Path(wilor_config['input_vipe_dir'])
    output_dir = Path(wilor_config['output_dir'])
    device = wilor_config['device']
    num_videos_to_process = wilor_config['num_videos_to_process']
    vis_flag = wilor_config['vis_flag']
    vis_path = Path(wilor_config['vis_path']) if vis_flag else None
    vis_fps = wilor_config['vis_fps']
    continue_mode = wilor_config.get('continue', False)
    
    # Delete previous output directories if not continuing
    if not continue_mode:
        if output_dir.exists():
            print(f"\n🗑️  Deleting previous output directory: {output_dir}")
            shutil.rmtree(output_dir)
            print(f"✓ Previous output deleted")
        if vis_flag and vis_path and vis_path.exists():
            print(f"🗑️  Deleting previous visualizations directory: {vis_path}")
            shutil.rmtree(vis_path)
            print(f"✓ Previous visualizations deleted\n")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if vis_flag:
        vis_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video folders
    video_folders = sorted([d for d in input_frames_dir.iterdir() if d.is_dir()])
    
    if num_videos_to_process is not None:
        video_folders = video_folders[:num_videos_to_process]
    
    print(f"\n{'='*60}")
    print(f"WILOR HAND POSE PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input frames directory: {input_frames_dir}")
    print(f"Input ViPE directory: {input_vipe_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Videos to process: {len(video_folders)}")
    print(f"Visualization: {'Enabled' if vis_flag else 'Disabled'}")
    if vis_flag:
        print(f"Visualization path: {vis_path}")
    print(f"{'='*60}\n")
    
    # Load models
    print(f"{'='*60}")
    print("LOADING MODELS (one-time initialization)")
    print(f"{'='*60}")
    
    print("Loading WiLoR model...")
    from wilor.models import load_wilor
    model, model_cfg = load_wilor(
        checkpoint_path=str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/wilor_final.ckpt"),
        cfg_path=str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/model_config.yaml")
    )
    model = model.to(device)
    model.eval()
    print("✅ WiLoR loaded")
    
    print("Loading YOLO hand detector...")
    from ultralytics import YOLO
    detector = YOLO(str(PROJECT_ROOT / "thirdparty/WiLoR/pretrained_models/detector.pt"))
    detector = detector.to(device)
    print("✅ YOLO detector loaded")
    
    print("Setting up renderer...")
    from wilor.utils.renderer import Renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    print("✅ Renderer setup complete")
    print(f"{'='*60}\n")
    
    # Process videos
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    videos_to_process = []
    
    # Check for existing outputs if continuing
    if continue_mode and output_dir.exists():
        print(f"\n{'='*60}")
        print(f"CHECKING EXISTING OUTPUTS FOR COMPLETENESS")
        print(f"{'='*60}\n")
        
        for idx, video_folder in enumerate(video_folders):
            output_file = output_dir / f"{video_folder.name}.pt"
            
            if output_file.exists():
                print(f"✓ Output file {video_folder.name}.pt exists")
                skipped_videos += 1
            else:
                videos_to_process.append(idx)
        
        if videos_to_process:
            print(f"\n🔄 Found {len(videos_to_process)} videos to process")
            print(f"⏭️  Skipping {skipped_videos} already complete videos\n")
        else:
            print(f"\n✅ All {skipped_videos} videos are already complete!\n")
    else:
        videos_to_process = list(range(len(video_folders)))
    
    # Process each video
    for idx in videos_to_process:
        video_folder = video_folders[idx]
        video_name = video_folder.name
        
        print(f"\n{'='*60}")
        print(f"Processing video {idx + 1}/{len(video_folders)}: {video_name}")
        print(f"{'='*60}")
        
        video_start_time = time.time()
        
        # Check for ViPE data
        vipe_file = input_vipe_dir / f"{video_name}.npz"
        if not vipe_file.exists():
            print(f"⚠ ViPE file not found: {vipe_file}")
            continue
        
        result = process_video(
            video_name=video_name,
            frames_dir=video_folder,
            vipe_file=vipe_file,
            output_dir=output_dir,
            model=model,
            model_cfg=model_cfg,
            detector=detector,
            renderer=renderer,
            device=device,
            config=wilor_config,
            vis_flag=vis_flag,
            vis_path=vis_path,
            vis_fps=vis_fps,
        )
        
        if result is None:
            print(f"⚠ Failed to process {video_name}")
            continue
        
        # Save results
        output_file = output_dir / f"{video_name}.pt"
        torch.save(result, output_file)
        
        # Print summary
        video_elapsed = time.time() - video_start_time
        left_valid_count = result['left_valid'].sum()
        right_valid_count = result['right_valid'].sum()
        
        print(f"\n✓ Completed in {video_elapsed:.2f}s")
        print(f"  Frames: {result['num_frames']}")
        print(f"  Left hand valid: {left_valid_count}/{result['num_frames']} frames")
        print(f"  Right hand valid: {right_valid_count}/{result['num_frames']} frames")
        print(f"  Saved to: {output_file}")
        if vis_flag:
            print(f"  Visualization: {vis_path / f'{video_name}.mp4'}")
        
        total_videos_processed += 1
        
        # Clear GPU cache
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
    print(f"Output saved to: {output_dir}")
    if vis_flag:
        print(f"Visualizations saved to: {vis_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

