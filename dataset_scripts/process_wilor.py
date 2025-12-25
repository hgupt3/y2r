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
import torch
from omegaconf import OmegaConf
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import json
import struct
import zlib

# Add thirdparty to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "thirdparty" / "WiLoR"))

# Set PyOpenGL platform before importing pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file using OmegaConf"""
    return OmegaConf.load(config_path)


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


def slerp_rotation(R0, R1, t):
    """
    Spherical linear interpolation between two rotation matrices.
    Uses quaternion SLERP for smooth rotation interpolation.
    """
    from scipy.spatial.transform import Rotation, Slerp
    
    rotations = Rotation.from_matrix(np.stack([R0, R1]))
    slerp = Slerp([0, 1], rotations)
    return slerp(t).as_matrix()


def interpolate_hand_data(result, vis_data, side, max_gap, num_frames):
    """
    Interpolate missing frames for a hand side if gaps are within max_gap.
    
    Args:
        result: The result dict with hand data
        vis_data: Visualization data dict (or None)
        side: 'left' or 'right'
        max_gap: Maximum gap to interpolate over
        num_frames: Total number of frames
    
    Returns:
        Number of frames interpolated
    """
    valid = result[f'{side}_valid']
    wrist_poses = result[f'{side}_wrist_pose']
    hand_poses = result[f'{side}_hand_pose']
    joints_3d = result[f'{side}_joints_3d']
    
    interpolated_count = 0
    
    # Find gaps
    i = 0
    while i < num_frames:
        if valid[i]:
            i += 1
            continue
        
        # Found start of a gap - find where it ends
        gap_start = i
        while i < num_frames and not valid[i]:
            i += 1
        gap_end = i  # First valid frame after gap (or num_frames if none)
        
        gap_length = gap_end - gap_start
        
        # Check if we have valid frames on both sides and gap is within limit
        if gap_start > 0 and gap_end < num_frames and gap_length <= max_gap:
            prev_idx = gap_start - 1
            next_idx = gap_end
            
            # Interpolate each frame in the gap
            for j in range(gap_start, gap_end):
                t = (j - prev_idx) / (next_idx - prev_idx)
                
                # Interpolate wrist pose (position linear, rotation SLERP)
                prev_pose = wrist_poses[prev_idx]
                next_pose = wrist_poses[next_idx]
                
                interp_pos = (1 - t) * prev_pose[:3, 3] + t * next_pose[:3, 3]
                interp_rot = slerp_rotation(prev_pose[:3, :3], next_pose[:3, :3], t)
                
                interp_wrist_pose = np.eye(4, dtype=np.float32)
                interp_wrist_pose[:3, :3] = interp_rot
                interp_wrist_pose[:3, 3] = interp_pos
                
                # Interpolate hand pose (linear in rotation matrix space - not perfect but simple)
                interp_hand_pose = (1 - t) * hand_poses[prev_idx] + t * hand_poses[next_idx]
                
                # Interpolate joints
                interp_joints = (1 - t) * joints_3d[prev_idx] + t * joints_3d[next_idx]
                
                # Store interpolated values
                result[f'{side}_valid'][j] = True
                result[f'{side}_wrist_pose'][j] = interp_wrist_pose
                result[f'{side}_hand_pose'][j] = interp_hand_pose
                result[f'{side}_joints_3d'][j] = interp_joints
                
                # Interpolate vis_data if available
                if vis_data is not None and vis_data[prev_idx][side] is not None and vis_data[next_idx][side] is not None:
                    prev_verts = vis_data[prev_idx][side]['verts_repositioned']
                    next_verts = vis_data[next_idx][side]['verts_repositioned']
                    interp_verts = (1 - t) * prev_verts + t * next_verts
                    
                    vis_data[j][side] = {
                        'verts_repositioned': interp_verts,
                        'wrist_pos': interp_pos,
                        'wrist_rot': interp_rot,
                        'is_right': vis_data[prev_idx][side]['is_right'],
                    }
                
                interpolated_count += 1
    
    return interpolated_count


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


def create_3d_visualization(
    video_name: str,
    frames: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    vis_data: dict,
    mano_faces: np.ndarray,
    vis_path: Path,
    vis_fps: int = 12,
    vis_width: int = 256,
    vis_height: int = 192,
):
    """
    Create 3D visualization data for WebGL viewer.
    Uses same format as TAPIP3D for compatibility.
    """
    num_frames = len(frames)
    H_orig, W_orig = frames.shape[1:3]
    num_verts = 778  # MANO vertex count
    fixed_size = (vis_width, vis_height)
    
    # Resize frames and depths
    rgb_resized = np.zeros((num_frames, vis_height, vis_width, 3), dtype=np.uint8)
    depth_float = np.zeros((num_frames, vis_height, vis_width), dtype=np.float32)
    
    for i in range(num_frames):
        rgb_resized[i] = cv2.resize(frames[i], fixed_size, interpolation=cv2.INTER_AREA)
        depth_float[i] = cv2.resize(depths[i], fixed_size, interpolation=cv2.INTER_NEAREST)
    
    # Scale intrinsics and tile for all frames
    scale_x = vis_width / W_orig
    scale_y = vis_height / H_orig
    intrinsics_scaled = np.tile(intrinsics[np.newaxis, :, :], (num_frames, 1, 1)).astype(np.float32)
    intrinsics_scaled[:, 0, :] *= scale_x
    intrinsics_scaled[:, 1, :] *= scale_y
    
    # Compute FOV
    fx, fy = intrinsics_scaled[0, 0, 0], intrinsics_scaled[0, 1, 1]
    fov_y = 2 * np.arctan(vis_height / (2 * fy)) * (180 / np.pi)
    
    # Process depth - encode as 16-bit in RGB channels (same as TAPIP3D)
    valid_depths = depth_float[depth_float > 0]
    min_depth = float(valid_depths.min()) * 0.8 if len(valid_depths) > 0 else 0.1
    max_depth = float(valid_depths.max()) * 1.5 if len(valid_depths) > 0 else 10.0
    
    depth_normalized = np.clip((depth_float - min_depth) / (max_depth - min_depth), 0, 1)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((num_frames, vis_height, vis_width, 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    # Identity extrinsics (camera at origin looking down -Z)
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, :, :], (num_frames, 1, 1))
    inv_extrinsics = extrinsics.copy()
    
    # Prepare hand vertices as "trajectories" - shape (T, N, 3) where N = left + right verts
    # We'll pack: left_verts (778) + right_verts (778) = 1556 points per frame
    left_verts = np.zeros((num_frames, num_verts, 3), dtype=np.float32)
    right_verts = np.zeros((num_frames, num_verts, 3), dtype=np.float32)
    left_valid = np.zeros(num_frames, dtype=np.uint8)
    right_valid = np.zeros(num_frames, dtype=np.uint8)
    
    for frame_idx in range(num_frames):
        if vis_data[frame_idx]['left'] is not None:
            left_valid[frame_idx] = 1
            left_verts[frame_idx] = vis_data[frame_idx]['left']['verts_repositioned']
        
        if vis_data[frame_idx]['right'] is not None:
            right_valid[frame_idx] = 1
            right_verts[frame_idx] = vis_data[frame_idx]['right']['verts_repositioned']
    
    # Build arrays dict
    arrays = {
        "rgb_video": rgb_resized,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics_scaled,
        "extrinsics": extrinsics,
        "inv_extrinsics": inv_extrinsics,
        "left_verts": left_verts,
        "right_verts": right_verts,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "faces": mano_faces.astype(np.int32),
        "cameraZ": np.float64(0.0),
    }
    
    header = {}
    blob_parts = []
    offset = 0
    
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    compressed_blob = zlib.compress(b"".join(blob_parts), level=9)
    
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": num_frames,
        "resolution": list(fixed_size),
        "baseFrameRate": vis_fps,
        "fov": float(fov_y),
        "numVerts": num_verts,
    }
    
    # Write file
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    
    output_file = vis_path / f"{video_name}_data.bin"
    with open(output_file, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(compressed_blob)
    
    return output_file


def create_wilor_viz_html(vis_path: Path, video_list: list):
    """
    Create HTML files for WiLoR 3D visualization.
    
    Args:
        vis_path: Output directory
        video_list: List of video names
    """
    # Copy the template
    template_path = SCRIPT_DIR / 'wilor_viz.html'
    
    if not template_path.exists():
        print(f"Warning: Template {template_path} not found")
        return
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Create individual video HTML files
    for i, video_name in enumerate(video_list):
        html = template.replace("fetch('data.bin')", f"fetch('{video_name}_data.bin')")
        
        # Add navigation
        prev_link = f'<a href="{video_list[i-1]}.html">← Prev</a>' if i > 0 else '<span style="color:#666">← Prev</span>'
        next_link = f'<a href="{video_list[i+1]}.html">Next →</a>' if i < len(video_list) - 1 else '<span style="color:#666">Next →</span>'
        
        nav_html = f'''<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);background:rgba(22,33,62,0.95);padding:8px 16px;border-radius:8px;display:flex;gap:16px;align-items:center;font-size:14px;z-index:100;border:1px solid #333;">
    <a href="index.html" style="color:#e67e22;">Index</a>
    <span style="color:#666;">|</span>
    {prev_link}
    <span style="font-weight:600;">{video_name}</span>
    <span style="color:#888;">({i+1}/{len(video_list)})</span>
    {next_link}
</div>'''
        
        html = html.replace('<div id="frame-info">', nav_html + '\n  <div id="frame-info">')
        
        with open(vis_path / f'{video_name}.html', 'w') as f:
            f.write(html)
    
    # Create index page
    index_html = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>WiLoR 3D Visualization</title>
<style>
body { margin: 0; padding: 20px; font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; }
h1 { color: #e67e22; margin-bottom: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
.card { background: #16213e; border-radius: 8px; padding: 16px; text-decoration: none; color: #eee; border: 1px solid #333; transition: all 0.2s; }
.card:hover { background: #1f3460; border-color: #e67e22; transform: translateY(-2px); }
.card h3 { margin: 0 0 8px 0; color: #e67e22; font-size: 14px; }
.card p { margin: 0; font-size: 12px; color: #888; }
</style></head><body>
<h1>🖐️ WiLoR Hand Pose Visualization</h1>
<div class="grid">
'''
    
    for i, v in enumerate(video_list):
        index_html += f'<a href="{v}.html" class="card"><h3>{v}</h3><p>Video {i+1}/{len(video_list)}</p></a>\n'
    
    index_html += '</div></body></html>'
    
    with open(vis_path / 'index.html', 'w') as f:
        f.write(index_html)


def process_video(
    video_name,
    frames_dir,
    vipe_file,
    output_dir,
    model,
    model_cfg,
    detector,
    device,
    config,
    mano_faces=None,
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
    
    # ========== PHASE 4: Process outputs and compute 3D positions (BATCHED) ==========
    print(f"  Computing 3D positions with depth anchoring...")
    
    left_betas_list = []
    right_betas_list = []
    
    # For visualization - store per-frame data
    vis_data = {i: {'left': None, 'right': None} for i in range(num_frames)} if vis_flag else None
    
    num_dets = len(all_wilor_outputs)
    if num_dets == 0:
        return result
    
    # ===== Batch cam_crop_to_full for all detections at once =====
    pred_cam_all = np.stack([out['pred_cam'] for out in all_wilor_outputs])
    box_center_all = np.stack([out['box_center'] for out in all_wilor_outputs])
    box_size_all = np.stack([out['box_size'] for out in all_wilor_outputs])
    img_size_all = np.stack([out['img_size'] for out in all_wilor_outputs])
    is_right_all = np.array([out['right'] for out in all_wilor_outputs])
    
    # Adjust pred_cam for handedness
    multiplier_all = (2 * is_right_all - 1)
    pred_cam_adjusted_all = pred_cam_all.copy()
    pred_cam_adjusted_all[:, 1] = multiplier_all * pred_cam_adjusted_all[:, 1]
    
    # Compute scaled focal lengths
    scaled_focal_lengths = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size_all.max(axis=1)
    
    # Batch call to cam_crop_to_full
    pred_cam_t_full_all = cam_crop_to_full(
        torch.from_numpy(pred_cam_adjusted_all).float(),
        torch.from_numpy(box_center_all).float(),
        torch.from_numpy(box_size_all).float(),
        torch.from_numpy(img_size_all).float(),
        torch.from_numpy(scaled_focal_lengths).float(),
    ).numpy()
    
    # ===== Process each detection (fast now that cam_crop_to_full is done) =====
    for det_idx, ((frame_idx, side, is_right), wilor_out) in enumerate(zip(all_metadata, all_wilor_outputs)):
        depth_map = depths[frame_idx]
        
        global_orient = wilor_out['global_orient']
        hand_pose = wilor_out['hand_pose']
        betas = wilor_out['betas']
        pred_keypoints_3d = wilor_out['pred_keypoints_3d']
        verts = wilor_out['pred_vertices']
        is_right_val = wilor_out['right']
        
        pred_cam_t_full = pred_cam_t_full_all[det_idx]
        scaled_focal_length = scaled_focal_lengths[det_idx]
        
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
    
    # ========== PHASE 4.5: Velocity-based filtering ==========
    max_hand_velocity = config.get('max_hand_velocity', None)
    fps = config.get('fps', 10)
    
    if max_hand_velocity is not None and max_hand_velocity > 0:
        dt = 1.0 / fps
        max_dist_per_frame = max_hand_velocity * dt
        
        left_filtered = 0
        right_filtered = 0
        
        for side in ['left', 'right']:
            wrist_poses = result[f'{side}_wrist_pose']
            
            # Track last valid frame for velocity comparison
            last_valid_idx = None
            last_valid_pos = None
            
            for frame_idx in range(num_frames):
                if not result[f'{side}_valid'][frame_idx]:
                    continue
                
                curr_pos = wrist_poses[frame_idx, :3, 3]
                
                if last_valid_idx is not None:
                    # Compute distance from last valid frame
                    dist = np.linalg.norm(curr_pos - last_valid_pos)
                    
                    # Compute allowed distance based on frame gap
                    frame_gap = frame_idx - last_valid_idx
                    max_allowed_dist = max_dist_per_frame * frame_gap
                    
                    if dist > max_allowed_dist:
                        # Mark current frame as invalid (the one that "jumped")
                        result[f'{side}_valid'][frame_idx] = False
                        result[f'{side}_wrist_pose'][frame_idx] = np.full((4, 4), np.nan, dtype=np.float32)
                        result[f'{side}_hand_pose'][frame_idx] = np.full((15, 3, 3), np.nan, dtype=np.float32)
                        result[f'{side}_joints_3d'][frame_idx] = np.full((21, 3), np.nan, dtype=np.float32)
                        
                        # Also invalidate vis_data if we're visualizing
                        if vis_flag and vis_data is not None:
                            vis_data[frame_idx][side] = None
                        
                        if side == 'left':
                            left_filtered += 1
                        else:
                            right_filtered += 1
                        
                        # Don't update last_valid - keep comparing against last known good frame
                        continue
                
                # This frame is valid, update tracking
                last_valid_idx = frame_idx
                last_valid_pos = curr_pos
        
        if left_filtered > 0 or right_filtered > 0:
            print(f"  Velocity filter (>{max_hand_velocity:.1f} m/s): removed {left_filtered} left, {right_filtered} right frames")
    
    # ========== PHASE 4.6: Interpolate gaps ==========
    max_interpolation_gap = config.get('max_interpolation_gap', 0)
    
    if max_interpolation_gap > 0:
        left_interp = interpolate_hand_data(result, vis_data, 'left', max_interpolation_gap, num_frames)
        right_interp = interpolate_hand_data(result, vis_data, 'right', max_interpolation_gap, num_frames)
        
        if left_interp > 0 or right_interp > 0:
            print(f"  Interpolated gaps (max {max_interpolation_gap}): {left_interp} left, {right_interp} right frames")
    
    # ========== PHASE 5: Generate 3D visualization ==========
    if vis_flag and vis_path:
        print(f"  Generating 3D visualization...")
        create_3d_visualization(
            video_name=video_name,
            frames=frames,
            depths=depths,
            intrinsics=intrinsics,
            vis_data=vis_data,
            mano_faces=mano_faces,
            vis_path=vis_path,
            vis_fps=vis_fps,
        )
    
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
    
    # Get MANO faces for visualization
    mano_faces = model.mano.faces.copy()
    print(f"✅ MANO faces loaded ({len(mano_faces)} triangles)")
    print(f"{'='*60}\n")
    
    # Process videos
    total_videos_processed = 0
    skipped_videos = 0
    start_time = time.time()
    videos_to_process = []
    processed_video_names = []
    
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
                processed_video_names.append(video_folder.name)  # Include for HTML index
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
            device=device,
            config=wilor_config,
            mano_faces=mano_faces,
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
        
        processed_video_names.append(video_name)
        
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
            print(f"  3D Visualization: {vis_path / f'{video_name}_data.bin'}")
        
        total_videos_processed += 1
        
        # Clear GPU cache
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed_time = time.time() - start_time
    
    # Generate HTML index for visualizations
    if vis_flag and vis_path and processed_video_names:
        print(f"\nGenerating HTML viewer...")
        create_wilor_viz_html(vis_path, processed_video_names)
    
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
        print(f"\n{'='*60}")
        print("TO VIEW 3D VISUALIZATION:")
        print(f"{'='*60}")
        print(f"  cd {vis_path}")
        print(f"  python -m http.server 8000")
        print(f"  Then open: http://localhost:8000/index.html")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

