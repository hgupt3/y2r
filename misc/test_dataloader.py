#!/usr/bin/env python3
"""
Test script to verify HDF5 dataset works with the updated dataloader.
Supports both 2D and 3D track formats with visualization.

Usage:
    python test_dataloader.py                    # Basic test
    python test_dataloader.py --visualize        # Test with visualization
    python test_dataloader.py --visualize --num_samples 10
"""
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import random
import yaml
import shutil
import h5py
import json
import zlib
import struct

# Add parent directory to path to import y2r
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.dataloaders.utils import get_dataloader, NormalizationStats


def load_config():
    """Load configuration from dataset_config.yaml"""
    config_path = PROJECT_ROOT / "configs" / "dataset_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def denormalize_sample(sample, norm_stats, track_type):
    """
    Denormalize a sample back to raw values.
    
    Args:
        sample: Dict with normalized data
        norm_stats: NormalizationStats instance
        track_type: '2d' or '3d'
    
    Returns:
        Dict with denormalized data
    """
    result = {}
    
    # Images: denormalize from ImageNet normalization
    imgs = sample['imgs'].clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    result['imgs'] = imgs * std + mean  # Back to [0, 1]
    
    # Query coords
    query_coords = sample['query_coords'].clone()
    if track_type == '3d' and query_coords.shape[-1] == 3:
        # Denormalize depth (3rd dimension)
        query_coords[:, 2] = norm_stats.denormalize_depth(query_coords[:, 2].numpy())
        query_coords[:, 2] = torch.from_numpy(query_coords[:, 2].numpy())
    result['query_coords'] = query_coords
    
    # Displacements
    displacements = sample['displacements'].clone().numpy()
    displacements = norm_stats.denormalize_displacement(displacements)
    result['displacements'] = torch.from_numpy(displacements)
    
    # Poses (3D only)
    if sample.get('poses') is not None:
        poses = sample['poses'].clone().numpy()
        poses = norm_stats.denormalize_pose(poses)
        result['poses'] = torch.from_numpy(poses)
    else:
        result['poses'] = None
    
    # Depth maps (3D only)
    if sample.get('depth') is not None:
        depth = sample['depth'].clone().numpy()
        depth = norm_stats.denormalize_depth(depth)
        result['depth'] = torch.from_numpy(depth)
    else:
        result['depth'] = None
    
    return result


def reconstruct_tracks(query_coords, displacements):
    """
    Reconstruct full trajectory from query_coords and displacements.
    
    Args:
        query_coords: (N, coord_dim) - initial positions
        displacements: (T, N, coord_dim) - motion relative to t=0
    
    Returns:
        tracks: (T, N, coord_dim) - full trajectory
    """
    T, N, coord_dim = displacements.shape
    tracks = query_coords.unsqueeze(0).expand(T, -1, -1) + displacements
    return tracks


def pose_9d_to_4x4(pose_9d):
    """
    Convert 9D pose (6D rotation + 3D translation) back to 4x4 matrix.
    
    Args:
        pose_9d: (9,) array - first 6 are rotation (first 2 columns of R), last 3 are translation
    
    Returns:
        (4, 4) transformation matrix
    """
    # Extract rotation columns and translation
    r1 = pose_9d[:3]  # First column of R
    r2 = pose_9d[3:6]  # Second column of R
    t = pose_9d[6:9]   # Translation
    
    # Reconstruct rotation matrix using Gram-Schmidt
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    
    # Build 4x4 matrix
    pose_4x4 = np.eye(4, dtype=np.float32)
    pose_4x4[:3, 0] = r1
    pose_4x4[:3, 1] = r2
    pose_4x4[:3, 2] = r3
    pose_4x4[:3, 3] = t
    
    return pose_4x4


def poses_9d_to_4x4(poses_9d):
    """
    Convert batch of 9D poses to 4x4 matrices.
    
    Args:
        poses_9d: (T, 9) array
    
    Returns:
        (T, 4, 4) transformation matrices
    """
    T = poses_9d.shape[0]
    poses_4x4 = np.zeros((T, 4, 4), dtype=np.float32)
    for t in range(T):
        poses_4x4[t] = pose_9d_to_4x4(poses_9d[t])
    return poses_4x4


def visualize_sample_2d(sample, sample_idx, output_dir, img_size, norm_stats=None, vis_scale=2):
    """
    Visualize a sample with 2D trajectory overlay.
    
    Args:
        sample: Dict with 'imgs', 'query_coords', 'displacements'
        sample_idx: Sample index for filename
        output_dir: Directory to save visualization
        img_size: Image size
        norm_stats: NormalizationStats for denormalizing displacements
        vis_scale: Scale factor for visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import CoTracker visualizer
    sys.path.insert(0, str(PROJECT_ROOT / "thirdparty"))
    from cotracker3.cotracker.utils.visualizer import Visualizer
    import torch.nn.functional as F
    
    imgs = sample['imgs']
    query_coords = sample['query_coords'].clone()
    displacements = sample['displacements'].clone()
    
    # Denormalize displacements before reconstruction
    if norm_stats is not None:
        disp_np = displacements.numpy()
        disp_np = norm_stats.denormalize_displacement(disp_np)
        displacements = torch.from_numpy(disp_np.astype(np.float32))
    
    # Reconstruct tracks
    tracks = reconstruct_tracks(query_coords, displacements)  # (T, N, coord_dim)
    
    # Get last frame (most recent observation)
    frame = imgs[-1].clone()  # (C, H, W), ImageNet normalized
    
    # Denormalize from ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame = frame * std + mean
    frame = torch.clamp(frame, 0, 1) * 255
    
    # Upscale frame
    vis_size = img_size * vis_scale
    frame_upscaled = F.interpolate(
        frame.unsqueeze(0),
        size=(vis_size, vis_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # Darken for visibility
    frame_darkened = frame_upscaled * 0.5
    
    # Convert tracks to pixel coords (only use u, v)
    tracks_2d = tracks[..., :2].clone()  # (T, N, 2)
    tracks_2d = tracks_2d * vis_size  # Denormalize to pixel space
    
    # Add batch dimension for visualizer
    tracks_batch = tracks_2d.unsqueeze(0)  # (1, T, N, 2)
    
    # Create visualizer
    vis = Visualizer(save_dir=str(output_dir), pad_value=0, linewidth=1)
    
    # Render
    rendered_frame = vis.visualize_trajectory_on_frame(
        frame=frame_darkened,
        tracks=tracks_batch,
        visibility=None,
        segm_mask=None,
        query_frame=0,
        opacity=1.0,
    )
    
    # Save
    output_path = output_dir / f"sample_{sample_idx:03d}_2d.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
    
    return output_path


def visualize_sample_3d(sample, sample_idx, output_dir, intrinsics, img_size, norm_stats=None):
    """
    Create 3D visualization using the same system as process_tapip3d.py.
    
    Args:
        sample: Dict with 'imgs', 'query_coords', 'displacements', 'poses'
        sample_idx: Sample index for filename
        output_dir: Directory to save visualization
        intrinsics: (3, 3) camera intrinsics matrix
        img_size: Image size for coordinate unprojection
        norm_stats: NormalizationStats for denormalizing data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    imgs = sample['imgs']
    query_coords = sample['query_coords'].clone()  # (N, 3) - (u, v, d)
    displacements = sample['displacements'].clone()  # (T, N, 3) - (du, dv, dd)
    
    # Denormalize before reconstruction
    if norm_stats is not None:
        # Denormalize depth in query_coords (3rd dimension)
        query_coords[:, 2] = torch.from_numpy(
            norm_stats.denormalize_depth(query_coords[:, 2].numpy()).astype(np.float32)
        )
        # Denormalize displacements
        disp_np = displacements.numpy()
        disp_np = norm_stats.denormalize_displacement(disp_np)
        displacements = torch.from_numpy(disp_np.astype(np.float32))
    
    # Reconstruct full tracks in (u, v, d) space
    tracks = reconstruct_tracks(query_coords, displacements)  # (T, N, 3)
    T, N, _ = tracks.shape
    
    # Unproject (u, v, d) to 3D camera coords
    u = tracks[..., 0].numpy() * img_size  # pixel x
    v = tracks[..., 1].numpy() * img_size  # pixel y
    d = tracks[..., 2].numpy()  # depth
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Unproject to 3D camera coords
    x_3d = (u - cx) * d / fx
    y_3d = (v - cy) * d / fy
    z_3d = d
    
    points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1)  # (T, N, 3)
    
    # Get RGB frame (denormalize from ImageNet)
    frame = imgs[-1].clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame = (frame * std + mean).clamp(0, 1)
    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Get depth frame if available (use first frame corresponding to query/observation time)
    depth_frame = None
    if sample['depth'] is not None:
        # sample['depth'] is (num_track_ts, H, W) - use first frame (observation time)
        # Depth is normalized by dataloader, so denormalize it
        depth_tensor = sample['depth'][0]
        if norm_stats is not None:
            depth_frame = norm_stats.denormalize_depth(depth_tensor.numpy())
        else:
            depth_frame = depth_tensor.numpy()
    
    # Get camera poses if available
    camera_poses = None
    if sample['poses'] is not None:
        poses_9d = sample['poses'].clone()  # (T, 9)
        # Denormalize poses
        if norm_stats is not None:
            poses_9d = torch.from_numpy(
                norm_stats.denormalize_pose(poses_9d.numpy()).astype(np.float32)
            )
        # Convert 9D to 4x4 matrices
        camera_poses = poses_9d_to_4x4(poses_9d.numpy())  # (T, 4, 4)
    
    # Create visualization using TAPIP3D format
    create_3d_viz_data(
        points_3d, frame_np, intrinsics, img_size,
        output_dir, sample_idx, depth_frame=depth_frame, camera_poses=camera_poses
    )
    
    return output_dir / f"sample_{sample_idx:03d}_3d.html"


def add_navigation_to_3d_html(output_dir, all_sample_indices):
    """Add navigation links to all 3D HTML files after they're created."""
    sorted_indices = sorted(all_sample_indices)
    
    for i, sample_idx in enumerate(sorted_indices):
        html_path = output_dir / f"sample_{sample_idx:03d}_3d.html"
        if not html_path.exists():
            continue
        
        with open(html_path, 'r') as f:
            html = f.read()
        
        # Build prev/next links
        prev_idx = sorted_indices[i - 1] if i > 0 else None
        next_idx = sorted_indices[i + 1] if i < len(sorted_indices) - 1 else None
        
        prev_link = f"<a href='sample_{prev_idx:03d}_3d.html' style='color:#a78bfa;'>← Prev</a>" if prev_idx is not None else "<span style='color:#666;'>← Prev</span>"
        next_link = f"<a href='sample_{next_idx:03d}_3d.html' style='color:#a78bfa;'>Next →</a>" if next_idx is not None else "<span style='color:#666;'>Next →</span>"
        
        nav_html = f'''<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(0,0,0,0.8);padding:8px 12px;border-radius:8px;display:flex;gap:10px;font-family:system-ui;color:#eee;">
<a href="index.html" style="color:#a78bfa;">Index</a>
<span>|</span>
{prev_link}
<span style="font-weight:600;">Sample {sample_idx}</span>
<span style="color:#888;">({i+1}/{len(sorted_indices)})</span>
{next_link}
</div>'''
        
        # Insert navigation after body tag (will appear alongside existing info overlay)
        html = html.replace('<body>', '<body>' + nav_html)
        
        with open(html_path, 'w') as f:
            f.write(html)


def create_3d_viz_data(points_3d, frame_rgb, intrinsics, img_size, output_dir, sample_idx, depth_frame=None, camera_poses=None):
    """
    Create visualization data using the same format as process_tapip3d.py.
    
    Args:
        points_3d: (T, N, 3) 3D points in camera coords
        frame_rgb: (H, W, 3) RGB frame as uint8
        intrinsics: (3, 3) camera intrinsics
        img_size: Image size
        output_dir: Output directory
        sample_idx: Sample index
        depth_frame: (H, W) optional depth frame in meters
        camera_poses: (T, 4, 4) optional camera poses for each timestep
    """
    T, N, _ = points_3d.shape
    H, W = frame_rgb.shape[:2]
    
    # Fixed visualization size
    fixed_size = (256, 192)  # (width, height)
    
    # Resize frame
    rgb_resized = cv2.resize(frame_rgb, fixed_size, interpolation=cv2.INTER_AREA)
    rgb_video = rgb_resized[np.newaxis, ...]  # (1, H, W, 3)
    
    # Process depth frame
    if depth_frame is not None:
        # Resize actual depth
        depth_resized = cv2.resize(depth_frame, fixed_size, interpolation=cv2.INTER_NEAREST)
        valid_depths = depth_resized[depth_resized > 0]
        if len(valid_depths) > 0:
            min_depth = float(valid_depths.min()) * 0.8
            max_depth = float(valid_depths.max()) * 1.5
        else:
            min_depth, max_depth = 0.1, 10.0
    else:
        # Fallback: use trajectory depths
        avg_depth = np.mean(points_3d[..., 2])
        min_depth = max(0.1, float(np.min(points_3d[..., 2])) * 0.8)
        max_depth = float(np.max(points_3d[..., 2])) * 1.5
        depth_resized = np.full((fixed_size[1], fixed_size[0]), avg_depth, dtype=np.float32)
    
    depth_normalized = np.clip((depth_resized - min_depth) / (max_depth - min_depth), 0, 1)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((1, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[0, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[0, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    # Scale intrinsics for visualization size
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics_scaled = intrinsics.copy()
    intrinsics_scaled[0, :] *= scale_x
    intrinsics_scaled[1, :] *= scale_y
    intrinsics_arr = intrinsics_scaled[np.newaxis, ...].astype(np.float32)  # (1, 3, 3)
    
    # Compute FOV
    fx, fy = intrinsics_scaled[0, 0], intrinsics_scaled[1, 1]
    fov_y = 2 * np.arctan(fixed_size[1] / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(fixed_size[0] / (2 * fx)) * (180 / np.pi)
    
    # Identity extrinsics (camera-0 frame is world frame)
    extrinsics = np.eye(4, dtype=np.float32)[np.newaxis, ...]  # (1, 4, 4)
    inv_extrinsics = np.eye(4, dtype=np.float32)[np.newaxis, ...]  # (1, 4, 4)
    
    # Trajectories: (1, T, N, 3) - single "window" with T timesteps
    trajectories = points_3d[np.newaxis, ...].astype(np.float32)  # (1, T, N, 3)
    
    # Build binary data
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics_arr,
        "extrinsics": extrinsics,
        "inv_extrinsics": inv_extrinsics,
        "trajectories": trajectories,
        "cameraZ": np.float64(0.0),
    }
    
    # Add camera poses if available
    has_camera_poses = camera_poses is not None
    if has_camera_poses:
        # camera_poses: (T, 4, 4) -> extract positions and forward directions
        # Position is the translation (4th column)
        # Forward direction is the negative Z axis of the camera
        camera_positions = camera_poses[:, :3, 3].astype(np.float32)  # (T, 3)
        camera_forwards = -camera_poses[:, :3, 2].astype(np.float32)  # (T, 3) - negative Z is forward
        camera_ups = camera_poses[:, :3, 1].astype(np.float32)  # (T, 3) - Y is up
        
        arrays["camera_positions"] = camera_positions
        arrays["camera_forwards"] = camera_forwards
        arrays["camera_ups"] = camera_ups
    
    header = {}
    blob_parts = []
    offset = 0
    
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset, "length": len(arr_bytes)}
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    compressed_blob = zlib.compress(b"".join(blob_parts), level=9)
    
    header["meta"] = {
        "depthRange": [float(min_depth), float(max_depth)],
        "totalFrames": 1,
        "resolution": list(fixed_size),
        "baseFrameRate": 1,
        "numTrajectoryPoints": N,
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "windowLength": T,
        "stride": 1,
        "windowedMode": True,
        "hasCameraPoses": has_camera_poses,
        "numCameraPoses": T if has_camera_poses else 0,
    }
    
    # Write binary data file
    bin_path = output_dir / f"sample_{sample_idx:03d}_data.bin"
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(bin_path, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(compressed_blob)
    
    # Create HTML that uses this data file
    create_sample_viz_html(output_dir, sample_idx, N, T)


def create_sample_viz_html(output_dir, sample_idx, num_points, num_timesteps):
    """
    Create HTML visualization using the TAPIP3D viz.html as template.
    This reuses the exact same viewer as process_tapip3d.py.
    """
    # Path to the TAPIP3D viz.html template
    viz_template = PROJECT_ROOT / "thirdparty" / "TAPIP3D" / "utils" / "viz.html"
    
    if not viz_template.exists():
        print(f"    Warning: viz.html template not found at {viz_template}")
        return
    
    with open(viz_template, 'r') as f:
        html = f.read()
    
    # Apply the same modifications as create_windowed_viz_html in process_tapip3d.py
    # Detect windowed mode by checking for 4D trajectory shape
    html = html.replace(
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        const [totalFrames, numTrajectories] = shape;''',
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        // Windowed mode: shape is (num_windows, T_window, N, 3)
        const isWindowedMode = shape.length === 4;
        let totalFrames, numTrajectories, windowLength;
        if (isWindowedMode) {
          [totalFrames, windowLength, numTrajectories] = shape;
        } else {
          [totalFrames, numTrajectories] = shape;
          windowLength = 1;
        }
        
        this.isWindowedMode = isWindowedMode;
        this.windowLength = windowLength;'''
    )
    
    # Add end marker creation for windowed mode
    html = html.replace(
        '''trajectoryGroup.userData = {
            marker: positionMarker,
            line: trajectoryLine,
            color: colors[i]
          };''',
        '''let endMarker = null;
          if (this.isWindowedMode) {
            const endSphereGeometry = new THREE.SphereGeometry(ballSize * 1.5, 16, 16);
            const endSphereMaterial = new THREE.MeshBasicMaterial({ 
              color: colors[i],
              transparent: true,
              opacity: 1.0
            });
            endMarker = new THREE.Mesh(endSphereGeometry, endSphereMaterial);
            trajectoryGroup.add(endMarker);
          }
          
          trajectoryGroup.userData = {
            marker: positionMarker,
            line: trajectoryLine,
            endMarker: endMarker,
            color: colors[i]
          };'''
    )
    
    # Replace updateTrajectories function
    old_update = '''updateTrajectories(frameIndex) {
        if (!this.data.trajectories || this.trajectories.length === 0) return;
        
        const trajectoryData = this.data.trajectories.data;
        const [totalFrames, numTrajectories] = this.data.trajectories.shape;
        const historyFramesSetting = parseInt(this.ui.trajectoryHistory.value);
        const historyFrames = Math.min(historyFramesSetting, this.config.totalFrames);
        
        for (let i = 0; i < numTrajectories; i++) {
          const trajectoryGroup = this.trajectories[i];
          const { marker, line } = trajectoryGroup.userData;
          
          const currentPos = new THREE.Vector3();
          const currentOffset = (frameIndex * numTrajectories + i) * 3;
          
          currentPos.x = trajectoryData[currentOffset];
          currentPos.y = -trajectoryData[currentOffset + 1];
          currentPos.z = -trajectoryData[currentOffset + 2];
          
          marker.position.copy(currentPos);
          
          const positions = [];
          const historyToShow = Math.min(historyFrames, frameIndex + 1);
          
          for (let j = 0; j < historyToShow; j++) {
            const historyFrame = Math.max(0, frameIndex - j);
            const historyOffset = (historyFrame * numTrajectories + i) * 3;
            
            positions.push(
              trajectoryData[historyOffset],
              -trajectoryData[historyOffset + 1],
              -trajectoryData[historyOffset + 2]
            );
          }
          
          for (let j = historyToShow; j < historyFrames; j++) {
            positions.push(currentPos.x, currentPos.y, currentPos.z);
          }
          
          line.geometry.setPositions(positions);
          
          line.visible = frameIndex > 0;
        }
      }'''
    
    new_update = '''updateTrajectories(frameIndex) {
        if (!this.data.trajectories || this.trajectories.length === 0) return;
        
        const trajectoryData = this.data.trajectories.data;
        const shape = this.data.trajectories.shape;
        
        if (this.isWindowedMode) {
          const [numWindows, windowLength, numTrajectories] = shape;
          
          for (let i = 0; i < numTrajectories; i++) {
            const trajectoryGroup = this.trajectories[i];
            const { marker, line, endMarker } = trajectoryGroup.userData;
            
            const positions = [];
            for (let t = 0; t < windowLength; t++) {
              const offset = ((frameIndex * windowLength + t) * numTrajectories + i) * 3;
              positions.push(
                trajectoryData[offset],
                -trajectoryData[offset + 1],
                -trajectoryData[offset + 2]
              );
            }
            
            const startOffset = (frameIndex * windowLength * numTrajectories + i) * 3;
            marker.position.set(
              trajectoryData[startOffset],
              -trajectoryData[startOffset + 1],
              -trajectoryData[startOffset + 2]
            );
            
            if (endMarker) {
              const endOffset = ((frameIndex * windowLength + windowLength - 1) * numTrajectories + i) * 3;
              endMarker.position.set(
                trajectoryData[endOffset],
                -trajectoryData[endOffset + 1],
                -trajectoryData[endOffset + 2]
              );
            }
            
            line.geometry.setPositions(positions);
            line.visible = true;
          }
        } else {
          const [totalFrames, numTrajectories] = shape;
          const historyFramesSetting = parseInt(this.ui.trajectoryHistory.value);
          const historyFrames = Math.min(historyFramesSetting, this.config.totalFrames);
          
          for (let i = 0; i < numTrajectories; i++) {
            const trajectoryGroup = this.trajectories[i];
            const { marker, line } = trajectoryGroup.userData;
            
            const currentPos = new THREE.Vector3();
            const currentOffset = (frameIndex * numTrajectories + i) * 3;
            
            currentPos.x = trajectoryData[currentOffset];
            currentPos.y = -trajectoryData[currentOffset + 1];
            currentPos.z = -trajectoryData[currentOffset + 2];
            
            marker.position.copy(currentPos);
            
            const positions = [];
            const historyToShow = Math.min(historyFrames, frameIndex + 1);
            
            for (let j = 0; j < historyToShow; j++) {
              const historyFrame = Math.max(0, frameIndex - j);
              const historyOffset = (historyFrame * numTrajectories + i) * 3;
              
              positions.push(
                trajectoryData[historyOffset],
                -trajectoryData[historyOffset + 1],
                -trajectoryData[historyOffset + 2]
              );
            }
            
            for (let j = historyToShow; j < historyFrames; j++) {
              positions.push(currentPos.x, currentPos.y, currentPos.z);
            }
            
            line.geometry.setPositions(positions);
            line.visible = frameIndex > 0;
          }
        }
      }'''
    
    html = html.replace(old_update, new_update)
    
    # Fix FPS bug
    html = html.replace(
        'this.playbackSpeed = this.config.baseFrameRate;',
        'this.playbackSpeed = 1;'
    )
    html = html.replace(
        'const speedRates = speeds.map(s => s * this.config.baseFrameRate);',
        'const speedRates = speeds;'
    )
    html = html.replace(
        'const normalizedSpeed = this.playbackSpeed / this.config.baseFrameRate;',
        'const normalizedSpeed = this.playbackSpeed;'
    )
    html = html.replace(
        'this.playbackSpeed = speedRates[nextIndex];',
        'this.playbackSpeed = speeds[nextIndex];'
    )
    
    # Point to the sample's data.bin file
    html = html.replace('data.bin', f'sample_{sample_idx:03d}_data.bin')
    
    # Add camera path visualization JavaScript
    camera_viz_js = '''
    // Camera path visualization
    setupCameraPath() {
      if (!this.data.camera_positions || !this.config.hasCameraPoses) return;
      
      const positions = this.data.camera_positions.data;
      const forwards = this.data.camera_forwards.data;
      const ups = this.data.camera_ups.data;
      const numPoses = this.config.numCameraPoses;
      
      // Create camera path line
      const pathPositions = [];
      for (let i = 0; i < numPoses; i++) {
        const x = positions[i * 3];
        const y = -positions[i * 3 + 1];  // Flip Y
        const z = -positions[i * 3 + 2];  // Flip Z
        pathPositions.push(x, y, z);
      }
      
      const pathGeometry = new LineGeometry();
      pathGeometry.setPositions(pathPositions);
      const pathMaterial = new LineMaterial({
        color: 0x00ffff,
        linewidth: 3,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
      });
      const cameraPath = new Line2(pathGeometry, pathMaterial);
      this.scene.add(cameraPath);
      
      // Create small camera frustums at each position
      const frustumSize = 0.02;  // Small frustums
      const frustumGroup = new THREE.Group();
      
      for (let i = 0; i < numPoses; i++) {
        const x = positions[i * 3];
        const y = -positions[i * 3 + 1];
        const z = -positions[i * 3 + 2];
        
        const fx = forwards[i * 3];
        const fy = -forwards[i * 3 + 1];
        const fz = -forwards[i * 3 + 2];
        
        const ux = ups[i * 3];
        const uy = -ups[i * 3 + 1];
        const uz = -ups[i * 3 + 2];
        
        // Create frustum as a small pyramid
        const frustumGeometry = new THREE.ConeGeometry(frustumSize * 0.6, frustumSize, 4);
        
        // Color gradient: cyan at start, magenta at end
        const t = i / Math.max(1, numPoses - 1);
        const color = new THREE.Color().setHSL(0.5 - t * 0.3, 1, 0.5);
        
        const frustumMaterial = new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: 0.8,
        });
        
        const frustum = new THREE.Mesh(frustumGeometry, frustumMaterial);
        frustum.position.set(x, y, z);
        
        // Orient frustum to point in forward direction
        const forward = new THREE.Vector3(fx, fy, fz).normalize();
        const up = new THREE.Vector3(ux, uy, uz).normalize();
        
        // Create rotation matrix from forward and up vectors
        const right = new THREE.Vector3().crossVectors(up, forward).normalize();
        const correctedUp = new THREE.Vector3().crossVectors(forward, right).normalize();
        
        const rotMatrix = new THREE.Matrix4();
        rotMatrix.makeBasis(right, correctedUp, forward);
        frustum.rotation.setFromRotationMatrix(rotMatrix);
        frustum.rotateX(Math.PI / 2);  // Cone points along Y by default, rotate to Z
        
        frustumGroup.add(frustum);
      }
      
      this.scene.add(frustumGroup);
      this.cameraPath = cameraPath;
      this.cameraFrustums = frustumGroup;
    }
'''
    
    # Insert camera path setup call after trajectory setup
    html = html.replace(
        'this.setupTrajectories();',
        'this.setupTrajectories();\n      this.setupCameraPath();'
    )
    
    # Insert the camera path function before the closing of the class
    html = html.replace(
        'updateTrajectories(frameIndex) {',
        camera_viz_js + '\n    updateTrajectories(frameIndex) {'
    )
    
    # Add info overlay (bottom-left, navigation will be added top-right later)
    info_html = f'''<div style="position:fixed;bottom:10px;left:10px;z-index:1000;background:rgba(0,0,0,0.8);padding:8px 12px;border-radius:8px;font-family:system-ui;color:#eee;font-size:12px;">
Points: {num_points} | Timesteps: {num_timesteps}
</div>'''
    html = html.replace('<body>', '<body>' + info_html)
    
    # Write HTML file
    html_path = output_dir / f"sample_{sample_idx:03d}_3d.html"
    with open(html_path, 'w') as f:
        f.write(html)


def create_vis_index_html(vis_dir, sample_indices, track_type):
    """Create an index.html for easy navigation of visualizations."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dataloader Test Visualizations</title>
    <style>
        body { 
            margin: 0; padding: 20px; 
            font-family: system-ui, -apple-system, sans-serif; 
            background: #1a1a2e; color: #eee; 
        }
        h1 { color: #a78bfa; margin-bottom: 10px; }
        .info { color: #888; margin-bottom: 20px; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
            gap: 16px; 
        }
        .card { 
            background: #16213e; 
            border-radius: 12px; 
            padding: 16px; 
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover { 
            transform: translateY(-4px); 
            box-shadow: 0 8px 24px rgba(167, 139, 250, 0.2);
        }
        .card h3 { margin: 0 0 12px 0; color: #a78bfa; }
        .card a { 
            display: inline-block;
            padding: 8px 16px; 
            margin: 4px 4px 4px 0;
            background: #0f3460; 
            color: #eee; 
            text-decoration: none; 
            border-radius: 6px;
            font-size: 14px;
        }
        .card a:hover { background: #1a4b8c; }
        .card a.vis-3d { background: #4a1d6e; }
        .card a.vis-3d:hover { background: #6b2d9e; }
        .card img { 
            width: 100%; 
            border-radius: 8px; 
            margin-bottom: 12px;
        }
    </style>
</head>
<body>
    <h1>Dataloader Test Visualizations</h1>
    <p class="info">Track type: ''' + track_type.upper() + f''' | Samples: {len(sample_indices)}</p>
    <div class="grid">
'''
    
    for idx in sorted(sample_indices):
        html += f'''        <div class="card">
            <h3>Sample {idx}</h3>
            <img src="sample_{idx:03d}_2d.png" alt="2D visualization">
            <div>
                <a href="sample_{idx:03d}_2d.png" target="_blank">2D Image</a>
'''
        if track_type == '3d':
            html += f'''                <a href="sample_{idx:03d}_3d.html" target="_blank" class="vis-3d">3D Viewer</a>
'''
        html += '''            </div>
        </div>
'''
    
    html += '''    </div>
</body>
</html>'''
    
    with open(vis_dir / "index.html", 'w') as f:
        f.write(html)


def load_intrinsics_from_h5(h5_path):
    """Load intrinsics from HDF5 file if available."""
    with h5py.File(h5_path, 'r') as f:
        if 'root/intrinsics' in f:
            return np.array(f['root/intrinsics'])
    return None


def test_dataloader(config, visualize=False, num_samples=5, test_augmentations=False, aug_prob=0.9):
    """
    Test loading data from HDF5 files.
    
    Args:
        config: Configuration dict from dataset_config.yaml
        visualize: Whether to create visualizations
        num_samples: Number of samples to visualize
        test_augmentations: Whether to enable augmentations
        aug_prob: Augmentation probability
    """
    dataset_dir = config['dataset_dir']
    img_size = config['img_size']
    frame_stack = config['frame_stack']
    num_track_ts = config['num_track_ts']
    num_track_ids = config['num_track_ids']
    downsample_factor = config['downsample_factor']
    track_type = config.get('track_type', '2d')
    
    print(f"\n{'='*70}")
    print("TESTING DATALOADER WITH H5 DATASET")
    print(f"{'='*70}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Track type: {track_type}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Frame stack: {frame_stack}")
    print(f"Downsample factor: {downsample_factor}x")
    print(f"Future track timesteps: {num_track_ts}")
    print(f"Minimum tracks per frame: {num_track_ids}")
    
    if test_augmentations:
        print(f"Augmentations: ENABLED (prob={aug_prob})")
    else:
        print(f"Augmentations: DISABLED")
    print(f"{'='*70}\n")
    
    # Prepare dataset kwargs
    dataset_kwargs = {
        'dataset_dir': dataset_dir,
        'img_size': img_size,
        'num_track_ts': num_track_ts,
        'num_track_ids': num_track_ids,
        'frame_stack': frame_stack,
        'downsample_factor': downsample_factor,
        'track_type': track_type,
        'cache_all': config.get('cache_all', True),
        'cache_image': config.get('cache_image', True),
        'num_demos': None,
        'aug_prob': aug_prob if test_augmentations else 0.0,
    }
    
    # Add augmentation config if enabled
    if test_augmentations:
        dataset_kwargs.update({
            'aug_color_jitter': config.get('aug_color_jitter'),
            'aug_translation_px': config.get('aug_translation_px', 0),
            'aug_rotation_deg': config.get('aug_rotation_deg', 0),
            'aug_hflip_prob': config.get('aug_hflip_prob', 0.0),
            'aug_vflip_prob': config.get('aug_vflip_prob', 0.0),
            'aug_noise_std': config.get('aug_noise_std', 0.0),
            'aug_depth_noise_std': config.get('aug_depth_noise_std', 0.0),
        })
    
    # Create dataset
    print("Creating dataset...")
    try:
        dataset = TrackDataset(**dataset_kwargs)
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        
        # Check normalization stats
        if dataset.norm_stats is not None:
            print(f"  Normalization stats loaded: ✓")
            print(f"    Track type: {dataset.norm_stats.track_type}")
            print(f"    Displacement mean: {dataset.norm_stats.disp_mean}")
            print(f"    Displacement std: {dataset.norm_stats.disp_std}")
            if track_type == '3d':
                print(f"    Depth mean: {dataset.norm_stats.depth_mean:.4f}")
                print(f"    Depth std: {dataset.norm_stats.depth_std:.4f}")
        else:
            print(f"  Warning: No normalization stats loaded")
            
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if len(dataset) == 0:
        print("✗ Dataset is empty (no valid samples)")
        return False
    
    # Test loading samples
    print(f"\nTesting sample loading...")
    num_samples_to_test = min(5, len(dataset))
    coord_dim = 3 if track_type == '3d' else 2
    
    for i in range(num_samples_to_test):
        try:
            sample = dataset[i]
            
            print(f"\nSample {i}:")
            print(f"  imgs: {sample['imgs'].shape}, dtype={sample['imgs'].dtype}")
            print(f"  query_coords: {sample['query_coords'].shape}")
            print(f"  displacements: {sample['displacements'].shape}")
            
            if sample['poses'] is not None:
                print(f"  poses: {sample['poses'].shape}")
            if sample['depth'] is not None:
                print(f"  depth: {sample['depth'].shape}")
            
            # Validate shapes
            assert sample['imgs'].shape == (frame_stack, 3, img_size, img_size), \
                f"Expected imgs shape ({frame_stack}, 3, {img_size}, {img_size})"
            assert sample['query_coords'].shape[1] == coord_dim, \
                f"Expected query_coords dim {coord_dim}, got {sample['query_coords'].shape[1]}"
            assert sample['displacements'].shape[0] == num_track_ts, \
                f"Expected {num_track_ts} timesteps"
            assert sample['displacements'].shape[2] == coord_dim, \
                f"Expected displacement dim {coord_dim}"
            
            if track_type == '3d':
                assert sample['poses'] is not None, "3D mode should have poses"
                assert sample['poses'].shape == (num_track_ts, 9), \
                    f"Expected poses shape ({num_track_ts}, 9)"
            
            print(f"  ✓ Sample {i} validated")
            
        except Exception as e:
            print(f"  ✗ Failed to load sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test dataloader with batching
    print(f"\nTesting dataloader with batching...")
    try:
        dataloader = get_dataloader(dataset, mode="train", num_workers=0, batch_size=2)
        batch = next(iter(dataloader))
        
        print(f"  Batch keys: {batch.keys()}")
        print(f"  imgs: {batch['imgs'].shape}")
        print(f"  query_coords: {batch['query_coords'].shape}")
        print(f"  displacements: {batch['displacements'].shape}")
        
        assert batch['imgs'].shape[0] == 2, "Expected batch size 2"
        print(f"  ✓ Dataloader batching works correctly")
        
    except Exception as e:
        print(f"  ✗ Failed to test dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Visualization
    if visualize:
        print(f"\nCreating visualizations...")
        vis_dir = PROJECT_ROOT / "misc" / "test_visualizations"
        
        if vis_dir.exists():
            print(f"  Deleting previous visualizations...")
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load intrinsics for 3D visualization
        intrinsics = None
        if track_type == '3d':
            h5_files = list(Path(dataset_dir).glob("*.hdf5"))
            if h5_files:
                intrinsics = load_intrinsics_from_h5(h5_files[0])
                if intrinsics is not None:
                    print(f"  Loaded intrinsics from {h5_files[0].name}")
        
        # Select random samples
        num_to_vis = min(num_samples, len(dataset))
        sample_indices = random.sample(range(len(dataset)), num_to_vis)
        
        visualized_samples = []
        for i, idx in enumerate(sample_indices):
            try:
                sample = dataset[idx]
                
                # 2D visualization (pass norm_stats for denormalization)
                path_2d = visualize_sample_2d(sample, idx, vis_dir, img_size, 
                                               norm_stats=dataset.norm_stats, vis_scale=2)
                print(f"  ✓ Sample {idx}: 2D saved to {path_2d.name}")
                
                # 3D visualization (if 3D mode and intrinsics available)
                if track_type == '3d' and intrinsics is not None:
                    path_3d = visualize_sample_3d(sample, idx, vis_dir, intrinsics, img_size,
                                                   norm_stats=dataset.norm_stats)
                    print(f"           3D saved to {path_3d.name}")
                
                visualized_samples.append(idx)
                    
            except Exception as e:
                print(f"  ✗ Failed to visualize sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create index.html for easy navigation
        create_vis_index_html(vis_dir, visualized_samples, track_type)
        
        # Add navigation to 3D HTML files
        if track_type == '3d' and visualized_samples:
            add_navigation_to_3d_html(vis_dir, visualized_samples)
        
        print(f"\n  Visualizations saved to: {vis_dir}")
        print(f"\n  To view visualizations:")
        print(f"    python misc/serve_test_vis.py")
        print(f"    Then open: http://localhost:8001")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED!")
    print(f"{'='*70}\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HDF5 dataset with dataloader")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--test_augmentations", action="store_true", help="Enable augmentations for testing")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Augmentation probability")
    
    args = parser.parse_args()
    
    # Load config from dataset_config.yaml
    config = load_config()
    
    success = test_dataloader(
        config=config,
        visualize=args.visualize,
        num_samples=args.num_samples,
        test_augmentations=args.test_augmentations,
        aug_prob=args.aug_prob,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
