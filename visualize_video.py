"""
Video validation script for IntentTracker models.

Generates MP4 videos (2D) or interactive HTML viewers (3D) showing model predictions.
"""

import argparse
import os
import sys
import warnings
import yaml
import json
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress xFormers informational warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is available")

from y2r.models.factory import create_model
from y2r.dataloaders.split_dataset import create_train_val_split
from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.dataloaders.utils import NormalizationStats
from y2r.visualization import visualize_tracks_on_frame

# For 3D visualization reuse
PROJECT_ROOT = Path(__file__).parent
VIZ_HTML_TEMPLATE = PROJECT_ROOT / "thirdparty" / "TAPIP3D" / "utils" / "viz.html"


# ============================================================================
# HAND POSE VISUALIZATION HELPERS
# Adapted from misc/test_dataloader.py for consistent visualization style
# ============================================================================

def rot_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation back to 3x3 rotation matrix.
    
    Args:
        rot_6d: (6,) array - first two columns of rotation matrix
    
    Returns:
        (3, 3) rotation matrix
    """
    r1 = rot_6d[:3]
    r2 = rot_6d[3:6]
    
    # Gram-Schmidt orthogonalization
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    
    R = np.stack([r1, r2, r3], axis=1)  # (3, 3)
    return R


def compute_wrist_3d_from_uvd(query_uvd, uvd_displacements, intrinsics, img_size):
    """
    Convert wrist (u, v, d) trajectory to 3D camera coordinates.
    
    Args:
        query_uvd: (3,) initial wrist (u, v, d) - denormalized
        uvd_displacements: (T, 3) displacements - denormalized
        intrinsics: (3, 3) camera intrinsics
        img_size: Image size
    
    Returns:
        wrist_3d: (T, 3) 3D wrist positions in camera coords
    """
    # Reconstruct full trajectory: query + displacements
    T = uvd_displacements.shape[0]
    wrist_uvd = query_uvd[np.newaxis, :] + uvd_displacements  # (T, 3)
    
    # Unproject to 3D
    u = wrist_uvd[:, 0] * img_size
    v = wrist_uvd[:, 1] * img_size
    d = wrist_uvd[:, 2]
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_3d = (u - cx) * d / fx
    y_3d = (v - cy) * d / fy
    z_3d = d
    
    return np.stack([x_3d, y_3d, z_3d], axis=-1)  # (T, 3)


def compute_wrist_rotations(query_rot_6d, rot_displacements):
    """
    Convert wrist rotation data to 3x3 rotation matrices.
    
    Args:
        query_rot_6d: (6,) initial 6D rotation - denormalized
        rot_displacements: (T, 6) relative rotation displacements - denormalized
    
    Returns:
        rotations: (T, 3, 3) rotation matrices
    """
    T = rot_displacements.shape[0]
    
    # Convert query rotation to matrix
    R_0 = rot_6d_to_matrix(query_rot_6d)  # (3, 3)
    
    # For each timestep, the relative rotation is stored as 6D
    # R_rel = R_t @ R_0.T, so R_t = R_rel @ R_0
    rotations = np.zeros((T, 3, 3), dtype=np.float32)
    for t in range(T):
        R_rel = rot_6d_to_matrix(rot_displacements[t])
        R_t = R_rel @ R_0
        rotations[t] = R_t
    
    return rotations


def draw_wrist_trajectories_2d(frame, wrist_data, vis_size):
    """
    Draw wrist trajectories and rotation axes on a 2D visualization frame.
    Just line + pose axes at ALL timesteps, no markers.
    Same style as misc/test_dataloader.py.
    
    Args:
        frame: RGB frame as numpy array (H, W, 3), uint8
        wrist_data: Dict with 'gt' and/or 'pred' keys, each containing:
            - 'left'/'right' with 'uvd' (T, 3) and 'rotations' (T, 3, 3)
        vis_size: Visualization size (width and height)
    
    Returns:
        Modified frame with wrist trajectories and pose axes drawn
    """
    # Make frame contiguous for OpenCV
    frame_bgr = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Colors for GT vs Pred (both left and right same color within each category)
    # GT: green, Pred: purple
    traj_colors = {
        'gt': (0, 255, 0),      # Green for all GT (BGR)
        'pred': (255, 0, 255),  # Purple/Magenta for all Pred (BGR)
    }
    axis_colors = {
        'x': (0, 0, 255),    # Red in BGR
        'y': (0, 255, 0),    # Green in BGR
        'z': (255, 0, 0),    # Blue in BGR
    }
    
    axis_length_px = 40  # Length of rotation axes in pixels (same as test_dataloader.py)
    
    for data_type in ['gt', 'pred']:
        if data_type not in wrist_data:
            continue
        
        traj_color = traj_colors[data_type]
        
        for side in ['left', 'right']:
            if side not in wrist_data[data_type]:
                continue
            
            data = wrist_data[data_type][side]
            uvd_traj = data.get('uvd')  # (T, 3) in normalized [0,1] coords
            rotations = data.get('rotations')  # (T, 3, 3)
            
            if uvd_traj is None:
                continue
            
            T = uvd_traj.shape[0]
            
            # Convert to pixel coordinates (use u, v only)
            wrist_px = (uvd_traj[:, :2] * vis_size).astype(np.int32)  # (T, 2)
            
            # Draw trajectory line only (no markers)
            for t in range(T - 1):
                pt1 = tuple(wrist_px[t])
                pt2 = tuple(wrist_px[t + 1])
                cv2.line(frame_bgr, pt1, pt2, traj_color, thickness=2)
            
            # Draw rotation axes at ALL timesteps
            if rotations is not None:
                for t in range(T):
                    R_t = rotations[t]
                    wrist_pos = wrist_px[t]
                    
                    # Draw each axis (X=red, Y=green, Z=blue)
                    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                        axis_3d = R_t[:, axis_idx]  # (3,) - the axis direction
                        
                        # Project to 2D: use X and Y with some Z perspective
                        axis_2d_x = axis_3d[0] - axis_3d[2] * 0.3
                        axis_2d_y = axis_3d[1] - axis_3d[2] * 0.3
                        
                        # Scale and compute end point
                        end_x = int(wrist_pos[0] + axis_2d_x * axis_length_px)
                        end_y = int(wrist_pos[1] + axis_2d_y * axis_length_px)
                        
                        # Draw axis line
                        cv2.line(frame_bgr, tuple(wrist_pos), (end_x, end_y), 
                                 axis_colors[axis_name], thickness=2)
    
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_config(config_path):
    """Load YAML configuration file and associated dataset config."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load dataset config if referenced
    if 'dataset_config' in cfg:
        dataset_config_path = cfg['dataset_config']
        # Handle relative paths
        if not os.path.isabs(dataset_config_path):
            config_dir = os.path.dirname(config_path)
            dataset_config_path = os.path.join(config_dir, dataset_config_path)
        
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
        
        # Add dataset config to main config
        cfg['dataset_cfg'] = dataset_cfg
        cfg['dataset_dir'] = dataset_cfg['dataset_dir']
        
        # Derive model parameters from dataset config
        if 'model' in cfg:
            cfg['model']['num_future_steps'] = dataset_cfg['num_track_ts']
            cfg['model']['frame_stack'] = dataset_cfg['frame_stack']
    
    # Convert to namespace for easier access
    class Namespace:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Namespace(v))
                else:
                    setattr(self, k, v)
    
    return Namespace(cfg)


def load_normalization_stats(stats_path):
    """Load displacement normalization statistics."""
    with open(stats_path, 'r') as f:
        stats = yaml.safe_load(f)
    return stats


def load_model(checkpoint_path, cfg, disp_stats, device, text_mode=False):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load to CPU first to avoid CUDA memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model using factory
    model_type = getattr(cfg.model, 'model_type', 'direct')
    is_diffusion = (model_type == 'diffusion')
    
    model = create_model(cfg, disp_stats=disp_stats, device=device, text_mode=text_mode)
    
    # Load state dict (handle both EMA and regular checkpoints)
    if 'ema_model_state_dict' in checkpoint:
        # Use EMA model for validation (better quality)
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print("Loaded EMA model state")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded regular model state")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state directly")
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Checkpoint epoch: {epoch}")
    
    return model, is_diffusion, epoch


# ============================================================================
# 3D VISUALIZATION FUNCTIONS
# These functions are adapted from dataset_scripts/process_tapip3d.py
# See that file for the original implementation and documentation.
# ============================================================================

def compress_and_write(filename, header, blob):
    """Write compressed binary data with header."""
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)


def create_windowed_viz_html(vis_path):
    """Create the windowed visualization HTML that shows full trajectory polylines."""
    
    if not VIZ_HTML_TEMPLATE.exists():
        print(f"Warning: viz.html template not found at {VIZ_HTML_TEMPLATE}")
        return None
    
    # Read the original viz.html as template
    with open(VIZ_HTML_TEMPLATE, 'r') as f:
        html = f.read()
    
    # Detect windowed mode by checking for 4D trajectory shape
    html = html.replace(
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        const [totalFrames, numTrajectories] = shape;''',
        '''const shape = this.data.trajectories.shape;
        if (!shape || shape.length < 2) return;
        
        // Windowed mode: shape is (num_windows, T_window, N, 3)
        // Original mode: shape is (totalFrames, N, 3)
        const isWindowedMode = shape.length === 4;
        let totalFrames, numTrajectories, windowLength;
        if (isWindowedMode) {
          [totalFrames, windowLength, numTrajectories] = shape;
        } else {
          [totalFrames, numTrajectories] = shape;
          windowLength = 1;
        }
        
        // Store for updateTrajectories
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
        '''// Create end marker for windowed mode
          let endMarker = null;
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
    
    # Replace updateTrajectories function to handle full polylines
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
          // Windowed mode: shape is (num_windows, T_window, N, 3)
          const [numWindows, windowLength, numTrajectories] = shape;
          
          for (let i = 0; i < numTrajectories; i++) {
            const trajectoryGroup = this.trajectories[i];
            const { marker, line, endMarker } = trajectoryGroup.userData;
            
            // Collect all positions for this trajectory
            const positions = [];
            for (let t = 0; t < windowLength; t++) {
              const offset = ((frameIndex * windowLength + t) * numTrajectories + i) * 3;
              positions.push(
                trajectoryData[offset],
                -trajectoryData[offset + 1],
                -trajectoryData[offset + 2]
              );
            }
            
            // Start position (t=0)
            const startOffset = (frameIndex * windowLength * numTrajectories + i) * 3;
            marker.position.set(
              trajectoryData[startOffset],
              -trajectoryData[startOffset + 1],
              -trajectoryData[startOffset + 2]
            );
            
            // End position (t=windowLength-1)
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
          // Original mode
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
    
    # Replace createColorPalette to use GT (green) / Pred (purple) color scheme
    # First half of points = GT (greens), second half = Pred (purples)
    old_color_palette = '''createColorPalette(count) {
        const colors = [];
        const hueStep = 360 / count;
        
        for (let i = 0; i < count; i++) {
          const hue = (i * hueStep) % 360;
          const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.6);
          colors.push(color);
        }
        
        return colors;
      }'''
    
    new_color_palette = '''createColorPalette(count) {
        const colors = [];
        const halfCount = Math.floor(count / 2);
        
        for (let i = 0; i < count; i++) {
          let color;
          if (i < halfCount) {
            // GT points: green hues (100-140 degrees, varies slightly per point)
            const hue = 120 + (i / halfCount) * 20 - 10;  // 110-130 range
            color = new THREE.Color().setHSL(hue / 360, 0.9, 0.45);
          } else {
            // Pred points: purple/magenta hues (270-310 degrees)
            const predIdx = i - halfCount;
            const hue = 280 + (predIdx / halfCount) * 30 - 15;  // 265-295 range
            color = new THREE.Color().setHSL(hue / 360, 0.85, 0.55);
          }
          colors.push(color);
        }
        
        return colors;
      }'''
    
    html = html.replace(old_color_palette, new_color_palette)
    
    # Fix FPS bug
    html = html.replace(
        'this.playbackSpeed = this.config.baseFrameRate;',
        'this.playbackSpeed = 1;  // Speed multiplier (1x normal speed)'
    )
    html = html.replace(
        'const speedRates = speeds.map(s => s * this.config.baseFrameRate);',
        'const speedRates = speeds;  // Speed multipliers directly'
    )
    html = html.replace(
        'const normalizedSpeed = this.playbackSpeed / this.config.baseFrameRate;',
        'const normalizedSpeed = this.playbackSpeed;  // Already a multiplier'
    )
    html = html.replace(
        'this.playbackSpeed = speedRates[nextIndex];',
        'this.playbackSpeed = speeds[nextIndex];'
    )
    
    # Add wrist visualization JavaScript for GT vs Pred wrist trajectories
    # Same style as misc/test_dataloader.py: line + axes at ALL timesteps, no markers
    wrist_viz_js = '''
    // Wrist visualization: trajectory line + pose axes at ALL timesteps (same as test_dataloader.py)
    setupWristVisualization() {
      this.wristGroups = {};
      const axisLength = 0.03;  // Length of rotation axes
      const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
      
      // Colors: GT=green (both left and right), Pred=purple (both left and right)
      const colors = {
        gt_left: { main: 0x00ff00, x: 0xff0000, y: 0x00ff00, z: 0x0000ff },     // Green for GT
        gt_right: { main: 0x00ff00, x: 0xff0000, y: 0x00ff00, z: 0x0000ff },    // Green for GT
        pred_left: { main: 0xff00ff, x: 0xff0000, y: 0x00ff00, z: 0x0000ff },   // Purple for Pred
        pred_right: { main: 0xff00ff, x: 0xff0000, y: 0x00ff00, z: 0x0000ff },  // Purple for Pred
      };
      
      const wristConfigs = [
        { key: 'gt_left', posKey: 'gt_left_wrist_positions', configKey: 'hasGtLeftWrist' },
        { key: 'gt_right', posKey: 'gt_right_wrist_positions', configKey: 'hasGtRightWrist' },
        { key: 'pred_left', posKey: 'pred_left_wrist_positions', configKey: 'hasPredLeftWrist' },
        { key: 'pred_right', posKey: 'pred_right_wrist_positions', configKey: 'hasPredRightWrist' },
      ];
      
      for (const cfg of wristConfigs) {
        if (!this.config[cfg.configKey] || !this.data[cfg.posKey]) continue;
        
        const positions = this.data[cfg.posKey];
        const rotX = this.data[cfg.key + '_wrist_rot_x'];
        const rotY = this.data[cfg.key + '_wrist_rot_y'];
        const rotZ = this.data[cfg.key + '_wrist_rot_z'];
        if (!positions) continue;
        
        const group = new THREE.Group();
        const sideColors = colors[cfg.key];
        
        // Store for updateWristVisualization
        this.wristGroups[cfg.key] = {
          group: group,
          positions: positions,
          rotX: rotX,
          rotY: rotY,
          rotZ: rotZ,
          colors: sideColors,
          axisLength: axisLength,
          resolution: resolution,
        };
        
        this.scene.add(group);
      }
    }
    
    updateWristVisualization(frameIndex) {
      if (!this.wristGroups) return;
      
      const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
      const axisLength = 0.03;
      
      for (const key of Object.keys(this.wristGroups)) {
        const wristData = this.wristGroups[key];
        const { group, positions, rotX, rotY, rotZ, colors } = wristData;
        
        // Clear previous geometry
        while (group.children.length > 0) {
          const child = group.children[0];
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
          group.remove(child);
        }
        
        if (!positions || !positions.data) continue;
        
        // Shape: (num_frames, T, 3)
        const shape = positions.shape;
        const numFrames = shape[0];
        const T = shape[1];
        
        if (frameIndex >= numFrames) continue;
        
        // Build trajectory line for this frame
        const pathPositions = [];
        for (let t = 0; t < T; t++) {
          const offset = (frameIndex * T + t) * 3;
          const x = positions.data[offset];
          const y = -positions.data[offset + 1];
          const z = -positions.data[offset + 2];
          pathPositions.push(x, y, z);
        }
        
        // linewidth = 4 for trajectory (same as test_dataloader.py)
        const pathGeometry = new THREE.LineGeometry();
        pathGeometry.setPositions(pathPositions);
        const pathMaterial = new THREE.LineMaterial({
          color: colors.main,
          linewidth: 4,
          resolution: resolution,
        });
        const trajectoryLine = new THREE.Line2(pathGeometry, pathMaterial);
        group.add(trajectoryLine);
        
        // No start/end markers - just line + axes (same as test_dataloader.py)
        
        // Draw rotation axes at ALL timesteps (same as test_dataloader.py)
        if (rotX && rotY && rotZ && rotX.data && rotY.data && rotZ.data) {
          for (let t = 0; t < T; t++) {
            const posOffset = (frameIndex * T + t) * 3;
            const px = positions.data[posOffset];
            const py = -positions.data[posOffset + 1];
            const pz = -positions.data[posOffset + 2];
            
            const origin = new THREE.Vector3(px, py, pz);
            
            // X axis (red)
            const rxOffset = (frameIndex * T + t) * 3;
            const rxDir = new THREE.Vector3(
              rotX.data[rxOffset],
              -rotX.data[rxOffset + 1],
              -rotX.data[rxOffset + 2]
            ).normalize().multiplyScalar(axisLength);
            const xEnd = origin.clone().add(rxDir);
            const xGeo = new THREE.LineGeometry();
            xGeo.setPositions([origin.x, origin.y, origin.z, xEnd.x, xEnd.y, xEnd.z]);
            const xMat = new THREE.LineMaterial({ color: colors.x, linewidth: 3, resolution: resolution });
            group.add(new THREE.Line2(xGeo, xMat));
            
            // Y axis (green)
            const ryDir = new THREE.Vector3(
              rotY.data[rxOffset],
              -rotY.data[rxOffset + 1],
              -rotY.data[rxOffset + 2]
            ).normalize().multiplyScalar(axisLength);
            const yEnd = origin.clone().add(ryDir);
            const yGeo = new THREE.LineGeometry();
            yGeo.setPositions([origin.x, origin.y, origin.z, yEnd.x, yEnd.y, yEnd.z]);
            const yMat = new THREE.LineMaterial({ color: colors.y, linewidth: 3, resolution: resolution });
            group.add(new THREE.Line2(yGeo, yMat));
            
            // Z axis (blue)
            const rzDir = new THREE.Vector3(
              rotZ.data[rxOffset],
              -rotZ.data[rxOffset + 1],
              -rotZ.data[rxOffset + 2]
            ).normalize().multiplyScalar(axisLength);
            const zEnd = origin.clone().add(rzDir);
            const zGeo = new THREE.LineGeometry();
            zGeo.setPositions([origin.x, origin.y, origin.z, zEnd.x, zEnd.y, zEnd.z]);
            const zMat = new THREE.LineMaterial({ color: colors.z, linewidth: 3, resolution: resolution });
            group.add(new THREE.Line2(zGeo, zMat));
          }
        }
      }
    }
'''
    
    # Insert wrist visualization setup call after trajectory init
    html = html.replace(
        'this.initTrajectories();',
        'this.initTrajectories();\n          this.setupWristVisualization();'
    )
    
    # Insert the wrist visualization functions before updateTrajectories
    html = html.replace(
        'updateTrajectories(frameIndex) {',
        wrist_viz_js + '\n    updateTrajectories(frameIndex) {'
    )
    
    # Call updateWristVisualization at the end of updateTrajectories (windowed mode branch)
    html = html.replace(
        '''line.geometry.setPositions(positions);
            line.visible = true;
          }
        } else {''',
        '''line.geometry.setPositions(positions);
            line.visible = true;
          }
          this.updateWristVisualization(frameIndex);
        } else {'''
    )
    
    # Save
    output_path = vis_path / "viz_windowed.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def create_3d_viz_index(vis_path, video_list, model_type, epoch):
    """Create index page with all 3D visualization videos."""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>3D Model Predictions - {model_type} (epoch {epoch})</title>
<style>
body {{ margin: 0; padding: 20px; font-family: system-ui; background: #111; color: #eee; }}
h1 {{ color: #a78bfa; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
.card {{ background: #222; border-radius: 8px; padding: 16px; text-decoration: none; color: #eee; }}
.card:hover {{ background: #333; }}
.card h3 {{ margin: 0; color: #a78bfa; }}
.train {{ border-left: 3px solid #4ade80; }}
.val {{ border-left: 3px solid #f97316; }}
</style></head><body>
<h1>3D Model Predictions - {model_type} (epoch {epoch})</h1>
<div class="grid">
"""
    for i, (name, split) in enumerate(video_list):
        card_class = "train" if split == "train" else "val"
        html += f'<a href="{name}.html" class="card {card_class}"><h3>{name}</h3><p>{split} | {i+1}/{len(video_list)}</p></a>\n'
    html += "</div></body></html>"
    
    with open(vis_path / "index.html", 'w') as f:
        f.write(html)


def create_3d_video_html(video_name, video_list, vis_path):
    """Create HTML for a 3D video with prev/next navigation."""
    video_names = [v[0] for v in video_list]
    idx = video_names.index(video_name)
    prev_v = video_names[idx - 1] if idx > 0 else None
    next_v = video_names[idx + 1] if idx < len(video_names) - 1 else None
    
    viz_html_path = vis_path / "viz_windowed.html"
    if not viz_html_path.exists():
        return
    
    with open(viz_html_path, 'r') as f:
        html = f.read()
    
    html = html.replace('data.bin', f'{video_name}_data.bin')
    
    nav = f"""<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(0,0,0,0.8);padding:8px 12px;border-radius:8px;display:flex;gap:10px;font-family:system-ui;color:#eee;">
<a href="index.html" style="color:#a78bfa;">Index</a>
<span>|</span>
{"<a href='" + prev_v + ".html' style='color:#a78bfa;'>← Prev</a>" if prev_v else "<span style='color:#666;'>← Prev</span>"}
<span style="font-weight:600;">{video_name}</span>
<span style="color:#888;">({idx+1}/{len(video_list)})</span>
{"<a href='" + next_v + ".html' style='color:#a78bfa;'>Next →</a>" if next_v else "<span style='color:#666;'>Next →</span>"}
</div>"""
    
    html = html.replace('<body>', '<body>' + nav)
    
    with open(vis_path / f"{video_name}.html", 'w') as f:
        f.write(html)


def generate_video_2d(model, h5_file, output_path, cfg, norm_stats, is_diffusion, device, text_mode=False):
    """
    Generate 2D video with model predictions overlaid.
    
    Args:
        model: trained model
        h5_file: path to HDF5 file
        output_path: where to save MP4
        cfg: configuration namespace
        norm_stats: NormalizationStats object
        is_diffusion: whether model is diffusion-based
        device: torch device
        text_mode: whether to use text conditioning
    """
    print(f"\nGenerating 2D video for: {h5_file}")
    print(f"Output: {output_path}")
    
    # Get hand_mode from config
    hand_mode = getattr(cfg.model, 'hand_mode', None)
    
    # Create dataset for this single H5 file (no augmentation - augmentations are applied on GPU in training)
    dataset = TrackDataset(
        h5_files=[h5_file],
        img_size=cfg.dataset_cfg.img_size,
        frame_stack=cfg.dataset_cfg.frame_stack,
        num_track_ts=cfg.dataset_cfg.num_track_ts,
        num_track_ids=cfg.dataset_cfg.num_track_ids,
        downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
        cache_all=cfg.dataset_cfg.cache_all,
        cache_image=cfg.dataset_cfg.cache_image,
        track_type='2d',
        hand_mode=hand_mode,
        text_mode=text_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    stride = getattr(cfg.dataset_cfg, 'downsample_factor', 1)
    print(f"Dataset length: {len(dataset)} frames")
    print(f"Processing every {stride} frames")
    if hand_mode:
        print(f"Hand mode: {hand_mode}")
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    frame_size = (512, 512)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    errors = []
    
    for sample_idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):
        if sample_idx % stride != 0:
            continue
        
        # Unpack dict batch
        frames = batch['imgs'].to(device)  # (1, frame_stack, C, H, W)
        query_coords = batch['query_coords'].to(device)  # (1, N, 2)
        displacements = batch['displacements'].to(device)  # (1, T, N, 2) - normalized
        
        # Permute displacements: (B, T, N, D) -> (B, N, T, D)
        gt_disp_normalized = displacements.permute(0, 2, 1, 3)
        
        # Prepare hand data if available
        hand_query_uvd = None
        hand_query_rot = None
        gt_hand_uvd_disp = None
        gt_hand_rot_disp = None
        
        if hand_mode:
            hand_query_uvd_list = []
            hand_query_rot_list = []
            gt_hand_uvd_disp_list = []
            gt_hand_rot_disp_list = []
            
            sides = ['left'] if hand_mode == 'left' else ['right'] if hand_mode == 'right' else ['left', 'right']
            for side in sides:
                prefix = f'{side}_wrist'
                if f'{prefix}_query_uvd' in batch:
                    hand_query_uvd_list.append(batch[f'{prefix}_query_uvd'].to(device))
                    hand_query_rot_list.append(batch[f'{prefix}_query_rot_6d'].to(device))
                    gt_hand_uvd_disp_list.append(batch[f'{prefix}_uvd_displacements'].to(device))
                    gt_hand_rot_disp_list.append(batch[f'{prefix}_rot_displacements'].to(device))
            
            if hand_query_uvd_list:
                hand_query_uvd = torch.stack(hand_query_uvd_list, dim=1)  # (B, H, 3)
                hand_query_rot = torch.stack(hand_query_rot_list, dim=1)  # (B, H, 6)
                gt_hand_uvd_disp = torch.stack(gt_hand_uvd_disp_list, dim=1)  # (B, H, T, 3)
                gt_hand_rot_disp = torch.stack(gt_hand_rot_disp_list, dim=1)  # (B, H, T, 6)
        
        # Get text for text_mode
        text = batch.get('text') if text_mode else None
        
        # Run model prediction (use autocast for xformers compatibility on new GPUs)
        with torch.no_grad(), autocast(device_type=device.type, enabled=True):
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                outputs = model.predict(frames, query_coords, num_inference_steps=num_inference_steps,
                                       hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
            else:
                outputs = model.predict(frames, query_coords,
                                       hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
        
        # Handle dict output from model
        if isinstance(outputs, dict):
            pred_disp = outputs['track_disp']
            pred_hand_uvd_disp = outputs.get('hand_uvd_disp')
            pred_hand_rot_disp = outputs.get('hand_rot_disp')
        else:
            pred_disp = outputs
            pred_hand_uvd_disp = None
            pred_hand_rot_disp = None
        
        # Use last frame for visualization
        frame = frames[0, -1]  # (3, H, W) tensor
        pred_disp_single = pred_disp[0]  # (N, T, 2) - normalized
        gt_disp_single = gt_disp_normalized[0]  # (N, T, 2) - normalized
        query_coords_single = query_coords[0]  # (N, 2)
        
        # Denormalize displacements for visualization
        pred_disp_denorm = norm_stats.denormalize_displacement_torch(pred_disp_single)
        gt_disp_denorm = norm_stats.denormalize_displacement_torch(gt_disp_single)
        
        # Convert displacements to positions
        pred_tracks_viz = query_coords_single.unsqueeze(1) + pred_disp_denorm  # (N, T, 2)
        gt_tracks_viz = query_coords_single.unsqueeze(1) + gt_disp_denorm  # (N, T, 2)
        
        # Create visualization
        fig = visualize_tracks_on_frame(
            frame=frame,
            query_coords=query_coords_single,
            gt_tracks=gt_tracks_viz,
            pred_tracks=pred_tracks_viz,
            title=f"Frame {sample_idx}"
        )
        
        # Convert matplotlib figure to OpenCV frame
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        vis_img = buf[:, :, :3].copy()  # RGB uint8
        plt.close(fig)
        
        # Draw hand trajectories if available
        if hand_mode and gt_hand_uvd_disp is not None:
            # Build wrist_data dict for visualization
            wrist_data = {'gt': {}, 'pred': {}}
            
            sides = ['left'] if hand_mode == 'left' else ['right'] if hand_mode == 'right' else ['left', 'right']
            for h_idx, side in enumerate(sides):
                if h_idx >= gt_hand_uvd_disp.shape[1]:
                    continue
                
                # GT wrist data
                gt_query_uvd = hand_query_uvd[0, h_idx].cpu().numpy()  # (3,)
                gt_uvd_disp = gt_hand_uvd_disp[0, h_idx].cpu().numpy()  # (T, 3)
                gt_query_rot = hand_query_rot[0, h_idx].cpu().numpy()  # (6,)
                gt_rot_disp = gt_hand_rot_disp[0, h_idx].cpu().numpy()  # (T, 6)
                
                # Denormalize GT
                if norm_stats.has_hand_stats:
                    gt_uvd_disp = norm_stats.denormalize_hand_uvd_disp(gt_uvd_disp)
                    gt_rot_disp = norm_stats.denormalize_hand_rot_disp(gt_rot_disp)
                    gt_query_uvd_denorm = gt_query_uvd.copy()
                    gt_query_uvd_denorm[2] = norm_stats.denormalize_depth(gt_query_uvd[2])
                else:
                    gt_query_uvd_denorm = gt_query_uvd
                
                # Reconstruct GT trajectory
                T = gt_uvd_disp.shape[0]
                gt_uvd_traj = gt_query_uvd_denorm[np.newaxis, :] + gt_uvd_disp  # (T, 3)
                gt_rotations = compute_wrist_rotations(gt_query_rot, gt_rot_disp)
                
                wrist_data['gt'][side] = {
                    'uvd': gt_uvd_traj,
                    'rotations': gt_rotations,
                }
                
                # Pred wrist data
                if pred_hand_uvd_disp is not None and pred_hand_rot_disp is not None:
                    pred_uvd_disp = pred_hand_uvd_disp[0, h_idx].cpu().numpy()  # (T, 3)
                    pred_rot_disp = pred_hand_rot_disp[0, h_idx].cpu().numpy()  # (T, 6)
                    
                    # Denormalize pred (model outputs are normalized)
                    if norm_stats.has_hand_stats:
                        pred_uvd_disp = norm_stats.denormalize_hand_uvd_disp(pred_uvd_disp)
                        pred_rot_disp = norm_stats.denormalize_hand_rot_disp(pred_rot_disp)
                    
                    # Reconstruct pred trajectory
                    pred_uvd_traj = gt_query_uvd_denorm[np.newaxis, :] + pred_uvd_disp  # (T, 3)
                    pred_rotations = compute_wrist_rotations(gt_query_rot, pred_rot_disp)
                    
                    wrist_data['pred'][side] = {
                        'uvd': pred_uvd_traj,
                        'rotations': pred_rotations,
                    }
            
            # Draw wrist trajectories on the visualization
            vis_img = draw_wrist_trajectories_2d(vis_img, wrist_data, frame_size[0])
        
        # Resize and convert
        vis_frame = cv2.resize(vis_img, frame_size)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Compute error
        error = torch.norm(pred_disp_denorm - gt_disp_denorm, dim=-1).mean().item()
        errors.append(error)
        
        out.write(vis_frame)
    
    out.release()
    
    avg_error = np.mean(errors) if errors else 0.0
    print(f"Average error: {avg_error:.4f}")
    print(f"Video saved to: {output_path}")
    
    return avg_error


def load_intrinsics_from_h5(h5_path):
    """Load intrinsics from HDF5 file if available."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        if 'root/intrinsics' in f:
            return np.array(f['root/intrinsics'])
    return None


def unproject_to_3d(tracks_uvd, intrinsics, img_size):
    """
    Unproject (u, v, d) tracks to 3D camera coordinates.
    
    Args:
        tracks_uvd: (T, N, 3) or (N, T, 3) array with (u, v, d) in normalized [0,1] coords
        intrinsics: (3, 3) camera intrinsics matrix
        img_size: image size for coordinate conversion
    
    Returns:
        points_3d: same shape as input, with 3D camera coordinates
    """
    # Convert normalized (u, v) to pixel coordinates
    u = tracks_uvd[..., 0] * img_size  # pixel x
    v = tracks_uvd[..., 1] * img_size  # pixel y
    d = tracks_uvd[..., 2]  # depth in meters
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Unproject to 3D camera coords
    x_3d = (u - cx) * d / fx
    y_3d = (v - cy) * d / fy
    z_3d = d
    
    return np.stack([x_3d, y_3d, z_3d], axis=-1)


def generate_video_3d(model, h5_file, video_name, vis_path, cfg, norm_stats, is_diffusion, device, text_mode=False):
    """
    Generate 3D visualization for a video with model predictions.
    Creates interactive HTML viewer showing predicted vs GT trajectories.
    
    Uses proper intrinsics-based unprojection (same approach as misc/test_dataloader.py).
    
    Args:
        model: trained model
        h5_file: path to HDF5 file
        video_name: name for output files
        vis_path: directory to save visualization files
        cfg: configuration namespace
        norm_stats: NormalizationStats object
        is_diffusion: whether model is diffusion-based
        device: torch device
        text_mode: whether to use text conditioning
    """
    print(f"\nGenerating 3D visualization for: {h5_file}")
    
    # Get hand_mode from config
    hand_mode = getattr(cfg.model, 'hand_mode', None)
    
    # Load intrinsics from H5 file
    intrinsics = load_intrinsics_from_h5(h5_file)
    if intrinsics is None:
        print(f"  Warning: No intrinsics found in {h5_file}, using default")
        # Default intrinsics for 224x224 image (approximate)
        img_size = cfg.dataset_cfg.img_size
        fx = fy = img_size * 1.2  # Reasonable default focal length
        cx, cy = img_size / 2, img_size / 2
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        print(f"  Loaded intrinsics from H5 file")
    
    img_size = cfg.dataset_cfg.img_size
    
    # Create dataset for this single H5 file
    dataset = TrackDataset(
        h5_files=[h5_file],
        img_size=img_size,
        frame_stack=cfg.dataset_cfg.frame_stack,
        num_track_ts=cfg.dataset_cfg.num_track_ts,
        num_track_ids=cfg.dataset_cfg.num_track_ids,
        downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
        cache_all=cfg.dataset_cfg.cache_all,
        cache_image=cfg.dataset_cfg.cache_image,
        track_type='3d',
        hand_mode=hand_mode,
        text_mode=text_mode,
    )
    
    # Use ALL samples from the video (respecting dataset's natural frame ordering)
    num_samples = len(dataset)
    sample_indices = list(range(num_samples))
    
    print(f"  Dataset length: {len(dataset)} frames, visualizing all {num_samples}")
    if hand_mode:
        print(f"  Hand mode: {hand_mode}")
    
    # Fixed visualization size
    fixed_size = (256, 192)
    
    # Scale intrinsics for visualization size
    scale_x = fixed_size[0] / img_size
    scale_y = fixed_size[1] / img_size
    intrinsics_scaled = intrinsics.copy()
    intrinsics_scaled[0, :] *= scale_x
    intrinsics_scaled[1, :] *= scale_y
    
    # Compute FOV from scaled intrinsics
    fx_scaled, fy_scaled = intrinsics_scaled[0, 0], intrinsics_scaled[1, 1]
    fov_y = 2 * np.arctan(fixed_size[1] / (2 * fy_scaled)) * (180 / np.pi)
    fov_x = 2 * np.arctan(fixed_size[0] / (2 * fx_scaled)) * (180 / np.pi)
    
    # Store visualization data
    rgb_frames = []
    depth_frames = []
    gt_trajectories_3d = []
    pred_trajectories_3d = []
    
    # Store wrist data per frame (for each side: gt and pred positions/rotations)
    wrist_data_frames = {
        'gt_left_positions': [], 'gt_left_rotations': [],
        'gt_right_positions': [], 'gt_right_rotations': [],
        'pred_left_positions': [], 'pred_left_rotations': [],
        'pred_right_positions': [], 'pred_right_rotations': [],
    }
    has_left_wrist = False
    has_right_wrist = False
    
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        batch = dataset[sample_idx]
        
        # Move to device and add batch dimension
        frames = batch['imgs'].unsqueeze(0).to(device)  # (1, frame_stack, C, H, W)
        query_coords = batch['query_coords'].unsqueeze(0).to(device)  # (1, N, 3)
        displacements = batch['displacements'].unsqueeze(0).to(device)  # (1, T, N, 3) - normalized
        depth = batch['depth'].unsqueeze(0).to(device) if batch['depth'] is not None else None
        
        # Permute displacements: (B, T, N, D) -> (B, N, T, D)
        gt_disp_normalized = displacements.permute(0, 2, 1, 3)
        
        # Prepare depth for model
        model_depth = None
        if depth is not None:
            frame_stack = frames.shape[1]
            model_depth = depth[:, :frame_stack, :, :].unsqueeze(2)  # (B, frame_stack, 1, H, W)
        
        # Prepare hand data if available
        hand_query_uvd = None
        hand_query_rot = None
        gt_hand_uvd_disp = None
        gt_hand_rot_disp = None
        
        if hand_mode:
            hand_query_uvd_list = []
            hand_query_rot_list = []
            gt_hand_uvd_disp_list = []
            gt_hand_rot_disp_list = []
            
            sides = ['left'] if hand_mode == 'left' else ['right'] if hand_mode == 'right' else ['left', 'right']
            for side in sides:
                prefix = f'{side}_wrist'
                if batch.get(f'{prefix}_query_uvd') is not None:
                    hand_query_uvd_list.append(batch[f'{prefix}_query_uvd'].unsqueeze(0).to(device))
                    hand_query_rot_list.append(batch[f'{prefix}_query_rot_6d'].unsqueeze(0).to(device))
                    gt_hand_uvd_disp_list.append(batch[f'{prefix}_uvd_displacements'].unsqueeze(0).to(device))
                    gt_hand_rot_disp_list.append(batch[f'{prefix}_rot_displacements'].unsqueeze(0).to(device))
            
            if hand_query_uvd_list:
                # Stack along dim=1 to create hand dimension: (1, H, ...)
                hand_query_uvd = torch.stack(hand_query_uvd_list, dim=1)  # (1, H, 3)
                hand_query_rot = torch.stack(hand_query_rot_list, dim=1)  # (1, H, 6)
                gt_hand_uvd_disp = torch.stack(gt_hand_uvd_disp_list, dim=1)  # (1, H, T, 3)
                gt_hand_rot_disp = torch.stack(gt_hand_rot_disp_list, dim=1)  # (1, H, T, 6)
        
        # Get text for text_mode (wrap in list since we're processing single samples)
        text = [batch.get('text')] if text_mode and batch.get('text') is not None else None
        
        # Run model prediction (use autocast for xformers compatibility on new GPUs)
        with torch.no_grad(), autocast(device_type=device.type, enabled=True):
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                outputs = model.predict(frames, query_coords, depth=model_depth, 
                                        num_inference_steps=num_inference_steps,
                                        hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
            else:
                outputs = model.predict(frames, query_coords, depth=model_depth,
                                        hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
        
        # Handle dict output from model
        if isinstance(outputs, dict):
            pred_disp = outputs['track_disp']
            pred_hand_uvd_disp = outputs.get('hand_uvd_disp')
            pred_hand_rot_disp = outputs.get('hand_rot_disp')
        else:
            pred_disp = outputs
            pred_hand_uvd_disp = None
            pred_hand_rot_disp = None
        
        # Get single sample data (move to CPU)
        pred_disp_single = pred_disp[0].cpu()  # (N, T, 3) - normalized
        gt_disp_single = gt_disp_normalized[0].cpu()  # (N, T, 3) - normalized
        query_coords_single = query_coords[0].cpu()  # (N, 3)
        
        # Denormalize displacements
        pred_disp_denorm = norm_stats.denormalize_displacement_torch(pred_disp_single)
        gt_disp_denorm = norm_stats.denormalize_displacement_torch(gt_disp_single)
        
        # Denormalize depth in query_coords (only the depth component at index 2)
        query_depth_denorm = norm_stats.denormalize_depth_torch(query_coords_single[:, 2])
        query_coords_denorm = query_coords_single.clone()
        query_coords_denorm[:, 2] = query_depth_denorm
        
        # Reconstruct full tracks: position_t = query_coords + displacement_t
        # query_coords: (N, 3) = (u, v, d)
        # displacements: (N, T, 3) = (du, dv, dd)
        N = query_coords_denorm.shape[0]
        T = gt_disp_denorm.shape[1]
        
        # Expand query_coords to (N, T, 3) and add displacements
        query_expanded = query_coords_denorm.unsqueeze(1).expand(N, T, 3)  # (N, T, 3)
        gt_tracks_uvd = query_expanded + gt_disp_denorm  # (N, T, 3)
        pred_tracks_uvd = query_expanded + pred_disp_denorm  # (N, T, 3)
        
        # Unproject to 3D camera coordinates using intrinsics
        gt_tracks_3d = unproject_to_3d(gt_tracks_uvd.numpy(), intrinsics, img_size)  # (N, T, 3)
        pred_tracks_3d = unproject_to_3d(pred_tracks_uvd.numpy(), intrinsics, img_size)  # (N, T, 3)
        
        gt_trajectories_3d.append(gt_tracks_3d.astype(np.float32))
        pred_trajectories_3d.append(pred_tracks_3d.astype(np.float32))
        
        # Process hand data for 3D visualization
        if hand_mode and gt_hand_uvd_disp is not None:
            sides = ['left'] if hand_mode == 'left' else ['right'] if hand_mode == 'right' else ['left', 'right']
            
            for h_idx, side in enumerate(sides):
                if h_idx >= gt_hand_uvd_disp.shape[1]:
                    continue
                
                # Mark which wrists we have
                if side == 'left':
                    has_left_wrist = True
                else:
                    has_right_wrist = True
                
                # GT wrist data
                gt_query_uvd = hand_query_uvd[0, h_idx].cpu().numpy()  # (3,)
                gt_uvd_disp = gt_hand_uvd_disp[0, h_idx].cpu().numpy()  # (T, 3)
                gt_query_rot = hand_query_rot[0, h_idx].cpu().numpy()  # (6,)
                gt_rot_disp = gt_hand_rot_disp[0, h_idx].cpu().numpy()  # (T, 6)
                
                # Denormalize GT
                if norm_stats.has_hand_stats:
                    gt_uvd_disp = norm_stats.denormalize_hand_uvd_disp(gt_uvd_disp)
                    gt_rot_disp = norm_stats.denormalize_hand_rot_disp(gt_rot_disp)
                    gt_query_uvd_denorm = gt_query_uvd.copy()
                    gt_query_uvd_denorm[2] = norm_stats.denormalize_depth(gt_query_uvd[2])
                else:
                    gt_query_uvd_denorm = gt_query_uvd
                
                # Reconstruct GT trajectory and convert to 3D
                gt_uvd_traj = gt_query_uvd_denorm[np.newaxis, :] + gt_uvd_disp  # (T, 3)
                gt_wrist_3d = compute_wrist_3d_from_uvd(gt_query_uvd_denorm, gt_uvd_disp, intrinsics, img_size)
                gt_rotations = compute_wrist_rotations(gt_query_rot, gt_rot_disp)
                
                wrist_data_frames[f'gt_{side}_positions'].append(gt_wrist_3d.astype(np.float32))
                wrist_data_frames[f'gt_{side}_rotations'].append(gt_rotations.astype(np.float32))
                
                # Pred wrist data
                if pred_hand_uvd_disp is not None and pred_hand_rot_disp is not None:
                    pred_uvd_disp = pred_hand_uvd_disp[0, h_idx].cpu().numpy()  # (T, 3)
                    pred_rot_disp = pred_hand_rot_disp[0, h_idx].cpu().numpy()  # (T, 6)
                    
                    # Denormalize pred
                    if norm_stats.has_hand_stats:
                        pred_uvd_disp = norm_stats.denormalize_hand_uvd_disp(pred_uvd_disp)
                        pred_rot_disp = norm_stats.denormalize_hand_rot_disp(pred_rot_disp)
                    
                    # Reconstruct pred trajectory and convert to 3D
                    pred_wrist_3d = compute_wrist_3d_from_uvd(gt_query_uvd_denorm, pred_uvd_disp, intrinsics, img_size)
                    pred_rotations = compute_wrist_rotations(gt_query_rot, pred_rot_disp)
                    
                    wrist_data_frames[f'pred_{side}_positions'].append(pred_wrist_3d.astype(np.float32))
                    wrist_data_frames[f'pred_{side}_rotations'].append(pred_rotations.astype(np.float32))
        
        # Get RGB frame and denormalize from ImageNet normalization
        frame_np = frames[0, -1].cpu().numpy()  # (3, H, W)
        frame_np = frame_np * np.array([0.229, 0.224, 0.225])[:, None, None]
        frame_np = frame_np + np.array([0.485, 0.456, 0.406])[:, None, None]
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        frame_np = frame_np.transpose(1, 2, 0)  # (H, W, 3)
        
        rgb_resized = cv2.resize(frame_np, fixed_size, interpolation=cv2.INTER_AREA)
        rgb_frames.append(rgb_resized)
        
        # Get depth frame (use most recent frame from stack)
        if depth is not None:
            depth_np = depth[0, -1].cpu().numpy()  # (H, W) - use last frame like RGB
            depth_denorm = norm_stats.denormalize_depth(depth_np)
            depth_resized = cv2.resize(depth_denorm, fixed_size, interpolation=cv2.INTER_NEAREST)
            depth_frames.append(depth_resized)
        else:
            depth_frames.append(np.zeros(fixed_size[::-1], dtype=np.float32))
    
    # Stack arrays
    rgb_video = np.stack(rgb_frames)  # (num_frames, H, W, 3)
    depth_video = np.stack(depth_frames).astype(np.float32)  # (num_frames, H, W)
    
    num_frames = len(sample_indices)
    T = gt_trajectories_3d[0].shape[1]
    N = gt_trajectories_3d[0].shape[0]
    
    # Concatenate GT and pred trajectories along point dimension
    # First N points = GT, second N points = predictions
    # This gives them naturally different colors in the visualization
    gt_stacked = np.stack(gt_trajectories_3d, axis=0)  # (num_frames, N, T, 3)
    pred_stacked = np.stack(pred_trajectories_3d, axis=0)  # (num_frames, N, T, 3)
    trajectories = np.concatenate([gt_stacked, pred_stacked], axis=1)  # (num_frames, N*2, T, 3)
    
    # Reshape to (num_frames, T, N*2, 3) for windowed mode
    trajectories = trajectories.transpose(0, 2, 1, 3).astype(np.float32)
    
    # Total points is now doubled (GT + pred)
    total_points = N * 2
    
    # Intrinsics array for all frames
    intrinsics_arr = np.tile(intrinsics_scaled[np.newaxis], (num_frames, 1, 1)).astype(np.float32)
    
    # Identity extrinsics (camera-0 frame is world frame)
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (num_frames, 1, 1))
    
    # Process depth for visualization encoding
    valid_depths = depth_video[depth_video > 0]
    min_depth = float(valid_depths.min()) * 0.8 if len(valid_depths) > 0 else 0.1
    max_depth = float(valid_depths.max()) * 1.5 if len(valid_depths) > 0 else 10.0
    
    depth_normalized = np.clip((depth_video - min_depth) / (max_depth - min_depth), 0, 1)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((num_frames, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    # Build output arrays
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics_arr,
        "extrinsics": extrinsics,
        "inv_extrinsics": np.linalg.inv(extrinsics).astype(np.float32),
        "trajectories": trajectories,
        "cameraZ": np.float64(0.0),
    }
    
    # Add wrist data to arrays if available
    has_gt_left = len(wrist_data_frames['gt_left_positions']) > 0
    has_gt_right = len(wrist_data_frames['gt_right_positions']) > 0
    has_pred_left = len(wrist_data_frames['pred_left_positions']) > 0
    has_pred_right = len(wrist_data_frames['pred_right_positions']) > 0
    
    if has_gt_left:
        gt_left_pos = np.stack(wrist_data_frames['gt_left_positions'], axis=0)  # (num_frames, T, 3)
        gt_left_rot = np.stack(wrist_data_frames['gt_left_rotations'], axis=0)  # (num_frames, T, 3, 3)
        arrays["gt_left_wrist_positions"] = gt_left_pos
        arrays["gt_left_wrist_rot_x"] = gt_left_rot[:, :, :, 0]  # X axis
        arrays["gt_left_wrist_rot_y"] = gt_left_rot[:, :, :, 1]  # Y axis
        arrays["gt_left_wrist_rot_z"] = gt_left_rot[:, :, :, 2]  # Z axis
    
    if has_gt_right:
        gt_right_pos = np.stack(wrist_data_frames['gt_right_positions'], axis=0)
        gt_right_rot = np.stack(wrist_data_frames['gt_right_rotations'], axis=0)
        arrays["gt_right_wrist_positions"] = gt_right_pos
        arrays["gt_right_wrist_rot_x"] = gt_right_rot[:, :, :, 0]
        arrays["gt_right_wrist_rot_y"] = gt_right_rot[:, :, :, 1]
        arrays["gt_right_wrist_rot_z"] = gt_right_rot[:, :, :, 2]
    
    if has_pred_left:
        pred_left_pos = np.stack(wrist_data_frames['pred_left_positions'], axis=0)
        pred_left_rot = np.stack(wrist_data_frames['pred_left_rotations'], axis=0)
        arrays["pred_left_wrist_positions"] = pred_left_pos
        arrays["pred_left_wrist_rot_x"] = pred_left_rot[:, :, :, 0]
        arrays["pred_left_wrist_rot_y"] = pred_left_rot[:, :, :, 1]
        arrays["pred_left_wrist_rot_z"] = pred_left_rot[:, :, :, 2]
    
    if has_pred_right:
        pred_right_pos = np.stack(wrist_data_frames['pred_right_positions'], axis=0)
        pred_right_rot = np.stack(wrist_data_frames['pred_right_rotations'], axis=0)
        arrays["pred_right_wrist_positions"] = pred_right_pos
        arrays["pred_right_wrist_rot_x"] = pred_right_rot[:, :, :, 0]
        arrays["pred_right_wrist_rot_y"] = pred_right_rot[:, :, :, 1]
        arrays["pred_right_wrist_rot_z"] = pred_right_rot[:, :, :, 2]
    
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
        "depthRange": [min_depth, max_depth],
        "totalFrames": num_frames,
        "resolution": list(fixed_size),
        "baseFrameRate": 10,
        "numTrajectoryPoints": total_points,  # GT + pred (N * 2)
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "windowLength": T,
        "stride": 1,
        "windowedMode": True,
        "hasGtLeftWrist": has_gt_left,
        "hasGtRightWrist": has_gt_right,
        "hasPredLeftWrist": has_pred_left,
        "hasPredRightWrist": has_pred_right,
    }
    
    output_file = vis_path / f"{video_name}_data.bin"
    compress_and_write(str(output_file), header, compressed_blob)
    
    print(f"  3D visualization saved: {output_file}")
    if has_gt_left or has_gt_right:
        print(f"    With wrist data: GT left={has_gt_left}, GT right={has_gt_right}, Pred left={has_pred_left}, Pred right={has_pred_right}")
    return 0.0  # Error not computed for 3D visualization


def main():
    parser = argparse.ArgumentParser(description="Generate video visualizations of model predictions")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./data/validation_videos',
                       help='Directory to save output videos/visualizations')
    parser.add_argument('--num_train_videos', type=int, default=2,
                       help='Number of train set videos to visualize')
    parser.add_argument('--num_val_videos', type=int, default=2,
                       help='Number of validation set videos to visualize')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get track type and text_mode from config
    track_type = getattr(cfg.dataset_cfg, 'track_type', '2d')
    text_mode = getattr(cfg.dataset_cfg, 'text_mode', False)
    print(f"Track type: {track_type}")
    print(f"Text mode: {text_mode}")
    
    # Load normalization stats
    stats_path = Path(cfg.dataset_dir) / 'normalization_stats.yaml'
    disp_stats = load_normalization_stats(str(stats_path))
    norm_stats = NormalizationStats(str(stats_path))
    print(f"Loaded normalization statistics from: {stats_path}")
    
    # Set device from config
    device_str = getattr(cfg.training, 'device', 'cuda')
    if 'cuda' in device_str and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Load model
    model, is_diffusion, epoch = load_model(args.checkpoint, cfg, disp_stats, device, text_mode=text_mode)
    model_type = getattr(cfg.model, 'model_type', 'diffusion' if is_diffusion else 'direct')
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean previous visualization files
    print(f"\nCleaning output directory: {output_path}")
    for ext in ['*.mp4', '*.html', '*.bin']:
        for file in output_path.glob(ext):
            try:
                file.unlink()
                print(f"  Deleted: {file.name}")
            except Exception as e:
                print(f"  Warning: Could not delete {file.name}: {e}")
    
    # Create train/val split
    train_files, val_files = create_train_val_split(
        cfg.dataset_dir,
        val_ratio=cfg.dataset_cfg.val_split,
        seed=cfg.dataset_cfg.val_seed
    )
    
    print(f"\nDataset split:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")
    
    # Select random videos
    np.random.seed(42)
    train_videos = np.random.choice(train_files, min(args.num_train_videos, len(train_files)), replace=False)
    val_videos = np.random.choice(val_files, min(args.num_val_videos, len(val_files)), replace=False)
    
    if track_type == '3d':
        # Generate 3D HTML visualizations
        print("\n" + "="*50)
        print("Generating 3D visualizations")
        print("="*50)
        
        # Create viz HTML template
        create_windowed_viz_html(output_path)
        
        video_list = []  # (name, split) pairs
    
    # Process train videos
        print("\nProcessing TRAIN videos")
        for idx, h5_file in enumerate(train_videos):
            video_name = f'train_{idx}_{model_type}_epoch{epoch}'
            generate_video_3d(model, h5_file, video_name, output_path, cfg, norm_stats, is_diffusion, device, text_mode=text_mode)
            video_list.append((video_name, 'train'))
        
        # Process val videos
        print("\nProcessing VALIDATION videos")
        for idx, h5_file in enumerate(val_videos):
            video_name = f'val_{idx}_{model_type}_epoch{epoch}'
            generate_video_3d(model, h5_file, video_name, output_path, cfg, norm_stats, is_diffusion, device, text_mode=text_mode)
            video_list.append((video_name, 'val'))
        
        # Create index and video HTML files
        create_3d_viz_index(output_path, video_list, model_type, epoch)
        for video_name, _ in video_list:
            create_3d_video_html(video_name, video_list, output_path)
        
        print("\n" + "="*50)
        print("SUMMARY - 3D Visualization")
        print("="*50)
        print(f"Model: {model_type} (epoch {epoch})")
        print(f"Train videos: {len(train_videos)}")
        print(f"Val videos: {len(val_videos)}")
        print(f"\nTo view 3D visualizations:")
        print(f"  cd {output_path}")
        print(f"  python -m http.server 8000")
        print(f"  Open: http://localhost:8000/index.html")
        
    else:
        # Generate 2D MP4 videos
        print("\n" + "="*50)
        print("Processing TRAIN videos (2D)")
        print("="*50)
        train_errors = []
        for idx, h5_file in enumerate(train_videos):
            mp4_path = output_path / f'train_{idx}_{model_type}_epoch{epoch}.mp4'
            error = generate_video_2d(model, h5_file, str(mp4_path), cfg, norm_stats, is_diffusion, device, text_mode=text_mode)
            train_errors.append(error)
            
        print("\n" + "="*50)
        print("Processing VALIDATION videos (2D)")
        print("="*50)
        val_errors = []
        for idx, h5_file in enumerate(val_videos):
            mp4_path = output_path / f'val_{idx}_{model_type}_epoch{epoch}.mp4'
            error = generate_video_2d(model, h5_file, str(mp4_path), cfg, norm_stats, is_diffusion, device, text_mode=text_mode)
            val_errors.append(error)
            print("\n" + "="*50)

if __name__ == '__main__':
    main()

