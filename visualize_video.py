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


def load_model(checkpoint_path, cfg, disp_stats, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load to CPU first to avoid CUDA memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model using factory
    model_type = getattr(cfg.model, 'model_type', 'direct')
    is_diffusion = (model_type == 'diffusion')
    
    model = create_model(cfg, disp_stats=disp_stats, device=device)
    
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


def generate_video_2d(model, h5_file, output_path, cfg, norm_stats, is_diffusion, device):
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
    """
    print(f"\nGenerating 2D video for: {h5_file}")
    print(f"Output: {output_path}")
    
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
        
        # Run model prediction (use autocast for xformers compatibility on new GPUs)
        with torch.no_grad(), autocast(device_type=device.type, enabled=True):
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                pred_disp = model.predict(frames, query_coords, num_inference_steps=num_inference_steps)
            else:
                pred_disp = model.predict(frames, query_coords)
        
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
        vis_img = buf[:, :, :3]
        plt.close(fig)
        
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


def generate_video_3d(model, h5_file, video_name, vis_path, cfg, norm_stats, is_diffusion, device):
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
    """
    print(f"\nGenerating 3D visualization for: {h5_file}")
    
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
    )
    
    # Use ALL samples from the video (respecting dataset's natural frame ordering)
    num_samples = len(dataset)
    sample_indices = list(range(num_samples))
    
    print(f"  Dataset length: {len(dataset)} frames, visualizing all {num_samples}")
    
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
        
        # Run model prediction (use autocast for xformers compatibility on new GPUs)
        with torch.no_grad(), autocast(device_type=device.type, enabled=True):
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                pred_disp = model.predict(frames, query_coords, depth=model_depth, 
                                         num_inference_steps=num_inference_steps)
            else:
                pred_disp = model.predict(frames, query_coords, depth=model_depth)
        
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
    }
    
    output_file = vis_path / f"{video_name}_data.bin"
    compress_and_write(str(output_file), header, compressed_blob)
    
    print(f"  3D visualization saved: {output_file}")
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
    
    # Get track type from config
    track_type = getattr(cfg.dataset_cfg, 'track_type', '2d')
    print(f"Track type: {track_type}")
    
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
    model, is_diffusion, epoch = load_model(args.checkpoint, cfg, disp_stats, device)
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
            generate_video_3d(model, h5_file, video_name, output_path, cfg, norm_stats, is_diffusion, device)
            video_list.append((video_name, 'train'))
        
        # Process val videos
        print("\nProcessing VALIDATION videos")
        for idx, h5_file in enumerate(val_videos):
            video_name = f'val_{idx}_{model_type}_epoch{epoch}'
            generate_video_3d(model, h5_file, video_name, output_path, cfg, norm_stats, is_diffusion, device)
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
            error = generate_video_2d(model, h5_file, str(mp4_path), cfg, norm_stats, is_diffusion, device)
            train_errors.append(error)
        
        print("\n" + "="*50)
        print("Processing VALIDATION videos (2D)")
        print("="*50)
        val_errors = []
        for idx, h5_file in enumerate(val_videos):
            mp4_path = output_path / f'val_{idx}_{model_type}_epoch{epoch}.mp4'
            error = generate_video_2d(model, h5_file, str(mp4_path), cfg, norm_stats, is_diffusion, device)
            val_errors.append(error)
        
        print("\n" + "="*50)
        print("SUMMARY - 2D Videos")
        print("="*50)
        print(f"Model: {model_type} (epoch {epoch})")
        print(f"Train videos: {len(train_videos)}")
        print(f"  Avg error: {np.mean(train_errors):.4f}")
        print(f"Val videos: {len(val_videos)}")
        print(f"  Avg error: {np.mean(val_errors):.4f}")
        print(f"\nVideos saved to: {output_path}")


if __name__ == '__main__':
    main()

