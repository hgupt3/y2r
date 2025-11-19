"""
Video validation script for IntentTracker models.

Generates MP4 videos showing model predictions overlaid on video frames.
"""

import argparse
import os
import yaml
from pathlib import Path
import numpy as np
import torch
import h5py
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from y2r.models.model import IntentTracker
from y2r.models.diffusion_model import DiffusionIntentTracker
from y2r.models.autoreg_model import AutoregressiveIntentTracker
from y2r.dataloaders.split_dataset import create_train_val_split


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
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


def denormalize_displacements(disp_normalized, mean, std):
    """Denormalize displacements using dataset statistics."""
    disp = disp_normalized * std + mean
    return disp


def load_model(checkpoint_path, cfg, disp_stats, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load to CPU first to avoid CUDA memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine model type
    model_type = getattr(cfg.model, 'model_type', 'direct')
    is_diffusion = (model_type == 'diffusion')
    
    if is_diffusion:
        print("Loading DiffusionIntentTracker model...")
        model = DiffusionIntentTracker(
            num_future_steps=getattr(cfg.model, 'num_future_steps', 8),
            hidden_size=getattr(cfg.model, 'hidden_size', 384),
            model_resolution=tuple(getattr(cfg.model, 'model_resolution', [224, 224])),
            add_space_attn=getattr(cfg.model, 'add_space_attn', True),
            vit_model_name=getattr(cfg.model, 'vit_model_name', 'dinov2_vits14'),
            vit_frozen=getattr(cfg.model, 'vit_frozen', False),
            time_depth=getattr(cfg.model, 'time_depth', 6),
            space_depth=getattr(cfg.model, 'space_depth', 3),
            num_heads=getattr(cfg.model, 'num_heads', 8),
            mlp_ratio=getattr(cfg.model, 'mlp_ratio', 4.0),
            p_drop_attn=getattr(cfg.model, 'p_drop_attn', 0.0),
            frame_stack=getattr(cfg.model, 'frame_stack', 1),
            num_diffusion_steps=getattr(cfg.model, 'num_diffusion_steps', 100),
            beta_schedule=getattr(cfg.model, 'beta_schedule', 'squaredcos_cap_v2'),
            cache_quantized_position_encoding=getattr(cfg.model, 'cache_quantized_position_encoding', False),
            disp_mean=disp_stats['displacement_mean'],
            disp_std=disp_stats['displacement_std'],
        ).to(device)
    elif model_type == 'autoreg':
        print("Loading AutoregressiveIntentTracker model...")
        model = AutoregressiveIntentTracker(
            num_future_steps=getattr(cfg.model, 'num_future_steps', 10),
            hidden_size=getattr(cfg.model, 'hidden_size', 384),
            model_resolution=tuple(getattr(cfg.model, 'model_resolution', [224, 224])),
            add_space_attn=getattr(cfg.model, 'add_space_attn', True),
            vit_model_name=getattr(cfg.model, 'vit_model_name', 'dinov2_vits14'),
            vit_frozen=getattr(cfg.model, 'vit_frozen', False),
            time_depth=getattr(cfg.model, 'time_depth', 6),
            num_heads=getattr(cfg.model, 'num_heads', 8),
            mlp_ratio=getattr(cfg.model, 'mlp_ratio', 4.0),
            p_drop_attn=getattr(cfg.model, 'p_drop_attn', 0.0),
            frame_stack=getattr(cfg.model, 'frame_stack', 1),
            cache_quantized_position_encoding=getattr(cfg.model, 'cache_quantized_position_encoding', False),
        ).to(device)
    else:
        print("Loading IntentTracker model (direct prediction)...")
        model = IntentTracker(
            num_future_steps=getattr(cfg.model, 'num_future_steps', 10),
            hidden_size=getattr(cfg.model, 'hidden_size', 384),
            model_resolution=tuple(getattr(cfg.model, 'model_resolution', [224, 224])),
            add_space_attn=getattr(cfg.model, 'add_space_attn', True),
            vit_model_name=getattr(cfg.model, 'vit_model_name', 'dinov2_vits14'),
            vit_frozen=getattr(cfg.model, 'vit_frozen', False),
            time_depth=getattr(cfg.model, 'time_depth', 6),
            space_depth=getattr(cfg.model, 'space_depth', 3),
            num_heads=getattr(cfg.model, 'num_heads', 8),
            mlp_ratio=getattr(cfg.model, 'mlp_ratio', 4.0),
            p_drop_attn=getattr(cfg.model, 'p_drop_attn', 0.0),
            frame_stack=getattr(cfg.model, 'frame_stack', 1),
            cache_quantized_position_encoding=getattr(cfg.model, 'cache_quantized_position_encoding', False),
        ).to(device)
    
    # Load state dict (handle both EMA and regular checkpoints)
    if 'ema_model_state_dict' in checkpoint:
        # Use EMA model for validation (better quality)
        state_dict = checkpoint['ema_model_state_dict']
        # EMA from torch.optim.swa_utils.AveragedModel wraps with 'module.'
        # Remove 'module.' prefix if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Loaded EMA model state")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded regular model state")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model state directly")
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Checkpoint epoch: {epoch}")
    
    return model, is_diffusion, epoch


def normalize_frame_for_model(frame):
    """
    Normalize frame for model input.
    
    Args:
        frame: (H, W, 3) RGB frame in [0, 255]
    
    Returns:
        normalized: (3, H, W) ImageNet normalized frame
    """
    # Convert to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # (H, W, 3) -> (3, H, W)
    frame = frame.transpose(2, 0, 1)
    frame = (frame - mean) / std
    
    return frame


def generate_video(model, h5_file, output_path, cfg, disp_stats, is_diffusion, 
                   stride=1, max_points=16, device='cuda'):
    """
    Generate video with model predictions overlaid.
    
    Args:
        model: trained model
        h5_file: path to HDF5 file
        output_path: where to save MP4
        cfg: configuration namespace
        disp_stats: displacement statistics
        is_diffusion: whether model is diffusion-based
        stride: frame stride for inference
        max_points: max trajectory points to visualize
        device: torch device
    """
    print(f"\nGenerating video for: {h5_file}")
    print(f"Output: {output_path}")
    
    # Load H5 file
    with h5py.File(h5_file, 'r') as f:
        video = f['root']['video'][:]  # (T, C, H, W)
        # Transpose to (T, H, W, C) for visualization
        video = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
        num_frames = video.shape[0]
        
        print(f"Video shape: {video.shape}")
        print(f"Processing every {stride} frames")
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    frame_size = (512, 512)  # Upscale for better viewing
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Statistics
    errors = []
    
    # Get stats for denormalization
    mean = np.array(disp_stats['displacement_mean'])
    std = np.array(disp_stats['displacement_std'])
    
    # Process frames
    frame_stack = cfg.model.frame_stack
    num_future_steps = cfg.model.num_future_steps
    downsample_factor = getattr(cfg.dataset_cfg, 'downsample_factor', 1)
    
    # Get available track frame indices from H5 file
    with h5py.File(h5_file, 'r') as f:
        available_track_keys = sorted([k for k in f['root']['tracks'].keys() if k.startswith('frame_')])
        available_frame_indices = [int(k.split('_')[1]) for k in available_track_keys]
    
    print(f"Found {len(available_frame_indices)} frames with tracks (downsample_factor={downsample_factor})")
    
    for frame_idx in tqdm(available_frame_indices[::stride], desc="Processing frames"):
        # Skip if we're too close to the end
        if frame_idx + num_future_steps * downsample_factor >= num_frames:
            continue
            
        # Load frame stack for model
        start_idx = max(0, frame_idx - (frame_stack - 1))
        frames_raw = video[start_idx:frame_idx+1]  # (T_obs, H, W, 3)
        
        # Pad if needed at start of video
        if len(frames_raw) < frame_stack:
            padding = [frames_raw[0]] * (frame_stack - len(frames_raw))
            frames_raw = np.concatenate([np.stack(padding), frames_raw], axis=0)
        
        # Normalize frames for model
        frames_normalized = np.stack([normalize_frame_for_model(f) for f in frames_raw])  # (T_obs, 3, H, W)
        frames_tensor = torch.from_numpy(frames_normalized).float().unsqueeze(0).to(device)  # (1, T_obs, 3, H, W)
        
        # Load GT tracks at this frame
        track_key = f"frame_{frame_idx:04d}"
        with h5py.File(h5_file, 'r') as f:
            if track_key not in f['root']['tracks']:
                continue
            gt_tracks = f['root']['tracks'][track_key][:]  # (T, N, 2) in [0, 1]
        
        # Sample points (uniformly or randomly)
        num_points = gt_tracks.shape[1]
        if num_points > max_points:
            indices = np.linspace(0, num_points - 1, max_points, dtype=int)
            gt_tracks = gt_tracks[:, indices, :]
        
        # Extract query coordinates (initial positions)
        query_coords = gt_tracks[0:1, :, :]  # (1, N, 2) in [0, 1]
        query_tensor = torch.from_numpy(query_coords).float().to(device)
        
        # GT displacements
        gt_disp = gt_tracks - query_coords  # (T, N, 2)
        gt_disp = gt_disp.transpose(1, 0, 2)  # (N, T, 2)
        
        # Run model prediction
        with torch.no_grad():
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                pred_disp = model.predict(frames_tensor, query_tensor, num_inference_steps=num_inference_steps)
            else:
                pred_disp = model.predict(frames_tensor, query_tensor)
        
        pred_disp = pred_disp[0].cpu().numpy()  # (N, T_pred, 2) normalized
        
        # Truncate GT to match prediction length
        T_pred = pred_disp.shape[1]
        gt_disp = gt_disp[:, :T_pred, :]  # (N, T_pred, 2)
        
        # Denormalize predictions
        pred_disp_denorm = pred_disp * std + mean  # (N, T_pred, 2) in [0, 1] space
        gt_disp_denorm = gt_disp * std + mean  # (N, T_pred, 2) in [0, 1] space
        
        # Compute error
        error = np.linalg.norm(pred_disp_denorm - gt_disp_denorm, axis=-1).mean()
        errors.append(error)
        
        # Visualize
        vis_frame = visualize_trajectories(
            frames_raw[-1],  # Current frame (last in stack)
            query_coords[0],  # (N, 2)
            pred_disp_denorm,  # (N, T, 2)
            gt_disp_denorm,    # (N, T, 2)
            frame_idx,
            error,
            frame_size
        )
        
        # Write frame
        out.write(vis_frame)
    
    out.release()
    
    avg_error = np.mean(errors) if errors else 0.0
    print(f"Average error: {avg_error:.4f}")
    print(f"Video saved to: {output_path}")
    
    return avg_error


def _draw_gradient_trajectory(ax, traj_x, traj_y, cmap, linewidth=3, alpha=0.85):
    """
    Draw a single trajectory with proper gradient coloring using LineCollection.
    
    Args:
        ax: matplotlib axis
        traj_x: x coordinates array
        traj_y: y coordinates array
        cmap: colormap to use
        linewidth: line width
        alpha: transparency
    """
    # Create line segments
    points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection with gradient colors
    T = len(traj_x)
    colors = np.linspace(0, 1, T-1)  # Color values from 0 (light) to 1 (dark)
    
    lc = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=linewidth, capstyle='round')
    lc.set_array(colors)
    ax.add_collection(lc)


def visualize_trajectories(frame, query_coords, pred_disp, gt_disp, frame_idx, error, output_size=(512, 512)):
    """
    Visualize trajectories on frame.
    
    Args:
        frame: (H, W, 3) RGB frame in [0, 255]
        query_coords: (N, 2) query positions in [0, 1]
        pred_disp: (N, T, 2) predicted displacements in [0, 1] space
        gt_disp: (N, T, 2) GT displacements in [0, 1] space
        frame_idx: current frame index
        error: current error metric
        output_size: output frame size (H, W)
    
    Returns:
        vis_frame: (H, W, 3) RGB visualization in [0, 255]
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=64)
    
    # Show frame
    ax.imshow(frame)
    ax.axis('off')
    
    H, W = frame.shape[:2]
    N, T, _ = pred_disp.shape
    
    # Create custom colormaps (light to dark for time progression)
    # GT: Light green → Dark green
    gt_cmap = LinearSegmentedColormap.from_list(
        'gt_gradient', 
        ['#90EE90', '#7FD87F', '#50C850', '#228B22', '#006400']  # light green → dark green
    )
    
    # Pred: Light cyan → Dark blue
    pred_cmap = LinearSegmentedColormap.from_list(
        'pred_gradient',
        ['#B0E0E6', '#87CEEB', '#00BFFF', '#1E90FF', '#0000CD']  # light cyan → dark blue
    )
    
    # Draw trajectories
    for i in range(N):
        # Query point
        qx, qy = query_coords[i] * np.array([W, H])
        ax.plot(qx, qy, 'go', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        # GT trajectory with gradient
        gt_positions = query_coords[i:i+1] + gt_disp[i]  # (T, 2)
        gt_px = gt_positions[:, 0] * W
        gt_py = gt_positions[:, 1] * H
        _draw_gradient_trajectory(ax, gt_px, gt_py, gt_cmap, linewidth=3, alpha=0.85)
        
        # Predicted trajectory with gradient
        pred_positions = query_coords[i:i+1] + pred_disp[i]  # (T, 2)
        pred_px = pred_positions[:, 0] * W
        pred_py = pred_positions[:, 1] * H
        _draw_gradient_trajectory(ax, pred_px, pred_py, pred_cmap, linewidth=3, alpha=0.85)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#50C850', linewidth=3, alpha=0.85, label='GT'),
        Line2D([0], [0], color='#00BFFF', linewidth=3, alpha=0.85, label='Pred'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add text overlay
    text_str = f"Frame: {frame_idx}\nError: {error:.4f}"
    ax.text(10, 20, text_str, color='white', fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    vis_img = np.asarray(buf)[:, :, :3]  # Drop alpha
    plt.close(fig)
    
    # Resize to output size
    vis_img = cv2.resize(vis_img, output_size)
    
    # Convert RGB to BGR for OpenCV
    vis_frame = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    return vis_frame


def main():
    parser = argparse.ArgumentParser(description="Generate video visualizations of model predictions")
    parser.add_argument('--config', type=str, default='train_cfg_diffusion.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./validation_videos',
                       help='Directory to save output videos')
    parser.add_argument('--num_train_videos', type=int, default=2,
                       help='Number of train set videos to visualize')
    parser.add_argument('--num_val_videos', type=int, default=2,
                       help='Number of validation set videos to visualize')
    parser.add_argument('--stride', type=int, default=1,
                       help='Frame stride for inference')
    parser.add_argument('--max_points', type=int, default=16,
                       help='Max trajectory points to visualize per frame')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for inference (overrides config if specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Load normalization stats
    stats_path = Path(cfg.dataset_dir) / 'normalization_stats.yaml'
    disp_stats = load_normalization_stats(str(stats_path))
    print(f"Loaded displacement statistics:")
    print(f"  Mean: {disp_stats['displacement_mean']}")
    print(f"  Std: {disp_stats['displacement_std']}")
    
    # Set device (command line arg overrides config)
    if args.device is not None:
        device_str = args.device
    else:
        device_str = getattr(cfg.training, 'device', 'cuda')
    
    if 'cuda' in device_str and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Load model
    model, is_diffusion, epoch = load_model(args.checkpoint, cfg, disp_stats, device)
    model_type = 'diffusion' if is_diffusion else 'direct'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create train/val split
    train_files, val_files = create_train_val_split(
        cfg.dataset_dir,
        val_ratio=cfg.training.val_split,
        seed=cfg.training.val_seed
    )
    
    print(f"\nDataset split:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")
    
    # Select random videos
    np.random.seed(42)  # For reproducibility
    train_videos = np.random.choice(train_files, min(args.num_train_videos, len(train_files)), replace=False)
    val_videos = np.random.choice(val_files, min(args.num_val_videos, len(val_files)), replace=False)
    
    # Process train videos
    print("\n" + "="*50)
    print("Processing TRAIN videos")
    print("="*50)
    train_errors = []
    for idx, h5_file in enumerate(train_videos):
        output_path = os.path.join(args.output_dir, f'train_{idx}_{model_type}_epoch{epoch}.mp4')
        error = generate_video(model, h5_file, output_path, cfg, disp_stats, is_diffusion,
                              stride=args.stride, max_points=args.max_points, device=device)
        train_errors.append(error)
    
    # Process val videos
    print("\n" + "="*50)
    print("Processing VALIDATION videos")
    print("="*50)
    val_errors = []
    for idx, h5_file in enumerate(val_videos):
        output_path = os.path.join(args.output_dir, f'val_{idx}_{model_type}_epoch{epoch}.mp4')
        error = generate_video(model, h5_file, output_path, cfg, disp_stats, is_diffusion,
                              stride=args.stride, max_points=args.max_points, device=device)
        val_errors.append(error)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Model: {model_type} (epoch {epoch})")
    print(f"Train videos: {args.num_train_videos}")
    print(f"  Avg error: {np.mean(train_errors):.4f}")
    print(f"Val videos: {args.num_val_videos}")
    print(f"  Avg error: {np.mean(val_errors):.4f}")
    print(f"\nVideos saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

