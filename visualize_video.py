"""
Video validation script for IntentTracker models.

Generates MP4 videos showing model predictions overlaid on video frames.
"""

import argparse
import os
import warnings
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Suppress xFormers informational warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is available")

from y2r.models.factory import create_model
from y2r.dataloaders.split_dataset import create_train_val_split
from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.visualization import visualize_tracks_on_frame, denormalize_displacements


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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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


def generate_video(model, h5_file, output_path, cfg, disp_stats, is_diffusion, 
                   stride=1, max_points=16, device='cuda'):
    """
    Generate video with model predictions overlaid using TrackDataset.
    
    Args:
        model: trained model
        h5_file: path to HDF5 file
        output_path: where to save MP4
        cfg: configuration namespace
        disp_stats: displacement statistics
        is_diffusion: whether model is diffusion-based
        stride: frame stride for inference (sample every Nth frame)
        max_points: max trajectory points to visualize (ignored - dataset handles sampling)
        device: torch device
    """
    print(f"\nGenerating video for: {h5_file}")
    print(f"Output: {output_path}")
    
    # Create dataset for this single H5 file (no augmentation)
    dataset = TrackDataset(
        h5_files=[h5_file],
        img_size=cfg.dataset_cfg.img_size,
        frame_stack=cfg.dataset_cfg.frame_stack,
        num_track_ts=cfg.dataset_cfg.num_track_ts,
        num_track_ids=cfg.dataset_cfg.num_track_ids,
        downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
        aug_prob=0.0,  # No augmentation for visualization
        cache_all=cfg.dataset_cfg.cache_all,
        cache_image=cfg.dataset_cfg.cache_image
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"Dataset length: {len(dataset)} frames")
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
    
    # Iterate through dataset with stride using the exact structure from train.py
    for sample_idx, (imgs, tracks) in enumerate(tqdm(dataloader, desc="Processing frames")):
        if sample_idx % stride != 0:
            continue
        
        # imgs: (1, frame_stack, 3, H, W), tracks: (1, num_track_ts, N, 2)
        imgs = imgs.to(device)
        tracks = tracks.to(device)
        
        # Extract query coords (match train.py validation logic exactly)
        query_coords = tracks[:, 0, :, :]  # (1, N, 2)
        
        # Run model prediction (exactly like train.py)
        with torch.no_grad():
            if is_diffusion:
                num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
                pred_disp = model.predict(imgs, query_coords, num_inference_steps=num_inference_steps)
            else:
                pred_disp = model.predict(imgs, query_coords)
        
        # Use last frame for visualization (match train.py)
        frame = imgs[0, -1]  # (3, H, W) tensor
        gt_tracks_data = tracks[0]  # (T, N, 2) - positions
        pred_disp_single = pred_disp[0]  # (N, T, 2) - displacements (normalized)
        query_coords_single = query_coords[0]  # (N, 2)
        
        # GT tracks are already positions in [0, 1] (match visualization.py)
        gt_tracks_viz = gt_tracks_data.permute(1, 0, 2)  # (N, T, 2)
        
        # Denormalize predicted displacements (match visualization.py)
        pred_disp_denorm = denormalize_displacements(pred_disp_single, mean, std)
        
        # Convert displacements to positions (match visualization.py)
        pred_tracks_viz = query_coords_single.unsqueeze(1) + pred_disp_denorm  # (N, T, 2)
        
        # Create visualization using SAME function as training
        fig = visualize_tracks_on_frame(
            frame=frame,
            query_coords=query_coords_single,
            gt_tracks=gt_tracks_viz,
            pred_tracks=pred_tracks_viz,
            title=f"Frame {sample_idx}"
        )
        
        # Convert matplotlib figure to OpenCV frame
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        vis_img = buf[:, :, :3]
        plt.close(fig)
        
        # Resize to output size and convert RGB to BGR for OpenCV
        vis_frame = cv2.resize(vis_img, frame_size)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Compute error for logging
        gt_disp = gt_tracks_data - query_coords_single.unsqueeze(0)  # (T, N, 2)
        gt_disp = gt_disp.permute(1, 0, 2)  # (N, T, 2)
        error = torch.norm(pred_disp_denorm - gt_disp, dim=-1).mean().item()
        errors.append(error)
        
        # Write frame
        out.write(vis_frame)
    
    out.release()
    
    avg_error = np.mean(errors) if errors else 0.0
    print(f"Average error: {avg_error:.4f}")
    print(f"Video saved to: {output_path}")
    
    return avg_error


def main():
    parser = argparse.ArgumentParser(description="Generate video visualizations of model predictions")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./data/validation_videos',
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
    
    # Delete previous visualization videos in output directory
    print(f"\nCleaning output directory: {args.output_dir}")
    for file in Path(args.output_dir).glob('*.mp4'):
        try:
            file.unlink()
            print(f"  Deleted: {file.name}")
        except Exception as e:
            print(f"  Warning: Could not delete {file.name}: {e}")
    
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

