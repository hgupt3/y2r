"""
Main training script for IntentTracker model.

This script trains the model to predict future point trajectories from a single frame,
using mixed precision training, EMA, learning rate scheduling, and W&B logging.
"""

import argparse
import os
import warnings
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

# Suppress xFormers informational warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is available")

from y2r.models.model import IntentTracker
from y2r.models.diffusion_model import DiffusionIntentTracker
from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.dataloaders.split_dataset import create_train_val_split
from y2r.losses import normalized_displacement_loss
from y2r.visualization import visualize_predictions, visualize_diffusion_process


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


def normalize_displacements(disp, mean, std):
    """Normalize displacements using dataset statistics."""
    device = disp.device
    mean_tensor = torch.tensor(mean, device=device, dtype=disp.dtype)
    std_tensor = torch.tensor(std, device=device, dtype=disp.dtype)
    
    disp_normalized = (disp - mean_tensor) / std_tensor
    return disp_normalized


def validate(model, val_loader, disp_stats, device, vis_sample_indices=None, is_diffusion=False, num_inference_steps=10):
    """
    Run validation and return metrics + visualizations.
    
    Args:
        model: IntentTracker or DiffusionIntentTracker model
        val_loader: Validation dataloader
        disp_stats: dict with displacement statistics
        device: torch device
        vis_sample_indices: list of indices to visualize
        is_diffusion: bool, whether model is diffusion-based
        num_inference_steps: int, number of DDIM steps for diffusion models (ignored for direct models)
        
    Returns:
        metrics: dict with 'avg_error'
        vis_data: list of dicts with sample predictions for logging
    """
    model.eval()
    total_error = 0.0
    total_points = 0
    
    mean = disp_stats['displacement_mean']
    std = disp_stats['displacement_std']
    
    # Store specific samples for visualization
    vis_data = []
    
    with torch.no_grad():
        for batch_idx, (imgs, tracks) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            # imgs: (B, frame_stack, C, H, W)
            # tracks: (B, num_track_ts, N, 2) - GT future positions in [0, 1]
            
            B = imgs.shape[0]
            
            # Move to device
            imgs = imgs.to(device)
            tracks = tracks.to(device)
            
            # Extract query coords (initial positions)
            query_coords = tracks[:, 0, :, :]  # (B, N, 2)
            
            # Convert GT positions to displacements
            gt_disp = tracks - query_coords.unsqueeze(1)  # (B, T, N, 2)
            gt_disp = gt_disp.permute(0, 2, 1, 3)  # (B, N, T, 2)
            
            # Normalize GT displacements
            gt_disp_normalized = normalize_displacements(gt_disp, mean, std)
            
            # BATCHED prediction (much faster!)
            if is_diffusion:
                pred_disp = model.predict(imgs, query_coords, num_inference_steps=num_inference_steps)  # (B, N, T, 2)
            else:
                pred_disp = model.predict(imgs, query_coords)  # (B, N, T, 2)
            
            # Compute error (batched)
            loss = normalized_displacement_loss(pred_disp, gt_disp_normalized, std)
            
            total_error += loss.item() * pred_disp.shape[0] * pred_disp.shape[1] * pred_disp.shape[2]
            total_points += pred_disp.shape[0] * pred_disp.shape[1] * pred_disp.shape[2]
            
            # Collect visualization samples (only for specific indices)
            if vis_sample_indices is not None:
                for b in range(B):
                    global_idx = batch_idx * B + b
                    if global_idx in vis_sample_indices:
                        # For diffusion models, re-run prediction with intermediate for this sample
                        if is_diffusion:
                            frame = imgs[b:b+1]
                            qc = query_coords[b:b+1]
                            pred_single, intermediate = model.predict(
                                frame, qc, num_inference_steps=num_inference_steps, return_intermediate=True
                            )
                            vis_dict = {
                                'frame': frame.cpu(),
                                'gt_tracks': tracks[b].cpu(),  # (T, N, 2) - positions
                                'pred_disp': pred_single.cpu(),  # (1, N, T, 2)
                                'query_coords': qc[0].cpu(),
                                'intermediate': [x.cpu() for x in intermediate]
                            }
                        else:
                            vis_dict = {
                                'frame': imgs[b:b+1].cpu(),
                                'gt_tracks': tracks[b].cpu(),  # (T, N, 2) - positions
                                'pred_disp': pred_disp[b:b+1].cpu(),  # (1, N, T, 2)
                                'query_coords': query_coords[b].cpu()
                            }
                        vis_data.append(vis_dict)
    
    metrics = {
        'avg_error': total_error / total_points if total_points > 0 else 0.0,
    }
    
    return metrics, vis_data


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, ema_model, disp_stats, device, cfg, epoch, global_step):
    """
    Train for one epoch.
    
    Returns:
        global_step: updated global step counter
    """
    model.train()
    
    mean = disp_stats['displacement_mean']
    std = disp_stats['displacement_std']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (imgs, tracks) in enumerate(pbar):
        # imgs: (B, frame_stack, C, H, W)
        # tracks: (B, num_track_ts, N, 2) - GT future positions in [0, 1]
        
        B = imgs.shape[0]
        frame_stack = imgs.shape[1]
        
        # Use all frames from stack
        frames = imgs  # (B, frame_stack, C, H, W)
        
        # Convert tracks to displacements
        # tracks: (B, T, N, 2)
        query_coords = tracks[:, 0, :, :]  # (B, N, 2) - initial positions
        
        # Compute GT displacements: disp[t] = pos[t] - pos[0]
        gt_positions = tracks  # (B, T, N, 2)
        gt_disp = gt_positions - query_coords.unsqueeze(1)  # (B, T, N, 2)
        gt_disp = gt_disp.permute(0, 2, 1, 3)  # (B, N, T, 2)
        
        # Normalize GT displacements
        gt_disp_normalized = normalize_displacements(gt_disp, mean, std)
        
        # Move to device
        frames = frames.to(device)
        query_coords = query_coords.to(device)
        gt_disp_normalized = gt_disp_normalized.to(device)
        
        # Prepare batch dict for model
        batch = {
            'frames': frames,
            'query_coords': query_coords,
            'gt_disp_normalized': gt_disp_normalized,
            'disp_std': std
        }
        
        # Forward pass with mixed precision
        with autocast(enabled=cfg.training.use_amp):
            # Use model's compute_loss method
            loss = model.compute_loss(batch)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate (step-based scheduler)
        scheduler.step()
        
        # Update EMA model
        ema_model.update_parameters(model)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})
        
        # Log to W&B
        if global_step % cfg.training.log_every_n_steps == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/grad_norm': grad_norm.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
            }
            wandb.log(log_dict, step=global_step)
        
        global_step += 1
    
    return global_step


def save_checkpoint(model, optimizer, scheduler, ema_model, epoch, best_val_error, checkpoint_dir, filename):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'best_val_error': best_val_error,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train IntentTracker model")
    parser.add_argument(
        "--config",
        type=str,
        default="train_cfg.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped checkpoint directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = os.path.join(cfg.training.checkpoint_dir, timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Initialize W&B
    wandb.init(
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        config={
            'model': vars(cfg.model),
            'training': vars(cfg.training),
            'dataset': vars(cfg.dataset_cfg),
        },
        name=timestamp  # Use timestamp as run name
    )
    
    # Load displacement statistics from H5 directory
    stats_path = Path(cfg.dataset_dir) / 'normalization_stats.yaml'
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found at {stats_path}. "
            f"Make sure you've run create_h5_dataset.py to generate the H5 dataset and statistics."
        )
    
    disp_stats = load_normalization_stats(str(stats_path))
    print(f"Loaded displacement statistics from: {stats_path}")
    print(f"  Mean: {disp_stats['displacement_mean']}")
    print(f"  Std: {disp_stats['displacement_std']}")
    
    # Create train/val split
    train_files, val_files = create_train_val_split(
        cfg.dataset_dir,
        val_ratio=cfg.training.val_split,
        seed=cfg.training.val_seed
    )
    
    # Create datasets
    train_dataset = TrackDataset(
        h5_files=train_files,
        img_size=cfg.dataset_cfg.img_size,
        frame_stack=cfg.dataset_cfg.frame_stack,
        num_track_ts=cfg.dataset_cfg.num_track_ts,
        num_track_ids=cfg.dataset_cfg.num_track_ids,
        downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
        aug_prob=cfg.training.aug_prob,
        cache_all=cfg.dataset_cfg.cache_all,
        cache_image=cfg.dataset_cfg.cache_image,
        # Augmentation parameters (compact format)
        aug_color_jitter=getattr(cfg.dataset_cfg, 'aug_color_jitter', None),
        aug_translation_px=getattr(cfg.dataset_cfg, 'aug_translation_px', 0),
        aug_rotation_deg=getattr(cfg.dataset_cfg, 'aug_rotation_deg', 0),
        aug_hflip_prob=getattr(cfg.dataset_cfg, 'aug_hflip_prob', 0.0),
        aug_vflip_prob=getattr(cfg.dataset_cfg, 'aug_vflip_prob', 0.0),
        aug_noise_std=getattr(cfg.dataset_cfg, 'aug_noise_std', 0.0)
    )
    
    val_dataset = TrackDataset(
        h5_files=val_files,
        img_size=cfg.dataset_cfg.img_size,
        frame_stack=cfg.dataset_cfg.frame_stack,
        num_track_ts=cfg.dataset_cfg.num_track_ts,
        num_track_ids=cfg.dataset_cfg.num_track_ids,
        downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
        aug_prob=0.0,  # No augmentation for validation
        cache_all=cfg.dataset_cfg.cache_all,
        cache_image=cfg.dataset_cfg.cache_image
    )
    
    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.training.num_workers > 0 else False,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,  # Use same batch size as training
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.training.num_workers > 0 else False,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )
    
    # Select fixed samples for visualization
    num_val_samples = len(val_dataset)
    vis_sample_indices = np.linspace(0, num_val_samples-1, cfg.training.val_vis_samples, dtype=int).tolist()
    print(f"Visualization samples: {vis_sample_indices}")
    
    # Create model (support both direct and diffusion models)
    model_type = getattr(cfg.model, 'model_type', 'direct')
    is_diffusion = (model_type == 'diffusion')
    
    if is_diffusion:
        print(f"Creating DiffusionIntentTracker model...")
        model = DiffusionIntentTracker(
            num_future_steps=cfg.model.num_future_steps,
            hidden_size=cfg.model.hidden_size,
            model_resolution=cfg.model.model_resolution,
            add_space_attn=cfg.model.add_space_attn,
            vit_model_name=cfg.model.vit_model_name,
            vit_frozen=cfg.model.vit_frozen,
            time_depth=cfg.model.time_depth,
            space_depth=cfg.model.space_depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            p_drop_attn=cfg.model.p_drop_attn,
            frame_stack=cfg.model.frame_stack,
            num_diffusion_steps=getattr(cfg.model, 'num_diffusion_steps', 100),
            beta_schedule=getattr(cfg.model, 'beta_schedule', 'squaredcos_cap_v2'),
            cache_quantized_position_encoding=getattr(cfg.model, 'cache_quantized_position_encoding', False),
            disp_mean=disp_stats['displacement_mean'],
            disp_std=disp_stats['displacement_std'],
        ).to(device)
    else:
        print(f"Creating IntentTracker model (direct prediction)...")
        model = IntentTracker(
            num_future_steps=cfg.model.num_future_steps,
            hidden_size=cfg.model.hidden_size,
            model_resolution=cfg.model.model_resolution,
            add_space_attn=cfg.model.add_space_attn,
            vit_model_name=cfg.model.vit_model_name,
            vit_frozen=cfg.model.vit_frozen,
            time_depth=cfg.model.time_depth,
            space_depth=cfg.model.space_depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            p_drop_attn=cfg.model.p_drop_attn,
            frame_stack=cfg.model.frame_stack,
            cache_quantized_position_encoding=getattr(cfg.model, 'cache_quantized_position_encoding', False),
        ).to(device)
    
    # Compile model if requested
    if cfg.training.use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    
    # Create learning rate scheduler (warmup + cosine annealing)
    # Steps per epoch for step-based scheduling
    steps_per_epoch = len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.num_epochs * steps_per_epoch
    
    # Warmup scheduler: linear warmup from 0 to base_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6 / cfg.training.lr,  # Start from very small LR
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing scheduler: decay from base_lr to min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=cfg.training.min_lr
    )
    
    # Combine warmup and cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Create EMA using PyTorch's AveragedModel
    # Note: AveragedModel uses avg_fn to specify the averaging strategy
    def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        decay = cfg.training.ema_decay
        return decay * averaged_model_parameter + (1 - decay) * model_parameter
    
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=cfg.training.use_amp)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_error = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_error = checkpoint['best_val_error']
        print(f"Resumed from epoch {start_epoch}, best val error: {best_val_error:.4f}")
    
    # Training loop
    global_step = 0
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        print(f"{'='*50}")
        
        # Train (scheduler.step() is called per batch inside train_one_epoch)
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema_model, 
            disp_stats, device, cfg, epoch, global_step
        )
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Validation
        if (epoch + 1) % cfg.training.val_every_n_epochs == 0:
            print("\nRunning validation...")
            
            # Use EMA model for validation
            # Note: ema_model.module gives us the averaged model
            num_inference_steps = getattr(cfg.model, 'num_inference_steps', 10)
            metrics, vis_data = validate(
                ema_model.module, val_loader, disp_stats, device, vis_sample_indices, is_diffusion, num_inference_steps
            )
            
            print(f"Validation avg error: {metrics['avg_error']:.4f}")
            
            # Log metrics
            wandb.log({
                'val/avg_error': metrics['avg_error'],
                'epoch': epoch + 1,
            }, step=global_step)
            
            # Log visualizations
            if len(vis_data) > 0:
                images = visualize_predictions(vis_data, disp_stats, epoch + 1)
                wandb.log({'val/predictions': images}, step=global_step)
                
                # For diffusion models, also visualize the denoising process
                if is_diffusion:
                    # Only visualize first few samples to avoid clutter
                    diffusion_vis_samples = vis_data[:min(3, len(vis_data))]
                    diffusion_images = visualize_diffusion_process(
                        diffusion_vis_samples, disp_stats, epoch + 1
                    )
                    wandb.log({'val/diffusion_process': diffusion_images}, step=global_step)
            
            # Save checkpoints
            # Always save latest checkpoint
            save_checkpoint(
                model, optimizer, scheduler, ema_model, epoch, best_val_error,
                run_checkpoint_dir, 'latest.pth'
            )
            
            # Save best checkpoint if this is a new best
            if metrics['avg_error'] < best_val_error:
                best_val_error = metrics['avg_error']
                save_checkpoint(
                    model, optimizer, scheduler, ema_model, epoch, best_val_error,
                    run_checkpoint_dir, 'best.pth'
                )
                wandb.run.summary["best_val_error"] = best_val_error
                print(f"New best model saved! Val error: {best_val_error:.4f}")
    
    print("\nTraining completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
