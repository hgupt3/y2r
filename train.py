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
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

# Suppress xFormers informational warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is available")

from y2r.models.factory import create_model
from y2r.dataloaders.track_dataloader import TrackDataset
from y2r.dataloaders.split_dataset import create_train_val_split, create_sample_split
from y2r.dataloaders.gpu_augmentations import GPUAugmentations, create_gpu_augmentations
from y2r.visualization import visualize_predictions, visualize_diffusion_process


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


def validate(model, val_loader, device, track_type='2d', vis_sample_indices=None, is_diffusion=False, num_inference_steps=10, hand_mode=None, text_mode=False):
    """
    Run validation and return metrics + visualizations.
    
    Args:
        model: IntentTracker or DiffusionIntentTracker model
        val_loader: Validation dataloader (returns dict format)
        device: torch device
        track_type: "2d" or "3d" - determines whether depth is used
        vis_sample_indices: list of indices to visualize
        is_diffusion: bool, whether model is diffusion-based
        num_inference_steps: int, number of DDIM steps for diffusion models (ignored for direct models)
        hand_mode: "left", "right", "both", or None - determines hand data loading
        text_mode: bool - whether to use text conditioning
        
    Returns:
        metrics: dict with 'avg_error' and optionally hand errors
        vis_data: list of dicts with sample predictions for logging
    """
    model.eval()
    total_track_error = 0.0
    total_track_points = 0
    total_hand_uvd_error = 0.0
    total_hand_rot_error = 0.0
    total_hand_points = 0
    
    # Store specific samples for visualization
    vis_data = []
    
    with torch.no_grad(), autocast(device_type=device.type, enabled=True):  # Need autocast for xformers on new GPUs
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            # Unpack dict batch (dataloader returns normalized data)
            frames = batch['imgs'].to(device)  # (B, frame_stack, C, H, W)
            query_coords = batch['query_coords'].to(device)  # (B, N, 2) or (B, N, 3)
            displacements = batch['displacements'].to(device)  # (B, T, N, coord_dim)
            depth = batch.get('depth')
            if depth is not None:
                depth = depth.to(device)
            
            B = frames.shape[0]
            
            # Permute displacements: (B, T, N, D) -> (B, N, T, D)
            gt_disp_normalized = displacements.permute(0, 2, 1, 3)
            
            # Prepare depth for model (if 3D mode)
            model_depth = None
            if track_type == '3d' and depth is not None:
                # Model expects (B, frame_stack, 1, H, W)
                model_depth = depth.unsqueeze(2)  # (B, frame_stack, 1, H, W)
            
            # Prepare hand data if hand_mode is set
            hand_query_uvd = None
            hand_query_rot = None
            gt_hand_uvd_disp = None
            gt_hand_rot_disp = None
            
            if hand_mode is not None:
                hand_query_uvd_list = []
                hand_query_rot_list = []
                gt_hand_uvd_disp_list = []
                gt_hand_rot_disp_list = []
                
                sides = []
                if hand_mode in ('left', 'both'):
                    sides.append('left')
                if hand_mode in ('right', 'both'):
                    sides.append('right')
                
                for side in sides:
                    prefix = f'{side}_wrist'
                    if f'{prefix}_query_uvd' in batch:
                        hand_query_uvd_list.append(batch[f'{prefix}_query_uvd'].to(device))
                        hand_query_rot_list.append(batch[f'{prefix}_query_rot_6d'].to(device))
                        gt_hand_uvd_disp_list.append(batch[f'{prefix}_uvd_displacements'].to(device))
                        gt_hand_rot_disp_list.append(batch[f'{prefix}_rot_displacements'].to(device))
                
                if hand_query_uvd_list:
                    hand_query_uvd = torch.stack(hand_query_uvd_list, dim=1)
                    hand_query_rot = torch.stack(hand_query_rot_list, dim=1)
                    gt_hand_uvd_disp = torch.stack(gt_hand_uvd_disp_list, dim=1)
                    gt_hand_rot_disp = torch.stack(gt_hand_rot_disp_list, dim=1)
            
            # Extract text for text_mode
            text = batch.get('text') if text_mode else None
            
            # BATCHED prediction (much faster!)
            if is_diffusion:
                outputs = model.predict(frames, query_coords, depth=model_depth, num_inference_steps=num_inference_steps,
                                       hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
            else:
                outputs = model.predict(frames, query_coords, depth=model_depth,
                                       hand_query_uvd=hand_query_uvd, hand_query_rot=hand_query_rot, text=text)
            
            # Handle dict output (new format) or tensor output (backward compat)
            if isinstance(outputs, dict):
                pred_track_disp = outputs['track_disp']
            else:
                pred_track_disp = outputs
            
            # Compute track error in normalized space (L2 norm)
            track_error = torch.norm(pred_track_disp - gt_disp_normalized, dim=-1)  # (B, N, T)
            
            # Mask out t=0 (displacement at t=0 should be zero)
            loss_mask = torch.ones_like(track_error)
            loss_mask[:, :, 0] = 0.0
            
            total_track_error += (track_error * loss_mask).sum().item()
            total_track_points += loss_mask.sum().item()
            
            # Compute hand errors if available
            if hand_mode is not None and gt_hand_uvd_disp is not None and isinstance(outputs, dict) and 'hand_uvd_disp' in outputs:
                pred_hand_uvd = outputs['hand_uvd_disp']  # (B, H, T, 3)
                pred_hand_rot = outputs['hand_rot_disp']  # (B, H, T, 6)
                
                hand_uvd_error = torch.norm(pred_hand_uvd - gt_hand_uvd_disp, dim=-1)  # (B, H, T)
                hand_rot_error = torch.norm(pred_hand_rot - gt_hand_rot_disp, dim=-1)  # (B, H, T)
                
                hand_mask = torch.ones_like(hand_uvd_error)
                hand_mask[:, :, 0] = 0.0
                
                total_hand_uvd_error += (hand_uvd_error * hand_mask).sum().item()
                total_hand_rot_error += (hand_rot_error * hand_mask).sum().item()
                total_hand_points += hand_mask.sum().item()
            
            # Collect visualization samples (only for specific indices)
            if vis_sample_indices is not None:
                for b in range(B):
                    global_idx = batch_idx * val_loader.batch_size + b
                    if global_idx in vis_sample_indices:
                        # Reconstruct positions for visualization
                        # query_coords: (N, coord_dim), gt_disp: (N, T, coord_dim)
                        qc = query_coords[b:b+1]  # (1, N, coord_dim)
                        gt_disp_b = gt_disp_normalized[b]  # (N, T, coord_dim)
                        
                        # For diffusion models, re-run prediction with intermediate for this sample
                        if is_diffusion:
                            frame = frames[b:b+1]
                            d = model_depth[b:b+1] if model_depth is not None else None
                            sample_text = [text[b]] if text is not None else None
                            pred_single, intermediate = model.predict(
                                frame, qc, depth=d, num_inference_steps=num_inference_steps, return_intermediate=True, text=sample_text
                            )
                            # Handle dict or tensor output
                            if isinstance(pred_single, dict):
                                pred_disp_vis = pred_single['track_disp']
                            else:
                                pred_disp_vis = pred_single
                            vis_dict = {
                                'frame': frame.cpu(),
                                'gt_disp': gt_disp_b.cpu(),  # (N, T, coord_dim)
                                'pred_disp': pred_disp_vis.cpu(),  # (1, N, T, coord_dim)
                                'query_coords': qc[0].cpu(),  # (N, coord_dim)
                                'intermediate': [x.cpu() for x in intermediate],
                                'track_type': track_type,
                            }
                        else:
                            vis_dict = {
                                'frame': frames[b:b+1].cpu(),
                                'gt_disp': gt_disp_b.cpu(),  # (N, T, coord_dim)
                                'pred_disp': pred_track_disp[b:b+1].cpu(),  # (1, N, T, coord_dim)
                                'query_coords': query_coords[b].cpu(),  # (N, coord_dim)
                                'track_type': track_type,
                            }
                        
                        # Add hand data to vis_dict if available
                        if hand_mode is not None and hand_query_uvd is not None:
                            vis_dict['hand_query_uvd'] = hand_query_uvd[b].cpu()  # (H, 3)
                            vis_dict['hand_query_rot'] = hand_query_rot[b].cpu()  # (H, 6)
                            
                            if gt_hand_uvd_disp is not None:
                                vis_dict['gt_hand_uvd'] = gt_hand_uvd_disp[b].cpu()  # (H, T, 3)
                                vis_dict['gt_hand_rot'] = gt_hand_rot_disp[b].cpu()  # (H, T, 6)
                            
                            if isinstance(outputs, dict) and 'hand_uvd_disp' in outputs:
                                vis_dict['pred_hand_uvd'] = outputs['hand_uvd_disp'][b].cpu()  # (H, T, 3)
                                vis_dict['pred_hand_rot'] = outputs['hand_rot_disp'][b].cpu()  # (H, T, 6)
                        
                        vis_data.append(vis_dict)
    
    metrics = {
        'avg_error': total_track_error / total_track_points if total_track_points > 0 else 0.0,
    }
    
    # Add hand metrics if available
    if total_hand_points > 0:
        metrics['avg_hand_uvd_error'] = total_hand_uvd_error / total_hand_points
        metrics['avg_hand_rot_error'] = total_hand_rot_error / total_hand_points
    
    return metrics, vis_data


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, ema_model, device, cfg, epoch, global_step, track_type='2d', gpu_augmenter=None, aug_prob=0.0, hand_mode=None, text_mode=False):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training dataloader (returns dict format)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision
        ema_model: EMA model
        device: torch device
        cfg: Configuration namespace
        epoch: Current epoch number
        global_step: Current global step counter
        track_type: "2d" or "3d" - determines whether depth is used
        gpu_augmenter: GPUAugmentations instance for GPU-accelerated augmentations
        aug_prob: Probability of applying augmentations per batch
        hand_mode: "left", "right", "both", or None - determines hand data loading
        text_mode: bool - whether to use text conditioning
    
    Returns:
        global_step: updated global step counter
    """
    model.train()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        # === TO GPU ===
        frames = batch['imgs'].to(device, non_blocking=True)
        query_coords = batch['query_coords'].to(device, non_blocking=True)
        displacements = batch['displacements'].to(device, non_blocking=True)
        depth = batch.get('depth')
        if depth is not None:
            depth = depth.to(device, non_blocking=True)
        
        # === GPU AUGMENTATION ===
        if gpu_augmenter is not None and np.random.rand() < aug_prob:
            batch_gpu = {
                'imgs': frames,
                'query_coords': query_coords,
                'displacements': displacements,
                'depth': depth,
            }
            batch_gpu = gpu_augmenter(batch_gpu)
            frames = batch_gpu['imgs']
            query_coords = batch_gpu['query_coords']
            displacements = batch_gpu['displacements']
            depth = batch_gpu.get('depth')
        
        # Permute displacements: (B, T, N, D) -> (B, N, T, D)
        gt_disp_normalized = displacements.permute(0, 2, 1, 3)
        
        # Prepare depth for model (if 3D mode)
        model_depth = None
        if track_type == '3d' and depth is not None:
            frame_stack = frames.shape[1]
            model_depth = depth.unsqueeze(2)  # (B, frame_stack, 1, H, W)
        
        model_batch = {
            'frames': frames,
            'query_coords': query_coords,
            'gt_disp_normalized': gt_disp_normalized,
            'depth': model_depth,
        }
        
        # === TEXT DATA ===
        if text_mode and 'text' in batch:
            model_batch['text'] = batch['text']  # List[str]
        
        # === HAND DATA ===
        if hand_mode is not None:
            # Collect hand data from batch based on hand_mode
            hand_query_uvd_list = []
            hand_query_rot_list = []
            gt_hand_uvd_disp_list = []
            gt_hand_rot_disp_list = []
            
            sides = []
            if hand_mode in ('left', 'both'):
                sides.append('left')
            if hand_mode in ('right', 'both'):
                sides.append('right')
            
            for side in sides:
                prefix = f'{side}_wrist'
                if f'{prefix}_query_uvd' in batch:
                    hand_query_uvd_list.append(batch[f'{prefix}_query_uvd'].to(device, non_blocking=True))
                    hand_query_rot_list.append(batch[f'{prefix}_query_rot_6d'].to(device, non_blocking=True))
                    # Displacements are (B, T, dim), need to permute to (B, T, dim) -> keep as is
                    gt_hand_uvd_disp_list.append(batch[f'{prefix}_uvd_displacements'].to(device, non_blocking=True))
                    gt_hand_rot_disp_list.append(batch[f'{prefix}_rot_displacements'].to(device, non_blocking=True))
            
            if hand_query_uvd_list:
                # Stack along hand dimension: (B, H, dim)
                # hand_query_uvd: list of (B, 3) -> stack to (B, H, 3)
                model_batch['hand_query_uvd'] = torch.stack(hand_query_uvd_list, dim=1)
                model_batch['hand_query_rot'] = torch.stack(hand_query_rot_list, dim=1)
                # gt_hand_uvd_disp: list of (B, T, 3) -> stack to (B, H, T, 3)
                model_batch['gt_hand_uvd_disp'] = torch.stack(gt_hand_uvd_disp_list, dim=1)
                model_batch['gt_hand_rot_disp'] = torch.stack(gt_hand_rot_disp_list, dim=1)
        
        # === FORWARD ===
        with autocast(device_type='cuda', enabled=cfg.training.use_amp):
            loss_dict = model.compute_loss(model_batch)
            loss = loss_dict['total_loss']
            track_loss = loss_dict['track_loss']
            hand_uvd_loss = loss_dict.get('hand_uvd_loss', torch.tensor(0.0))
            hand_rot_loss = loss_dict.get('hand_rot_loss', torch.tensor(0.0))
        
        # === BACKWARD ===
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        
        # === OPTIMIZER ===
        # Track if optimizer step was skipped (due to inf/nan grads)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        
        # Only step scheduler if optimizer actually stepped (scale didn't decrease)
        if new_scale >= old_scale:
            scheduler.step()
            ema_model.update_parameters(model)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})
        
        # Log to W&B
        if global_step % cfg.training.log_every_n_steps == 0:
            log_dict = {
                'train/total_loss': loss.item(),
                'train/track_loss': track_loss.item(),
                'train/hand_uvd_loss': hand_uvd_loss.item() if torch.is_tensor(hand_uvd_loss) else hand_uvd_loss,
                'train/hand_rot_loss': hand_rot_loss.item() if torch.is_tensor(hand_rot_loss) else hand_rot_loss,
                'train/grad_norm': grad_norm.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
            }
            wandb.log(log_dict, step=global_step)
        
        global_step += 1
    
    return global_step


def _clean_state_dict(state_dict):
    """
    Remove torch.compile and EMA wrapper prefixes from state dict.
    - Removes '_orig_mod.' prefix (from torch.compile)
    - Removes 'module.' prefix (from AveragedModel/DDP)
    - Removes 'n_averaged' entry (from AveragedModel)
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k == 'n_averaged':
            continue
        # Remove module. prefix
        k = k.replace('module.', '', 1)
        # Remove _orig_mod. prefix
        k = k.replace('_orig_mod.', '', 1)
        cleaned[k] = v
    return cleaned


def save_checkpoint(model, optimizer, scheduler, ema_model, epoch, best_val_error, checkpoint_dir, filename):
    """Save model checkpoint with clean state dicts."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': _clean_state_dict(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_model_state_dict': _clean_state_dict(ema_model.state_dict()),
        'best_val_error': best_val_error,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train IntentTracker model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set device from config
    device_str = getattr(cfg.training, 'device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Determine model type for naming
    model_type = getattr(cfg.model, 'model_type', 'direct')
    
    # Create human-readable timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_type}_{timestamp}"
    
    # Create organized checkpoint directory: checkpoints/{model_type}/{timestamp}/
    run_checkpoint_dir = os.path.join(cfg.training.checkpoint_dir, model_type, timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Model type: {model_type}")
    print(f"Run name: {run_name}")
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Initialize W&B with descriptive run name
    wandb.init(
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        config={
            'model': vars(cfg.model),
            'training': vars(cfg.training),
            'dataset': vars(cfg.dataset_cfg),
        },
        name=run_name  # Use model_type_timestamp as run name
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
    
    # Get split configuration from dataset config
    split_mode = getattr(cfg.dataset_cfg, 'split_mode', 'episode')
    val_split = getattr(cfg.dataset_cfg, 'val_split', 0.1)
    val_seed = getattr(cfg.dataset_cfg, 'val_seed', 42)
    track_type = getattr(cfg.model, 'track_type', '2d')
    hand_mode = getattr(cfg.model, 'hand_mode', None)
    text_mode = getattr(cfg.model, 'text_mode', False)
    sample_stride = getattr(cfg.dataset_cfg, 'sample_stride', 1)
    
    # Encoder unfreezing schedule (percentage of training when encoders should be unfrozen)
    unfreeze_after = getattr(cfg.training, 'unfreeze_after', 0.0)  # Default: never freeze (backward compat)
    
    print(f"Track type: {track_type}")
    print(f"Hand mode: {hand_mode}")
    print(f"Text mode: {text_mode}")
    print(f"Split mode: {split_mode}")
    if unfreeze_after > 0:
        print(f"Encoder unfreezing at: {unfreeze_after * 100:.1f}% of training")
    
    # Create datasets based on split mode
    if split_mode == "episode":
        # Split by H5 files (episode-wise)
        train_files, val_files = create_train_val_split(
            cfg.dataset_dir,
            val_ratio=val_split,
            seed=val_seed,
            split_mode="episode"
        )
        
        train_dataset = TrackDataset(
            h5_files=train_files,
            img_size=cfg.dataset_cfg.img_size,
            frame_stack=cfg.dataset_cfg.frame_stack,
            num_track_ts=cfg.dataset_cfg.num_track_ts,
            num_track_ids=cfg.dataset_cfg.num_track_ids,
            downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
            cache_all=cfg.dataset_cfg.cache_all,
            cache_image=cfg.dataset_cfg.cache_image,
            track_type=track_type,
            hand_mode=hand_mode,
            sample_stride=sample_stride,
            text_mode=text_mode,
        )
        
        val_dataset = TrackDataset(
            h5_files=val_files,
            img_size=cfg.dataset_cfg.img_size,
            frame_stack=cfg.dataset_cfg.frame_stack,
            num_track_ts=cfg.dataset_cfg.num_track_ts,
            num_track_ids=cfg.dataset_cfg.num_track_ids,
            downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
            cache_all=cfg.dataset_cfg.cache_all,
            cache_image=cfg.dataset_cfg.cache_image,
            track_type=track_type,
            hand_mode=hand_mode,
            sample_stride=sample_stride,
            text_mode=text_mode,
        )
    else:
        # Split by individual samples
        full_dataset = TrackDataset(
            dataset_dir=cfg.dataset_dir,
            img_size=cfg.dataset_cfg.img_size,
            frame_stack=cfg.dataset_cfg.frame_stack,
            num_track_ts=cfg.dataset_cfg.num_track_ts,
            num_track_ids=cfg.dataset_cfg.num_track_ids,
            downsample_factor=getattr(cfg.dataset_cfg, 'downsample_factor', 1),
            cache_all=cfg.dataset_cfg.cache_all,
            cache_image=cfg.dataset_cfg.cache_image,
            track_type=track_type,
            hand_mode=hand_mode,
            sample_stride=sample_stride,
            text_mode=text_mode,
        )
        
        train_dataset, val_dataset = create_sample_split(
            full_dataset,
            val_ratio=val_split,
            seed=val_seed
        )
    
    # Create GPU augmenter (augmentations applied on GPU in training loop)
    gpu_augmenter = create_gpu_augmentations(cfg, device)
    aug_prob = getattr(cfg.dataset_cfg, 'aug_prob', 0.0)
    print(f"GPU augmentations created (prob={aug_prob})")
    
    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.training.num_workers > 0 else False,
        prefetch_factor=4 if cfg.training.num_workers > 0 else None,  # More prefetching
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
    
    # Create model using factory
    model_type = getattr(cfg.model, 'model_type', 'direct')
    is_diffusion = (model_type == 'diffusion')
    
    model = create_model(cfg, disp_stats=disp_stats, device=device)
    
    # Apply initial encoder freezing if unfreeze_after is set
    # The model starts with encoders frozen, then unfreezes at unfreeze_after% of training
    encoders_frozen = False
    if unfreeze_after > 0:
        model.freeze_encoders()
        encoders_frozen = True
    
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
    scaler = GradScaler('cuda', enabled=cfg.training.use_amp)
    
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
        
        # Check if we should unfreeze encoders at this epoch
        training_progress = epoch / cfg.training.num_epochs
        if encoders_frozen and training_progress >= unfreeze_after:
            model.unfreeze_encoders()
            encoders_frozen = False
            print(f"Encoders unfrozen at epoch {epoch+1} ({training_progress*100:.1f}% of training)")
            wandb.log({'training/encoders_unfrozen': 1}, step=global_step)
        
        # Train (scheduler.step() is called per batch inside train_one_epoch)
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema_model, 
            device, cfg, epoch, global_step, track_type=track_type,
            gpu_augmenter=gpu_augmenter, aug_prob=aug_prob, hand_mode=hand_mode, 
            text_mode=text_mode
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
                ema_model.module, val_loader, device, track_type=track_type,
                vis_sample_indices=vis_sample_indices, is_diffusion=is_diffusion, 
                num_inference_steps=num_inference_steps, hand_mode=hand_mode,
                text_mode=text_mode
            )
            
            print(f"Validation avg track error: {metrics['avg_error']:.4f}")
            if 'avg_hand_uvd_error' in metrics:
                print(f"Validation avg hand UVD error: {metrics['avg_hand_uvd_error']:.4f}")
                print(f"Validation avg hand rot error: {metrics['avg_hand_rot_error']:.4f}")
            
            # Log metrics
            log_metrics = {
                'val/avg_error': metrics['avg_error'],
                'epoch': epoch + 1,
            }
            if 'avg_hand_uvd_error' in metrics:
                log_metrics['val/avg_hand_uvd_error'] = metrics['avg_hand_uvd_error']
                log_metrics['val/avg_hand_rot_error'] = metrics['avg_hand_rot_error']
            wandb.log(log_metrics, step=global_step)
            
            # Log visualizations
            if len(vis_data) > 0:
                images = visualize_predictions(vis_data, disp_stats, epoch + 1, hand_mode=hand_mode)
                wandb.log({'val/predictions': images}, step=global_step)
                
                # For diffusion models, also visualize the denoising process
                if is_diffusion:
                    # Only visualize first few samples to avoid clutter
                    diffusion_vis_samples = vis_data[:min(3, len(vis_data))]
                    diffusion_images = visualize_diffusion_process(
                        diffusion_vis_samples, disp_stats, epoch + 1
                    )
                    wandb.log({'val/diffusion_process': diffusion_images}, step=global_step)
            
            # Save checkpoints every validation
            # 1. Save checkpoint for this specific epoch
            epoch_checkpoint_name = f'epoch_{epoch+1:03d}.pth'
            save_checkpoint(
                model, optimizer, scheduler, ema_model, epoch, best_val_error,
                run_checkpoint_dir, epoch_checkpoint_name
            )
            print(f"Saved checkpoint: {epoch_checkpoint_name}")
            
            # 2. Always save/update latest checkpoint
            save_checkpoint(
                model, optimizer, scheduler, ema_model, epoch, best_val_error,
                run_checkpoint_dir, 'latest.pth'
            )
            
            # 3. Save best checkpoint if this is a new best
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