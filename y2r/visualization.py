"""
Visualization utilities for tracking predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import wandb


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize ImageNet-normalized image for visualization.
    
    Args:
        img_tensor: (C, H, W) tensor with ImageNet normalization
        
    Returns:
        img_array: (H, W, C) numpy array in [0, 255] range
    """
    # Clone to avoid modifying original
    img = img_tensor.clone()
    
    # Denormalize
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    
    # Clip to [0, 1] and convert to numpy
    img = torch.clamp(img, 0, 1)
    img_array = img.permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    return img_array


def denormalize_displacements(disp_normalized, mean, std):
    """
    Denormalize displacements from normalized space back to [0, 1] coordinate space.
    
    Args:
        disp_normalized: (..., 2) tensor with normalized displacements
        mean: [mean_x, mean_y] list
        std: [std_x, std_y] list
        
    Returns:
        disp_denorm: (..., 2) tensor with denormalized displacements
    """
    device = disp_normalized.device
    mean_tensor = torch.tensor(mean, device=device, dtype=disp_normalized.dtype)
    std_tensor = torch.tensor(std, device=device, dtype=disp_normalized.dtype)
    
    disp_denorm = disp_normalized * std_tensor + mean_tensor
    
    return disp_denorm


def visualize_tracks_on_frame(
    frame,
    query_coords,
    gt_tracks=None,
    pred_tracks=None,
    title="Track Visualization",
    max_points=32
):
    """
    Visualize ground truth and predicted tracks overlaid on a frame.
    
    Args:
        frame: (C, H, W) tensor - ImageNet normalized frame
        query_coords: (N, 2) tensor - initial positions in [0, 1] coordinates
        gt_tracks: (N, T, 2) tensor - ground truth positions in [0, 1] coordinates (optional)
        pred_tracks: (N, T, 2) tensor - predicted positions in [0, 1] coordinates (optional)
        title: str - plot title
        max_points: int - maximum number of points to visualize
        
    Returns:
        fig: matplotlib figure
    """
    # Denormalize image
    img_array = denormalize_image(frame)
    H, W = img_array.shape[:2]
    
    # Limit number of points for visualization
    N = query_coords.shape[0]
    if N > max_points:
        indices = np.linspace(0, N-1, max_points, dtype=int)
        query_coords = query_coords[indices]
        if gt_tracks is not None:
            gt_tracks = gt_tracks[indices]
        if pred_tracks is not None:
            pred_tracks = pred_tracks[indices]
        N = max_points
    
    # Convert to numpy
    query_coords_np = query_coords.cpu().numpy()
    if gt_tracks is not None:
        gt_tracks_np = gt_tracks.cpu().numpy()
    if pred_tracks is not None:
        pred_tracks_np = pred_tracks.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.set_title(title)
    ax.axis('off')
    
    # Plot ground truth tracks
    if gt_tracks is not None:
        for i in range(N):
            traj_x = gt_tracks_np[i, :, 0] * W
            traj_y = gt_tracks_np[i, :, 1] * H
            ax.plot(traj_x, traj_y, 'g-', linewidth=2, alpha=0.7)
            ax.scatter(traj_x[-1], traj_y[-1], c='green', s=80, marker='x', zorder=4)
        
        # Add legend entry
        ax.plot([], [], 'g-', linewidth=2, label='GT')
    
    # Plot predicted tracks
    if pred_tracks is not None:
        for i in range(N):
            traj_x = pred_tracks_np[i, :, 0] * W
            traj_y = pred_tracks_np[i, :, 1] * H
            ax.plot(traj_x, traj_y, 'b--', linewidth=2, alpha=0.7)
            ax.scatter(traj_x[-1], traj_y[-1], c='blue', s=80, marker='x', zorder=4)
        
        # Add legend entry
        ax.plot([], [], 'b--', linewidth=2, label='Pred')
    
    ax.legend(loc='upper right')
    
    return fig


def fig_to_wandb_image(fig):
    """
    Convert matplotlib figure to wandb.Image.
    
    Args:
        fig: matplotlib figure
        
    Returns:
        wandb.Image
    """
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Load as PIL image
    pil_img = Image.open(buf)
    
    # Convert to wandb.Image
    wandb_img = wandb.Image(pil_img)
    
    plt.close(fig)
    buf.close()
    
    return wandb_img


def visualize_predictions(vis_data, disp_stats, epoch):
    """
    Create W&B visualization of predictions vs GT.
    Overlay tracks on the first frame.
    
    Args:
        vis_data: List of dicts with keys:
            - 'frame': (1, frame_stack, 3, H, W) tensor
            - 'gt_tracks': (num_track_ts, N, 2) tensor - GT positions in [0, 1]
            - 'pred_disp': (1, N, T, 2) tensor - predicted displacements (normalized)
            - 'query_coords': (N, 2) tensor - initial positions in [0, 1]
        disp_stats: dict with 'displacement_mean' and 'displacement_std'
        epoch: int - current epoch number
        
    Returns:
        List of wandb.Image objects for logging
    """
    wandb_images = []
    
    mean = disp_stats['displacement_mean']
    std = disp_stats['displacement_std']
    
    for idx, sample in enumerate(vis_data):
        # Use the last (most recent) frame for visualization
        frame = sample['frame'][0, -1]  # (3, H, W)
        gt_tracks_data = sample['gt_tracks']  # (T, N, 2) - positions
        pred_disp = sample['pred_disp'][0]  # (N, T, 2) - displacements (normalized)
        query_coords = sample['query_coords']  # (N, 2)
        
        # GT tracks are already positions in [0, 1]
        gt_tracks = gt_tracks_data.permute(1, 0, 2)  # (N, T, 2)
        
        # Denormalize predicted displacements
        pred_disp_denorm = denormalize_displacements(pred_disp, mean, std)
        
        # Convert displacements to positions
        pred_tracks = query_coords.unsqueeze(1) + pred_disp_denorm  # (N, T, 2)
        
        # Create visualization
        fig = visualize_tracks_on_frame(
            frame=frame,
            query_coords=query_coords,
            gt_tracks=gt_tracks,
            pred_tracks=pred_tracks,
            title=f"Epoch {epoch} - Sample {idx+1}"
        )
        
        # Convert to wandb image
        wandb_img = fig_to_wandb_image(fig)
        wandb_images.append(wandb_img)
    
    return wandb_images

