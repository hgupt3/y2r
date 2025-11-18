"""
Visualization utilities for tracking predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
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
    Uses gradient coloring where light colors = early time, dark colors = late time.
    
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
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_array)
    ax.set_title(title)
    ax.axis('off')
    
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
    
    # Plot ground truth tracks with gradient
    if gt_tracks is not None:
        for i in range(N):
            traj_x = gt_tracks_np[i, :, 0] * W
            traj_y = gt_tracks_np[i, :, 1] * H
            _draw_gradient_trajectory(ax, traj_x, traj_y, gt_cmap, linewidth=3, alpha=0.85)
    
    # Plot predicted tracks with gradient
    if pred_tracks is not None:
        for i in range(N):
            traj_x = pred_tracks_np[i, :, 0] * W
            traj_y = pred_tracks_np[i, :, 1] * H
            _draw_gradient_trajectory(ax, traj_x, traj_y, pred_cmap, linewidth=3, alpha=0.85)
    
    # Create simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#50C850', linewidth=3, alpha=0.85, label='GT'),
        Line2D([0], [0], color='#00BFFF', linewidth=3, alpha=0.85, label='Pred'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set axis limits to match image
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Flip y-axis for image coordinates
    
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


def visualize_diffusion_process(vis_data, disp_stats, epoch):
    """
    Create W&B visualization of the diffusion denoising process.
    Shows trajectories at different diffusion timesteps from noisy to clean.
    
    Args:
        vis_data: List of dicts with keys:
            - 'frame': (1, frame_stack, 3, H, W) tensor
            - 'gt_tracks': (num_track_ts, N, 2) tensor - GT positions in [0, 1]
            - 'pred_disp': (1, N, T, 2) tensor - final predicted displacements (normalized)
            - 'query_coords': (N, 2) tensor - initial positions in [0, 1]
            - 'intermediate': list of (1, N, T, 2) tensors - intermediate predictions
        disp_stats: dict with 'displacement_mean' and 'displacement_std'
        epoch: int - current epoch number
        
    Returns:
        List of wandb.Image objects for logging
    """
    wandb_images = []
    
    mean = disp_stats['displacement_mean']
    std = disp_stats['displacement_std']
    
    for sample_idx, sample in enumerate(vis_data):
        # Skip if no intermediate predictions
        if 'intermediate' not in sample or len(sample['intermediate']) == 0:
            continue
        
        # Use the last (most recent) frame for visualization
        frame = sample['frame'][0, -1]  # (3, H, W)
        gt_tracks_data = sample['gt_tracks']  # (T, N, 2) - positions
        query_coords = sample['query_coords']  # (N, 2)
        intermediate = sample['intermediate']  # List of (1, N, T, 2) tensors
        
        # GT tracks are already positions in [0, 1]
        gt_tracks = gt_tracks_data.permute(1, 0, 2)  # (N, T, 2)
        
        # Select timesteps to visualize (evenly spaced through the denoising process)
        num_steps = len(intermediate)
        # Show at most 8 steps: start, middle steps, and end
        if num_steps > 8:
            step_indices = [0, num_steps//6, num_steps//3, num_steps//2, 
                           2*num_steps//3, 5*num_steps//6, num_steps-1]
        else:
            step_indices = list(range(num_steps))
        
        # Create grid layout (2 rows x 4 cols or adjust based on num steps)
        num_cols = 4
        num_rows = (len(step_indices) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Denormalize image
        img_array = denormalize_image(frame)
        H, W = img_array.shape[:2]
        
        # Limit points for visualization
        N = query_coords.shape[0]
        max_points = 32
        if N > max_points:
            indices = np.linspace(0, N-1, max_points, dtype=int)
            query_coords_viz = query_coords[indices]
            gt_tracks_viz = gt_tracks[indices]
        else:
            query_coords_viz = query_coords
            gt_tracks_viz = gt_tracks
            indices = np.arange(N)
        
        # Convert to numpy
        query_coords_np = query_coords_viz.cpu().numpy()
        gt_tracks_np = gt_tracks_viz.cpu().numpy()
        
        # Create custom colormaps for diffusion stages
        # Start with red (noisy), transition through orange/yellow to blue (clean)
        stage_colors = [
            ('#FF6B6B', '#FF4444'),  # Red for noisy
            ('#FFA500', '#FF8C00'),  # Orange
            ('#FFD700', '#FFC700'),  # Gold
            ('#90EE90', '#50C850'),  # Green
            ('#87CEEB', '#00BFFF'),  # Light blue
            ('#1E90FF', '#0066CC'),  # Blue for clean
            ('#0000CD', '#000080'),  # Dark blue for final
        ]
        
        # Plot each denoising step
        for plot_idx, step_idx in enumerate(step_indices):
            row = plot_idx // num_cols
            col = plot_idx % num_cols
            ax = axes[row, col]
            
            # Show image
            ax.imshow(img_array)
            ax.axis('off')
            
            # Get intermediate prediction at this step
            intermediate_disp = intermediate[step_idx][0]  # (N, T, 2)
            intermediate_disp = intermediate_disp[indices] if N > max_points else intermediate_disp
            
            # Denormalize displacements
            intermediate_disp_denorm = denormalize_displacements(intermediate_disp, mean, std)
            
            # Convert displacements to positions
            pred_tracks = query_coords_viz.unsqueeze(1) + intermediate_disp_denorm  # (N, T, 2)
            pred_tracks_np = pred_tracks.cpu().numpy()
            
            # Choose color based on progress through denoising
            color_idx = min(plot_idx, len(stage_colors) - 1)
            pred_cmap = LinearSegmentedColormap.from_list(
                f'pred_gradient_{plot_idx}',
                [stage_colors[color_idx][0], stage_colors[color_idx][1]]
            )
            
            # Plot GT tracks (light, for reference)
            gt_cmap = LinearSegmentedColormap.from_list(
                'gt_gradient', 
                ['#90EE90', '#50C850']  # Light green
            )
            for i in range(len(query_coords_viz)):
                traj_x = gt_tracks_np[i, :, 0] * W
                traj_y = gt_tracks_np[i, :, 1] * H
                _draw_gradient_trajectory(ax, traj_x, traj_y, gt_cmap, linewidth=2, alpha=0.4)
            
            # Plot predicted tracks at this diffusion step
            for i in range(len(query_coords_viz)):
                traj_x = pred_tracks_np[i, :, 0] * W
                traj_y = pred_tracks_np[i, :, 1] * H
                _draw_gradient_trajectory(ax, traj_x, traj_y, pred_cmap, linewidth=3, alpha=0.9)
            
            # Set title based on step
            if step_idx == 0:
                title = f"Step 1 (noisy)"
            elif step_idx == num_steps - 1:
                title = f"Step {num_steps} (clean)"
            else:
                title = f"Step {step_idx + 1}"
            ax.set_title(title, fontsize=14)
            
            # Set axis limits
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
        
        # Hide unused subplots
        for plot_idx in range(len(step_indices), num_rows * num_cols):
            row = plot_idx // num_cols
            col = plot_idx % num_cols
            axes[row, col].axis('off')
        
        # Add overall title
        fig.suptitle(f"Diffusion Denoising Process - Epoch {epoch} - Sample {sample_idx+1}", 
                     fontsize=16, y=1.0)
        
        # Convert to wandb image
        wandb_img = fig_to_wandb_image(fig)
        wandb_images.append(wandb_img)
    
    return wandb_images

