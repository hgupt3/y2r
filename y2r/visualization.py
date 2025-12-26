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


def _draw_wrist_trajectory(ax, wrist_uv, rotations, H, W, color, axis_length_px=18, linewidth=2):
    """
    Draw wrist trajectory line + rotation axes at START and END only.
    Clean visualization with small orientation markers.
    
    Args:
        ax: matplotlib axis
        wrist_uv: (T, 2) array in [0, 1] coordinates
        rotations: (T, 3, 3) rotation matrices (optional, can be None)
        H, W: image height and width
        color: trajectory color (RGB tuple normalized to [0, 1])
        axis_length_px: length of rotation axes in pixels
        linewidth: line width for trajectory
    """
    T = wrist_uv.shape[0]
    
    # Convert to pixel coordinates
    wrist_px_x = wrist_uv[:, 0] * W
    wrist_px_y = wrist_uv[:, 1] * H
    
    # Draw trajectory line (no markers)
    ax.plot(wrist_px_x, wrist_px_y, color=color, linewidth=linewidth, alpha=0.85)
    
    # Draw small circle markers at start (hollow) and end (filled)
    ax.scatter(wrist_px_x[0], wrist_px_y[0], s=60, c='none', edgecolors=color, linewidths=2, zorder=5)  # Start: hollow
    ax.scatter(wrist_px_x[-1], wrist_px_y[-1], s=60, c=color, edgecolors=color, linewidths=1, zorder=5)  # End: filled
    
    # Draw rotation axes at START and END only (cleaner than all timesteps)
    if rotations is not None:
        axis_colors = {'x': '#FF0000', 'y': '#00FF00', 'z': '#0000FF'}  # RGB for XYZ axes
        
        for t_idx, t in enumerate([0, T-1]):  # Only first and last
            wrist_pos_x = wrist_px_x[t]
            wrist_pos_y = wrist_px_y[t]
            R_t = rotations[t]
            
            # Use slightly different alpha for start vs end
            alpha = 0.6 if t_idx == 0 else 0.9
            
            # Draw each axis (X=red, Y=green, Z=blue)
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                axis_3d = R_t[:, axis_idx]  # (3,) - the axis direction
                
                # Project to 2D: use X and Y with some Z perspective
                axis_2d_x = axis_3d[0] - axis_3d[2] * 0.3
                axis_2d_y = axis_3d[1] - axis_3d[2] * 0.3
                
                # Scale and compute end point
                end_x = wrist_pos_x + axis_2d_x * axis_length_px
                end_y = wrist_pos_y + axis_2d_y * axis_length_px
                
                # Draw axis line (thinner than before)
                ax.plot([wrist_pos_x, end_x], [wrist_pos_y, end_y], 
                       color=axis_colors[axis_name], linewidth=1.5, alpha=alpha)


def visualize_tracks_on_frame(
    frame,
    query_coords,
    gt_tracks=None,
    pred_tracks=None,
    title="Track Visualization",
    max_points=32,
    wrist_data=None
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
        wrist_data: dict with 'gt' and/or 'pred' keys, each containing:
            - 'left'/'right' with 'uv' (T, 2) and 'rotations' (T, 3, 3)
        
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
    
    # Pred: Light cyan → Dark blue (changed from purple for tracks to keep distinction)
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
    
    # Draw wrist trajectories if provided (GT=green, Pred=purple)
    if wrist_data is not None:
        # GT wrists = green (both left and right)
        gt_color = '#00FF00'  # Green
        # Pred wrists = purple (both left and right)
        pred_color = '#FF00FF'  # Purple/Magenta
        
        for data_type in ['gt', 'pred']:
            if data_type not in wrist_data:
                continue
            color = gt_color if data_type == 'gt' else pred_color
            
            for side in ['left', 'right']:
                if side not in wrist_data[data_type]:
                    continue
                data = wrist_data[data_type][side]
                wrist_uv = data.get('uv')  # (T, 2)
                rotations = data.get('rotations')  # (T, 3, 3) or None
                
                if wrist_uv is not None:
                    _draw_wrist_trajectory(ax, wrist_uv, rotations, H, W, color, 
                                          axis_length_px=40, linewidth=2)
    
    # Create simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#50C850', linewidth=3, alpha=0.85, label='GT Track'),
        Line2D([0], [0], color='#00BFFF', linewidth=3, alpha=0.85, label='Pred Track'),
    ]
    # Add wrist legend if present
    if wrist_data is not None:
        if 'gt' in wrist_data:
            legend_elements.append(Line2D([0], [0], color='#00FF00', linewidth=2, alpha=0.85, label='GT Wrist'))
        if 'pred' in wrist_data:
            legend_elements.append(Line2D([0], [0], color='#FF00FF', linewidth=2, alpha=0.85, label='Pred Wrist'))
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


def visualize_predictions(vis_data, disp_stats, epoch, hand_mode=None):
    """
    Create W&B visualization of predictions vs GT.
    Overlay tracks on the first frame.
    
    Args:
        vis_data: List of dicts with keys:
            - 'frame': (1, frame_stack, 3, H, W) tensor
            - 'gt_disp': (N, T, coord_dim) tensor - GT displacements (normalized)
            - 'pred_disp': (1, N, T, coord_dim) tensor - predicted displacements (normalized)
            - 'query_coords': (N, coord_dim) tensor - initial positions in [0, 1]
            Optional hand keys:
            - 'gt_hand_uvd': (H, T, 3) tensor - GT hand UVD positions
            - 'gt_hand_rot': (H, T, 6) tensor - GT hand 6D rotations
            - 'pred_hand_uvd': (H, T, 3) tensor - Pred hand UVD displacements
            - 'pred_hand_rot': (H, T, 6) tensor - Pred hand 6D rotations
            - 'hand_query_uvd': (H, 3) tensor - query hand UVD
            - 'hand_query_rot': (H, 6) tensor - query hand 6D rotation
        disp_stats: dict with 'displacement_mean' and 'displacement_std'
        epoch: int - current epoch number
        hand_mode: 'left', 'right', 'both', or None - determines which hand(s) are in the data
        
    Returns:
        List of wandb.Image objects for logging
    """
    wandb_images = []
    
    mean = disp_stats['displacement_mean']
    std = disp_stats['displacement_std']
    
    # Hand stats if available (use correct key names from normalization_stats.yaml)
    hand_uvd_mean = np.array(disp_stats.get('hand_uvd_disp_mean', [0, 0, 0]))
    hand_uvd_std = np.array(disp_stats.get('hand_uvd_disp_std', [1, 1, 1]))
    hand_rot_mean = np.array(disp_stats.get('hand_rot_disp_mean', [0]*6))
    hand_rot_std = np.array(disp_stats.get('hand_rot_disp_std', [1]*6))
    
    # Determine which sides are present based on hand_mode
    if hand_mode == 'left':
        hand_sides = ['left']
    elif hand_mode == 'right':
        hand_sides = ['right']
    elif hand_mode == 'both':
        hand_sides = ['left', 'right']
    else:
        hand_sides = []
    
    for idx, sample in enumerate(vis_data):
        # Use the last (most recent) frame for visualization
        frame = sample['frame'][0, -1]  # (3, H, W)
        gt_disp = sample['gt_disp']  # (N, T, coord_dim) - normalized displacements
        pred_disp = sample['pred_disp'][0]  # (N, T, coord_dim) - displacements (normalized)
        query_coords = sample['query_coords']  # (N, coord_dim)
        
        # Use only (x, y) for visualization (ignore depth for 3D)
        coord_dim = min(2, query_coords.shape[-1])
        query_xy = query_coords[..., :2]  # (N, 2)
        gt_disp_xy = gt_disp[..., :2]  # (N, T, 2)
        pred_disp_xy = pred_disp[..., :2]  # (N, T, 2)
        
        # Use only first 2 components of stats for 2D visualization
        mean_2d = mean[:2] if len(mean) > 2 else mean
        std_2d = std[:2] if len(std) > 2 else std
        
        # Denormalize displacements
        gt_disp_denorm = denormalize_displacements(gt_disp_xy, mean_2d, std_2d)
        pred_disp_denorm = denormalize_displacements(pred_disp_xy, mean_2d, std_2d)
        
        # Convert displacements to positions
        gt_tracks = query_xy.unsqueeze(1) + gt_disp_denorm  # (N, T, 2)
        pred_tracks = query_xy.unsqueeze(1) + pred_disp_denorm  # (N, T, 2)
        
        # Process hand data if available
        wrist_data = None
        if 'hand_query_uvd' in sample and sample['hand_query_uvd'] is not None:
            wrist_data = {}
            hand_query_uvd = sample['hand_query_uvd']  # (H, 3) - H is number of hands
            hand_query_rot = sample['hand_query_rot']  # (H, 6)
            
            # GT hand data
            if 'gt_hand_uvd' in sample and sample['gt_hand_uvd'] is not None:
                gt_hand_uvd_disp = sample['gt_hand_uvd']  # (H, T, 3) normalized
                gt_hand_rot_disp = sample['gt_hand_rot']  # (H, T, 6) normalized
                
                wrist_data['gt'] = {}
                H = gt_hand_uvd_disp.shape[0]
                # Use hand_sides from hand_mode (correct mapping)
                sides = hand_sides[:H] if hand_sides else ['left', 'right'][:H]
                
                for h, side in enumerate(sides):
                    # Denormalize UVD displacement
                    uvd_disp_h = gt_hand_uvd_disp[h].cpu().numpy()  # (T, 3)
                    uvd_disp_h = uvd_disp_h * hand_uvd_std + hand_uvd_mean
                    
                    # Reconstruct trajectory: query + disp (use only u, v)
                    query_uv = hand_query_uvd[h, :2].cpu().numpy()  # (2,)
                    wrist_uv = query_uv[None, :] + uvd_disp_h[:, :2]  # (T, 2)
                    
                    # Compute rotations
                    rot_disp_h = gt_hand_rot_disp[h].cpu().numpy()  # (T, 6)
                    rot_disp_h = rot_disp_h * hand_rot_std + hand_rot_mean
                    query_rot = hand_query_rot[h].cpu().numpy()  # (6,)
                    R_0 = rot_6d_to_matrix(query_rot)
                    
                    rotations = np.zeros((rot_disp_h.shape[0], 3, 3), dtype=np.float32)
                    for t in range(rot_disp_h.shape[0]):
                        R_rel = rot_6d_to_matrix(rot_disp_h[t])
                        rotations[t] = R_rel @ R_0
                    
                    wrist_data['gt'][side] = {'uv': wrist_uv, 'rotations': rotations}
            
            # Pred hand data
            if 'pred_hand_uvd' in sample and sample['pred_hand_uvd'] is not None:
                pred_hand_uvd_disp = sample['pred_hand_uvd']  # (H, T, 3) normalized
                pred_hand_rot_disp = sample['pred_hand_rot']  # (H, T, 6) normalized
                
                wrist_data['pred'] = {}
                H = pred_hand_uvd_disp.shape[0]
                # Use hand_sides from hand_mode (correct mapping)
                sides = hand_sides[:H] if hand_sides else ['left', 'right'][:H]
                
                for h, side in enumerate(sides):
                    # Denormalize UVD displacement
                    uvd_disp_h = pred_hand_uvd_disp[h].cpu().numpy()  # (T, 3)
                    uvd_disp_h = uvd_disp_h * hand_uvd_std + hand_uvd_mean
                    
                    # Reconstruct trajectory: query + disp (use only u, v)
                    query_uv = hand_query_uvd[h, :2].cpu().numpy()  # (2,)
                    wrist_uv = query_uv[None, :] + uvd_disp_h[:, :2]  # (T, 2)
                    
                    # Compute rotations
                    rot_disp_h = pred_hand_rot_disp[h].cpu().numpy()  # (T, 6)
                    rot_disp_h = rot_disp_h * hand_rot_std + hand_rot_mean
                    query_rot = hand_query_rot[h].cpu().numpy()  # (6,)
                    R_0 = rot_6d_to_matrix(query_rot)
                    
                    rotations = np.zeros((rot_disp_h.shape[0], 3, 3), dtype=np.float32)
                    for t in range(rot_disp_h.shape[0]):
                        R_rel = rot_6d_to_matrix(rot_disp_h[t])
                        rotations[t] = R_rel @ R_0
                    
                    wrist_data['pred'][side] = {'uv': wrist_uv, 'rotations': rotations}
        
        # Create visualization
        fig = visualize_tracks_on_frame(
            frame=frame,
            query_coords=query_xy,
            gt_tracks=gt_tracks,
            pred_tracks=pred_tracks,
            title=f"Epoch {epoch} - Sample {idx+1}",
            wrist_data=wrist_data
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
            - 'gt_disp': (N, T, coord_dim) tensor - GT displacements (normalized)
            - 'pred_disp': (1, N, T, coord_dim) tensor - predicted displacements (normalized)
            - 'query_coords': (N, coord_dim) tensor - initial positions in [0, 1]
            - 'intermediate': list of (1, N, T, coord_dim) tensors - intermediate predictions
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
        gt_disp = sample['gt_disp']  # (N, T, coord_dim) - normalized displacements
        query_coords = sample['query_coords']  # (N, coord_dim)
        intermediate = sample['intermediate']  # List of (1, N, T, coord_dim) tensors
        
        # Use only (x, y) for visualization
        query_xy = query_coords[..., :2]  # (N, 2)
        gt_disp_xy = gt_disp[..., :2]  # (N, T, 2)
        
        # Use only first 2 components of stats for 2D visualization
        mean_2d = mean[:2] if len(mean) > 2 else mean
        std_2d = std[:2] if len(std) > 2 else std
        
        # Denormalize GT displacements and convert to positions
        gt_disp_denorm = denormalize_displacements(gt_disp_xy, mean_2d, std_2d)
        gt_tracks = query_xy.unsqueeze(1) + gt_disp_denorm  # (N, T, 2)
        
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
        N = query_xy.shape[0]
        max_points = 32
        if N > max_points:
            indices = np.linspace(0, N-1, max_points, dtype=int)
            query_coords_viz = query_xy[indices]
            gt_tracks_viz = gt_tracks[indices]
        else:
            query_coords_viz = query_xy
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
            
            # Get intermediate prediction at this step (use only x, y)
            intermediate_disp = intermediate[step_idx][0][..., :2]  # (N, T, 2)
            intermediate_disp = intermediate_disp[indices] if N > max_points else intermediate_disp
            
            # Denormalize displacements
            intermediate_disp_denorm = denormalize_displacements(intermediate_disp, mean_2d, std_2d)
            
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

