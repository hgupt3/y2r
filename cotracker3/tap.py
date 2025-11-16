import torch
from cotracker3.cotracker.utils.visualizer import Visualizer, read_video_from_path
import numpy as np

def cotracker(
    model,
    frames,
    segm_mask,
    grid_size=10,
    visibility_threshold=0.9,
    query_frame_idx=0,
    save_dir=None,
    device="cuda"
    ):
    """
    Run CoTracker on video frames.
    
    Args:
        model: Pre-loaded CoTracker model
        frames: Video frames tensor of shape (T, C, H, W)
        segm_mask: Segmentation mask tensor of shape (H, W)
        grid_size: Grid density for point tracking
        visibility_threshold: Minimum visibility (0-1) to keep tracks; tracks below this at ANY frame are pruned
        query_frame_idx: Frame index for initialization (relative to window)
        save_dir: Optional directory to save visualization
        device: Device to run on
    
    Returns:
        pred_tracks: Predicted tracks tensor (with coordinates in original frame space)
        pred_visibility: Predicted visibility tensor
    """
    # Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    
    # Prepare mask: (H, W) -> (1, 1, H, W)
    if segm_mask.dim() == 2:
        segm_mask = segm_mask.unsqueeze(0).unsqueeze(0)
    elif segm_mask.dim() == 3:
        segm_mask = segm_mask.unsqueeze(0)
    
    # Move to device
    frames = frames.to(device)
    segm_mask = segm_mask.to(device)

    # Run CoTracker inference (offline model)
    pred_tracks, pred_visibility = model(
        frames,
        grid_size=grid_size,
        grid_query_frame=query_frame_idx,
        segm_mask=segm_mask,
        backward_tracking=True,
    )

    # Filter out tracks that are occluded at ANY point during tracking
    # pred_visibility shape: (B, T, N) where values are 0-1 (1=visible, 0=occluded)
    
    # Check if track is visible at ALL frames: (B, T, N) -> (B, N)
    all_frames_visible = torch.all(pred_visibility > visibility_threshold, dim=1)  # (B, N)
    
    # Keep only tracks that are visible throughout entire sequence
    B, T, N, _ = pred_tracks.shape
    
    # Filter tracks for each batch element
    filtered_tracks = []
    filtered_visibility = []
    
    for b in range(B):
        keep_mask = all_frames_visible[b]  # (N,)
        if keep_mask.any():
            filtered_tracks.append(pred_tracks[b, :, keep_mask, :])  # (T, num_kept, 2)
            filtered_visibility.append(pred_visibility[b, :, keep_mask])  # (T, num_kept)
        else:
            # No tracks kept, return empty tensors
            filtered_tracks.append(torch.zeros(T, 0, 2, device=pred_tracks.device))
            filtered_visibility.append(torch.zeros(T, 0, device=pred_visibility.device))
    
    # Stack back into batch
    pred_tracks = torch.stack(filtered_tracks, dim=0)  # (B, T, num_kept, 2)
    pred_visibility = torch.stack(filtered_visibility, dim=0)  # (B, T, num_kept)

    # Per-window visualization is now disabled for efficiency
    # Trajectory summary visualization is handled in process_cotracker.py instead
    
    return pred_tracks, pred_visibility

if __name__ == "__main__":
    video_path = "./assets/apple.mp4"
    save_dir = "./saved_videos"
    cotracker(video_path, save_dir=save_dir)