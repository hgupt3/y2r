"""
Loss functions for training the IntentTracker model.
"""

import torch
import torch.nn as nn


def displacement_loss(pred_disp, gt_disp, valid_mask=None):
    """
    L2 loss on predicted vs ground truth displacements.
    
    Args:
        pred_disp: (B, N, T, 2) - predicted cumulative displacements
        gt_disp: (B, N, T, 2) - ground truth cumulative displacements
        valid_mask: (B, N, T) - validity mask (optional)
    
    Returns:
        loss: scalar tensor
    """
    # Compute L2 distance
    error = torch.norm(pred_disp - gt_disp, dim=-1)  # (B, N, T)
    
    if valid_mask is not None:
        error = error * valid_mask
        loss = error.sum() / valid_mask.sum()
    else:
        loss = error.mean()
    
    return loss


def normalized_displacement_loss(pred_disp, gt_disp, disp_std, valid_mask=None):
    """
    L2 loss on predicted vs ground truth displacements, normalized by std.
    
    This allows the loss to be scale-invariant and easier to tune.
    
    Args:
        pred_disp: (B, N, T, 2) - predicted cumulative displacements (normalized)
        gt_disp: (B, N, T, 2) - ground truth cumulative displacements (normalized)
        disp_std: (2,) - displacement standard deviation for denormalization
        valid_mask: (B, N, T) - validity mask (optional)
    
    Returns:
        loss: scalar tensor
    """
    # Denormalize predictions and GT to pixel space
    device = pred_disp.device
    std_tensor = torch.tensor(disp_std, device=device, dtype=pred_disp.dtype)
    
    pred_disp_denorm = pred_disp * std_tensor
    gt_disp_denorm = gt_disp * std_tensor
    
    # Compute L2 distance in pixel space
    error = torch.norm(pred_disp_denorm - gt_disp_denorm, dim=-1)  # (B, N, T)
    
    if valid_mask is not None:
        error = error * valid_mask
        loss = error.sum() / valid_mask.sum()
    else:
        loss = error.mean()
    
    return loss

