"""
Model factory for creating IntentTracker models from configuration.

This module provides a unified interface for instantiating different model types
(direct, diffusion, autoregressive) based on configuration parameters.
"""

import torch
from y2r.models.model import IntentTracker
from y2r.models.diffusion_model import DiffusionIntentTracker
from y2r.models.autoreg_model import AutoregressiveIntentTracker


def create_model(cfg, disp_stats=None, device='cuda', from_pretrained=True):
    """
    Factory function to create models based on configuration.
    
    This function instantiates the appropriate model class based on the
    'model_type' field in the configuration, extracting all necessary
    parameters and handling model-specific requirements.
    
    Args:
        cfg: Configuration namespace with model parameters. Must have:
            - cfg.model.model_type: str, one of 'direct', 'diffusion', 'autoreg'
            - cfg.model.*: other model parameters
        disp_stats: Optional dict with displacement statistics:
            - 'displacement_mean': list/array of length 2
            - 'displacement_std': list/array of length 2
            Required for 'diffusion' model type.
        device: str or torch.device, device to place model on (default: 'cuda')
        from_pretrained: bool, if True loads ViT from torch.hub, if False skips
            (for loading from checkpoint). Default: True for backward compatibility.
    
    Returns:
        model: Instantiated model moved to specified device.
            One of: IntentTracker, DiffusionIntentTracker, AutoregressiveIntentTracker
    
    Raises:
        ValueError: If model_type is invalid or required parameters are missing
    
    Example:
        >>> cfg = load_config('configs/train_direct.yaml')
        >>> model = create_model(cfg, device='cuda:0')
    """
    # Extract model type
    model_type = getattr(cfg.model, 'model_type', 'direct')
    
    # Common parameters shared across all models
    common_params = {
        'num_future_steps': cfg.model.num_future_steps,
        'hidden_size': cfg.model.hidden_size,
        'model_resolution': cfg.model.model_resolution,
        'add_space_attn': cfg.model.add_space_attn,
        'vit_model_name': cfg.model.vit_model_name,
        'vit_frozen': cfg.model.vit_frozen,
        'num_heads': cfg.model.num_heads,
        'mlp_ratio': cfg.model.mlp_ratio,
        'p_drop_attn': cfg.model.p_drop_attn,
        'frame_stack': cfg.model.frame_stack,
        'from_pretrained': from_pretrained,
    }
    
    # Instantiate model based on type
    if model_type == 'direct':
        print("Creating IntentTracker model (direct prediction)...")
        model = IntentTracker(**common_params)
    
    elif model_type == 'diffusion':
        print("Creating DiffusionIntentTracker model...")
        
        # Diffusion models require displacement statistics
        if disp_stats is None:
            raise ValueError(
                "disp_stats must be provided for diffusion models. "
                "Load normalization_stats.yaml before creating the model."
            )
        
        # Add diffusion-specific parameters
        diffusion_params = {
            **common_params,
            'num_diffusion_steps': getattr(cfg.model, 'num_diffusion_steps', 100),
            'beta_schedule': getattr(cfg.model, 'beta_schedule', 'squaredcos_cap_v2'),
            'disp_mean': disp_stats['displacement_mean'],
            'disp_std': disp_stats['displacement_std'],
        }
        model = DiffusionIntentTracker(**diffusion_params)
    
    elif model_type == 'autoreg':
        print("Creating AutoregressiveIntentTracker model...")
        model = AutoregressiveIntentTracker(**common_params)
    
    else:
        raise ValueError(
            f"Invalid model_type: '{model_type}'. "
            f"Must be one of: 'direct', 'diffusion', 'autoreg'"
        )
    
    # Move model to device
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    
    print(f"Model created and moved to {device}")
    
    return model

