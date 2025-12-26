"""
Model factory for creating IntentTracker models from configuration.

This module provides a unified interface for instantiating different model types
(direct, diffusion, autoregressive) based on configuration parameters.
"""

import torch
from y2r.models.model import IntentTracker
from y2r.models.diffusion_model import DiffusionIntentTracker
from y2r.models.autoreg_model import AutoregressiveIntentTracker
from y2r.models.model_config import MODEL_SIZE_CONFIGS


def create_model(cfg, disp_stats=None, device='cuda', from_pretrained=True):
    """
    Factory function to create models based on configuration.
    
    This function instantiates the appropriate model class based on the
    'model_type' field in the configuration, extracting all necessary
    parameters and handling model-specific requirements.
    
    Args:
        cfg: Configuration namespace with model parameters. Must have:
            - cfg.model.model_type: str, one of 'direct', 'diffusion', 'autoreg'
            - cfg.model.model_size: str, one of 's', 'b', 'l'
            - cfg.model.text_mode: bool, whether to enable text conditioning
            - cfg.model.*: other model parameters
        disp_stats: Optional dict with displacement statistics:
            - 'displacement_mean': list/array of length 2 or 3
            - 'displacement_std': list/array of length 2 or 3
            Required for 'diffusion' model type.
        device: str or torch.device, device to place model on (default: 'cuda')
        from_pretrained: bool, if True loads ViT pretrained weights, if False skips
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
    # Extract model type, model size, track type, hand_mode, and text_mode from config
    model_type = getattr(cfg.model, 'model_type', 'direct')
    model_size = getattr(cfg.model, 'model_size', 's')
    track_type = getattr(cfg.model, 'track_type', '2d')
    hand_mode = getattr(cfg.model, 'hand_mode', None)
    text_mode = getattr(cfg.model, 'text_mode', False)
    
    # Validate model_size
    if model_size not in MODEL_SIZE_CONFIGS:
        raise ValueError(
            f"Invalid model_size: '{model_size}'. "
            f"Must be one of: {list(MODEL_SIZE_CONFIGS.keys())}"
        )
    
    # Common parameters shared across all models
    # Note: model_size replaces hidden_size, num_heads, mlp_ratio, vit_model_name
    common_params = {
        'model_size': model_size,
        'num_future_steps': cfg.model.num_future_steps,
        'model_resolution': tuple(cfg.model.model_resolution),
        'add_space_attn': cfg.model.add_space_attn,
        'vit_frozen': cfg.model.vit_frozen,
        'p_drop_attn': cfg.model.p_drop_attn,
        'frame_stack': cfg.model.frame_stack,
        'track_type': track_type,
        'from_pretrained': from_pretrained,
        'hand_mode': hand_mode,
        'text_mode': text_mode,
    }
    
    # Get config info for logging
    size_cfg = MODEL_SIZE_CONFIGS[model_size]
    
    # Instantiate model based on type
    if model_type == 'direct':
        print(f"Creating IntentTracker model (direct prediction, size={model_size})...")
        print(f"  hidden_size={size_cfg['hidden_size']}, num_heads={size_cfg['num_heads']}, "
              f"time_depth={size_cfg['time_depth']}, vit={size_cfg['vit_model_name']}")
        model = IntentTracker(**common_params)
        if hand_mode:
            print(f"  Hand mode enabled: {hand_mode}")
        if text_mode:
            print(f"  Text mode enabled: siglip={size_cfg['siglip_model_name']}")
    
    elif model_type == 'diffusion':
        print(f"Creating DiffusionIntentTracker model (size={model_size})...")
        print(f"  hidden_size={size_cfg['hidden_size']}, num_heads={size_cfg['num_heads']}, "
              f"time_depth={size_cfg['time_depth']}, vit={size_cfg['vit_model_name']}")
        
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
            'num_inference_steps': getattr(cfg.model, 'num_inference_steps', 10),
            'disp_mean': disp_stats['displacement_mean'],
            'disp_std': disp_stats['displacement_std'],
        }
        model = DiffusionIntentTracker(**diffusion_params)
        if hand_mode:
            print(f"  Hand mode enabled: {hand_mode}")
        if text_mode:
            print(f"  Text mode enabled: siglip={size_cfg['siglip_model_name']}")
    
    elif model_type == 'autoreg':
        print(f"Creating AutoregressiveIntentTracker model (size={model_size})...")
        print(f"  hidden_size={size_cfg['hidden_size']}, num_heads={size_cfg['num_heads']}, "
              f"time_depth={size_cfg['time_depth']}, vit={size_cfg['vit_model_name']}")
        model = AutoregressiveIntentTracker(**common_params)
        if hand_mode:
            print(f"  Hand mode enabled: {hand_mode}")
        if text_mode:
            print(f"  Text mode enabled: siglip={size_cfg['siglip_model_name']}")
    
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
