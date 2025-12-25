"""
Model size presets and encoding dimension configurations.

This module defines standard configurations for small/base/large model sizes,
ensuring consistent hyperparameter settings across all model types.
"""

# Model size presets - auto-configure all hyperparameters based on size
MODEL_SIZE_CONFIGS = {
    's': {
        'hidden_size': 384,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'time_depth': 6,
        'vit_model_name': 'vit_small_patch16_dinov3',
        # SigLIP2 for text encoding (conservative: s/b use base, l uses large)
        'siglip_model_name': 'google/siglip-base-patch16-224',
        'text_embed_dim': 768,
    },
    'b': {
        'hidden_size': 768,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'time_depth': 12,
        'vit_model_name': 'vit_base_patch16_dinov3',
        'siglip_model_name': 'google/siglip-base-patch16-224',
        'text_embed_dim': 768,
    },
    'l': {
        'hidden_size': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'time_depth': 24,
        'vit_model_name': 'vit_large_patch16_dinov3',
        'siglip_model_name': 'google/siglip-large-patch16-256',
        'text_embed_dim': 1152,
    },
}

# Fixed encoding dimensions (independent of model size)
# These dimensions are chosen to give each input component appropriate representation
# without over-representing any single aspect
ENCODING_DIMS = {
    # Track token components (total: 384 for non-diffusion, 448 for diffusion)
    'track_position': 256,      # 2D sin/cos for (x, y) - 128 per axis
    'track_state': 64,          # Linear projection of displacement/noise (coord_dim -> 64)
    'temporal': 64,             # 1D sin/cos for timestep index
    
    # Hand token components (total: 384 for non-diffusion, 448 for diffusion)
    'hand_position': 128,       # 2D sin/cos for (u, v) - 64 per axis
    'hand_rotation': 128,       # Linear projection of 6D rotation
    'hand_state': 64,           # Linear projection of 9D hand state (UVD + rotation displacement)
    
    # Diffusion-specific (added to track/hand totals)
    'diffusion_timestep': 64,   # 1D sin/cos for diffusion timestep
}


def get_model_config(model_size: str) -> dict:
    """
    Get model configuration for a given size.
    
    Args:
        model_size: One of 's' (small), 'b' (base), 'l' (large)
        
    Returns:
        Dictionary with:
            - hidden_size, num_heads, mlp_ratio, time_depth
            - vit_model_name (DINOv3 variant)
            - siglip_model_name, text_embed_dim (SigLIP2 for text encoding)
        
    Raises:
        ValueError: If model_size is not recognized
    """
    if model_size not in MODEL_SIZE_CONFIGS:
        raise ValueError(
            f"Unknown model_size '{model_size}'. "
            f"Must be one of: {list(MODEL_SIZE_CONFIGS.keys())}"
        )
    return MODEL_SIZE_CONFIGS[model_size].copy()


def get_encoding_input_dim(include_diffusion: bool = False, is_hand: bool = False) -> int:
    """
    Calculate total input dimension for the encoding MLP.
    
    Args:
        include_diffusion: Whether to include diffusion timestep dimension
        is_hand: Whether this is for hand tokens (vs track tokens)
        
    Returns:
        Total dimension of concatenated encodings
    """
    dims = ENCODING_DIMS
    
    if is_hand:
        total = dims['hand_position'] + dims['hand_rotation'] + dims['hand_state'] + dims['temporal']
    else:
        total = dims['track_position'] + dims['track_state'] + dims['temporal']
    
    if include_diffusion:
        total += dims['diffusion_timestep']
    
    return total

