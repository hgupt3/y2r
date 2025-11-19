import torch
import torch.nn as nn
import torch.nn.functional as F

from y2r.models.blocks import EfficientUpdateFormer, Mlp
from y2r.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid

class IntentTracker(nn.Module):
    def __init__(
        self,
        num_future_steps=10,
        hidden_size=384,
        model_resolution=(224, 224),
        add_space_attn=True,
        vit_model_name='dinov2_vits14',
        vit_frozen=False,
        time_depth=6,
        space_depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        p_drop_attn=0.0,
        frame_stack=1,
        cache_quantized_position_encoding=False,
    ):
        super(IntentTracker, self).__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack
        self.cache_quantized_position_encoding = cache_quantized_position_encoding

        # ViT encoder (provides all visual context)
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_model_name)
        self.vit.requires_grad_(not vit_frozen)
        
        # Embedding dimensions (following modern diffusion literature like DiT)
        # Position (2D spatial) = full model dim (most complex signal)
        # Temporal (1D discrete) = ~25% of model dim (simpler signal)
        self.position_dim = hidden_size  # 384
        self.temporal_dim = hidden_size // 4  # 96
        
        # Temporal embeddings for future predictions
        time_grid = torch.linspace(0, num_future_steps - 1, num_future_steps).reshape(
            1, num_future_steps, 1
        )
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.temporal_dim, time_grid[0])
        )

        # Observation temporal embeddings for past frames
        # Frames at relative times: [-frame_stack+1, ..., -1, 0]
        # NOTE: This is ADDED to ViT features, so it must be hidden_size (384), not temporal_dim
        obs_time_grid = torch.linspace(-(frame_stack - 1), 0, frame_stack).reshape(
            1, frame_stack, 1
        )
        self.register_buffer(
            "obs_time_emb", get_1d_sincos_pos_embed_from_grid(hidden_size, obs_time_grid[0])
        )
        
        # MLP to combine position and temporal encodings
        # Input: position_dim + temporal_dim = 384 + 96 = 480
        self.encoding_combiner = Mlp(
            in_features=self.position_dim + self.temporal_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.GELU,
            drop=0.0
        )

        # Transformer
        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=2,                # Just (Δx, Δy)
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            p_drop_attn=p_drop_attn,
            linear_layer_for_vis_conf=False,
        )
        
        # Build position encoding cache if using cached encoding
        if cache_quantized_position_encoding:
            self._build_position_cache()
    
    def extract_vit_features(self, frame):
        """
        Extract DINOv2 patch features from frames with temporal encoding.
        
        Args:
            frame: (B, T_obs, 3, 224, 224) - RGB images where T_obs = frame_stack
                   (ImageNet normalized in dataloader)
            
        Returns:
            scene_tokens: (B, T_obs*num_patches, feature_dim) - ViT patch embeddings 
                         with temporal encoding, concatenated across frames
        """
        B, T_obs, C, H, W = frame.shape
        assert T_obs == self.frame_stack, f"Expected {self.frame_stack} frames, got {T_obs}"
        
        # Flatten temporal dimension: (B, T_obs, 3, H, W) -> (B*T_obs, 3, H, W)
        frame_flat = frame.view(B * T_obs, C, H, W)
        
        # Process all frames through ViT
        vit_output = self.vit.forward_features(frame_flat)
        scene_tokens = vit_output['x_norm_patchtokens']  # (B*T_obs, num_patches, feature_dim)
        
        # Get dimensions
        num_patches = scene_tokens.shape[1]
        feature_dim = scene_tokens.shape[2]
        
        # Reshape to separate batch and temporal dimensions
        scene_tokens = scene_tokens.view(B, T_obs, num_patches, feature_dim)  # (B, T_obs, num_patches, feature_dim)
        
        # Add temporal encoding to distinguish frames
        # obs_time_emb: (1, T_obs, feature_dim) - has extra batch dim from get_1d_sincos_pos_embed_from_grid
        # Reshape to (1, T_obs, 1, feature_dim) to broadcast over batch and patches
        obs_time_encoding = self.obs_time_emb.unsqueeze(2)  # (1, T_obs, 1, feature_dim)
        scene_tokens = scene_tokens + obs_time_encoding  # (B, T_obs, num_patches, feature_dim)
        
        # Concatenate tokens from all frames: (B, T_obs, num_patches, feature_dim) -> (B, T_obs*num_patches, feature_dim)
        scene_tokens = scene_tokens.view(B, T_obs * num_patches, feature_dim)
        
        return scene_tokens
    
    def _build_position_cache(self):
        """Pre-compute position encodings for all 224x224 pixel locations."""
        print(f"Building position encoding cache (224x224x{self.position_dim})...")
        cache = torch.zeros(224, 224, self.position_dim)
        
        for x in range(224):
            for y in range(224):
                grid = torch.tensor([[float(x)], [float(y)]])
                enc = get_2d_sincos_pos_embed_from_grid(self.position_dim, grid)
                cache[x, y] = enc.squeeze(0).squeeze(0)
        
        self.register_buffer('pos_encoding_cache', cache)
        cache_size_mb = cache.numel() * 4 / 1e6
        print(f"  Cache size: {cache_size_mb:.1f} MB (dim={self.position_dim})")
    
    def get_position_encoding_cached(self, positions):
        """
        Get position encodings from pre-computed cache using bilinear interpolation.
        
        Args:
            positions: (..., 2) in [0,1] normalized coordinates
                      Can be (B, N, 2) or (B, N, T, 2)
        
        Returns:
            encodings: (..., H) same shape as input with last dim = hidden_size
        """
        # Convert to continuous pixel coordinates [0, 223]
        pixel_coords = positions * 223.0  # (..., 2)
        
        # Clamp to valid range
        pixel_coords = pixel_coords.clamp(0.0, 223.0)
        
        # Get integer coordinates for the 4 corners
        x0 = torch.floor(pixel_coords[..., 0]).long().clamp(0, 223)  # (...)
        y0 = torch.floor(pixel_coords[..., 1]).long().clamp(0, 223)  # (...)
        x1 = torch.ceil(pixel_coords[..., 0]).long().clamp(0, 223)   # (...)
        y1 = torch.ceil(pixel_coords[..., 1]).long().clamp(0, 223)   # (...)
        
        # Get fractional parts (these are differentiable)
        wx = pixel_coords[..., 0] - x0.float()  # (...) in [0, 1]
        wy = pixel_coords[..., 1] - y0.float()  # (...) in [0, 1]
        
        # Get encodings at 4 corners
        enc_00 = self.pos_encoding_cache[x0, y0]  # (..., H)
        enc_01 = self.pos_encoding_cache[x0, y1]  # (..., H)
        enc_10 = self.pos_encoding_cache[x1, y0]  # (..., H)
        enc_11 = self.pos_encoding_cache[x1, y1]  # (..., H)
        
        # Bilinear interpolation
        # First interpolate along x-axis
        enc_0 = enc_00 * (1 - wx.unsqueeze(-1)) + enc_10 * wx.unsqueeze(-1)  # (..., H)
        enc_1 = enc_01 * (1 - wx.unsqueeze(-1)) + enc_11 * wx.unsqueeze(-1)  # (..., H)
        
        # Then interpolate along y-axis
        encodings = enc_0 * (1 - wy.unsqueeze(-1)) + enc_1 * wy.unsqueeze(-1)  # (..., H)
        
        return encodings
    
    def forward(self, frame, query_coords):
        """
        Predict future point trajectories from observation frames.
        Tokens are purely positional + temporal, all visual info from ViT via cross-attention.

        Args:
            frame: (B, frame_stack, 3, 224, 224) - RGB frames (ImageNet normalized in dataloader)
            query_coords: (B, N, 2) - Initial (x, y) positions in [0, 1] normalized coordinates
            
        Returns:
            displacements: (B, N, T, 2) - Cumulative displacements in [0, 1] normalized space
        """
        B = frame.shape[0]
        N = query_coords.shape[1]
        T = self.num_future_steps
        
        # 1. Extract ViT scene features with temporal encoding (provides ALL visual context)
        scene_tokens = self.extract_vit_features(frame)  # (B, frame_stack*256, H)
        
        # 2. Position encoding - CONDITIONAL based on cache_quantized_position_encoding flag
        if self.cache_quantized_position_encoding:
            # Use cached lookup for initial positions (faster)
            pos_encoding = self.get_position_encoding_cached(query_coords)  # (B, N, H)
            # Expand to all future timesteps
            pos_encoding_expanded = pos_encoding.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, H)
        else:
            # Current behavior: compute position encoding from scratch
            coords_normalized = query_coords * 224.0  # Scale to [0, 224]
            grid = coords_normalized.permute(2, 0, 1)  # (2, B, N)
            pos_encoding = get_2d_sincos_pos_embed_from_grid(self.position_dim, grid)  # (1, B*N, position_dim)
            pos_encoding = pos_encoding.squeeze(0).reshape(B, N, self.position_dim)
            # Expand to all future timesteps
            pos_encoding_expanded = pos_encoding.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, position_dim)
        
        # 4. Prepare temporal encoding
        # time_emb shape: (1, T, temporal_dim), we need (B, N, T, temporal_dim)
        temporal_encoding = self.time_emb.view(1, 1, T, self.temporal_dim).expand(B, N, -1, -1)  # (B, N, T, temporal_dim)
        
        # 5. Combine position and temporal encodings using MLP
        # Concatenate position and temporal encodings
        combined_encoding = torch.cat([pos_encoding_expanded, temporal_encoding], dim=-1)
        # Shape: (B, N, T, 2*H)
        
        # Pass through MLP to combine into single hidden representation
        transformer_input = self.encoding_combiner(combined_encoding)
        # Shape: (B, N, T, H) - combined positional/temporal tokens
        
        # 6. Transformer queries scene via cross-attention for all visual info
        displacements = self.updateformer(
            transformer_input, 
            scene_tokens, 
            add_space_attn=True
        )
        # Shape: (B, N, T, 2) - direct prediction
        
        return displacements  # Cumulative displacements (0→1, 0→2, ..., 0→T)
    
    def compute_loss(self, batch):
        """
        Compute loss for training. Encapsulates loss computation within the model.
        
        Args:
            batch: dict with keys:
                - 'frames': (B, frame_stack, 3, H, W) - RGB frames
                - 'query_coords': (B, N, 2) - initial positions in [0, 1]
                - 'gt_disp_normalized': (B, N, T, 2) - GT displacements (normalized)
                - 'disp_std': (2,) - displacement std for loss scaling
        
        Returns:
            loss: scalar tensor
        """
        frames = batch['frames']
        query_coords = batch['query_coords']
        gt_disp_normalized = batch['gt_disp_normalized']
        disp_std = batch['disp_std']
        
        # Forward pass
        pred_disp = self(frames, query_coords)  # (B, N, T, 2)
        
        # Compute normalized displacement loss
        device = pred_disp.device
        std_tensor = torch.tensor(disp_std, device=device, dtype=pred_disp.dtype)
        
        # Denormalize to compute loss in pixel space
        pred_disp_denorm = pred_disp * std_tensor
        gt_disp_denorm = gt_disp_normalized * std_tensor
        
        # L2 distance
        error = torch.norm(pred_disp_denorm - gt_disp_denorm, dim=-1)  # (B, N, T)
        
        # Mask out t=0 (should always be zero displacement)
        loss_mask = torch.ones_like(error)
        loss_mask[:, :, 0] = 0.0  # Don't compute loss for t=0
        
        masked_error = error * loss_mask
        loss = masked_error.sum() / loss_mask.sum()
        
        return loss
    
    def predict(self, frames, query_coords, return_intermediate=False):
        """
        Prediction method for inference. Wrapper around forward for API consistency.
        
        Args:
            frames: (B, frame_stack, 3, H, W) - RGB frames
            query_coords: (B, N, 2) - initial positions in [0, 1]
            return_intermediate: bool - ignored for direct model (for API compatibility)
        
        Returns:
            pred_disp: (B, N, T, 2) - predicted displacements (normalized)
            (If return_intermediate=True, returns only pred_disp since no intermediate steps)
        """
        pred_disp = self(frames, query_coords)
        
        # Force t=0 to zero (no displacement at initial timestep)
        pred_disp[:, :, 0, :] = 0.0
        
        # For API compatibility with diffusion model
        # Direct model has no intermediate steps, so just return final prediction
        if return_intermediate:
            return pred_disp, []  # Empty list for intermediate
        return pred_disp
