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
        from_pretrained=True,
    ):
        super(IntentTracker, self).__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack
        self.cache_quantized_position_encoding = cache_quantized_position_encoding

        # ViT encoder (provides all visual context)
        if from_pretrained:
            # Load pretrained weights from torch.hub (for training)
            self.vit = torch.hub.load('facebookresearch/dinov2', vit_model_name)
        else:
            # Create model architecture without pretrained weights (for inference from checkpoint)
            # Import from vendored dinov2 in thirdparty/ to avoid network dependency
            import sys
            from pathlib import Path
            # Add vendored dinov2 to path
            project_root = Path(__file__).parent.parent.parent
            dinov2_path = project_root / 'thirdparty' / 'dinov2'
            if str(dinov2_path) not in sys.path:
                sys.path.insert(0, str(dinov2_path))
            from dinov2.hub.backbones import (
                dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14,
                dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg
            )
            # Map model name to function
            model_map = {
                'dinov2_vits14': dinov2_vits14,
                'dinov2_vitb14': dinov2_vitb14,
                'dinov2_vitl14': dinov2_vitl14,
                'dinov2_vitg14': dinov2_vitg14,
                'dinov2_vits14_reg': dinov2_vits14_reg,
                'dinov2_vitb14_reg': dinov2_vitb14_reg,
                'dinov2_vitl14_reg': dinov2_vitl14_reg,
                'dinov2_vitg14_reg': dinov2_vitg14_reg,
            }
            if vit_model_name not in model_map:
                raise ValueError(f"Unknown ViT model: {vit_model_name}")
            # Create model without pretrained weights (pretrained=False)
            self.vit = model_map[vit_model_name](pretrained=False)
        
        self.vit.requires_grad_(not vit_frozen)
        
        # Embedding dimensions (following modern diffusion literature like DiT)
        # Query (2D spatial) = ~50% of model dim for explicit localization
        # Temporal (1D discrete) = ~50% of model dim
        self.query_dim = hidden_size
        self.temporal_dim = hidden_size
        
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

        # Transformer
        self.updateformer = EfficientUpdateFormer(
            depth=time_depth,
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=2,                # Just (Δx, Δy)
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            p_drop_attn=p_drop_attn,
            linear_layer_for_vis_conf=False,
        )
    
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
        
        # 2. Query encoding: encode initial query coordinates (where trajectories start)
        # Use 2D sinusoidal positional encoding for spatial localization
        coords_pixel = query_coords * 224.0  # (B, N, 2) -> [0, 224]
        grid = coords_pixel.permute(2, 0, 1)  # (2, B, N)
        query_encoding = get_2d_sincos_pos_embed_from_grid(self.query_dim, grid)  # (1, B*N, query_dim)
        query_encoding = query_encoding.squeeze(0).reshape(B, N, self.query_dim)
        # Expand to all future timesteps
        query_encoding_expanded = query_encoding.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, query_dim)
        
        # 3. Prepare temporal encoding
        # time_emb shape: (1, T, temporal_dim), we need (B, N, T, temporal_dim)
        temporal_encoding = self.time_emb.view(1, 1, T, self.temporal_dim).expand(B, N, -1, -1)  # (B, N, T, temporal_dim)
        
        # 4. Combine query and temporal encodings using MLP
        # Add query and temporal encodings
        transformer_input = query_encoding_expanded + temporal_encoding
        # Shape: (B, N, T, H) - combined query/temporal tokens
        
        # 5. Transformer queries scene via cross-attention for all visual info
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
        
        # For API compatibility with diffusion model
        # Direct model has no intermediate steps, so just return final prediction
        if return_intermediate:
            return pred_disp, []  # Empty list for intermediate
        return pred_disp
