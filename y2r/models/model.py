import torch
import torch.nn as nn

from y2r.models.blocks import EfficientUpdateFormer
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
    ):
        super(IntentTracker, self).__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack

        # ViT encoder (provides all visual context)
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_model_name)
        self.vit.requires_grad_(not vit_frozen)
        
        # Temporal embeddings for future predictions (dimension = hidden_size = H for ADDITION)
        time_grid = torch.linspace(0, num_future_steps - 1, num_future_steps).reshape(
            1, num_future_steps, 1
        )
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(hidden_size, time_grid[0])
        )

        # Observation temporal embeddings for past frames
        # Frames at relative times: [-frame_stack+1, ..., -1, 0]
        obs_time_grid = torch.linspace(-(frame_stack - 1), 0, frame_stack).reshape(
            1, frame_stack, 1
        )
        self.register_buffer(
            "obs_time_emb", get_1d_sincos_pos_embed_from_grid(hidden_size, obs_time_grid[0])
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
        
        # 2. Compute 2D sincos position encoding (same approach as temporal)
        # query_coords are in [0, 1], scale to reasonable range for sincos
        coords_normalized = query_coords * 224.0  # Scale to [0, 224] like image coordinates
        
        # Prepare grid format: (B, N, 2) → (2, B, N) where [0] is x, [1] is y
        grid = coords_normalized.permute(2, 0, 1)  # (2, B, N)
        
        # Apply 2d sincos embedding
        pos_encoding = get_2d_sincos_pos_embed_from_grid(self.hidden_size, grid)  # (1, B*N, H)
        
        # Reshape to (B, N, H)
        pos_encoding = pos_encoding.squeeze(0).reshape(B, N, self.hidden_size)
        
        # 3. Expand position encoding to all future timesteps
        pos_encoding_expanded = pos_encoding.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, H)
        
        # 4. Prepare temporal encoding
        # time_emb shape: (1, T, H), we need (B, N, T, H)
        temporal_encoding = self.time_emb.view(1, 1, T, self.hidden_size).expand(B, N, -1, -1)  # (B, N, T, H)
        
        # 5. ADD position + temporal (no visual features!)
        transformer_input = pos_encoding_expanded + temporal_encoding
        # Shape: (B, N, T, H) - purely positional/temporal tokens
        
        # 6. Transformer queries scene via cross-attention for all visual info
        displacements = self.updateformer(
            transformer_input, 
            scene_tokens, 
            add_space_attn=True
        )
        # Shape: (B, N, T, 2) - direct prediction
        
        return displacements  # Cumulative displacements (0→1, 0→2, ..., 0→T)
