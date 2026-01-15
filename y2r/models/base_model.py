"""
Base Intent Tracker model with shared components.

This module provides a base class that contains all shared functionality
across the different Intent Tracker variants (direct, autoregressive, diffusion).
"""

import torch
import torch.nn as nn
import timm
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from y2r.models.blocks import EfficientUpdateFormer, DiTUpdateFormer, Mlp
from y2r.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_embedding
from y2r.models.model_utils import extend_vit_to_rgbd
from y2r.models.model_config import get_model_config, ENCODING_DIMS


class BaseIntentTracker(nn.Module, ABC):
    """
    Abstract base class for Intent Tracker models.
    
    This class provides shared functionality including:
    - ViT encoder (DINOv3) for visual feature extraction
    - UMT5 text encoder for language conditioning
    - Temporal embeddings for observation and prediction timesteps
    - Common encoding utilities (position encoding, etc.)
    - Encoder freezing/unfreezing methods for curriculum learning
    
    Subclasses must implement:
    - forward(): Model forward pass
    - compute_loss(): Loss computation for training
    - predict(): Inference-time prediction
    """
    
    def __init__(
        self,
        model_size: str = 's',
        num_future_steps: int = 10,
        model_resolution: tuple = (256, 256),
        add_space_attn: bool = True,
        vit_frozen: bool = False,
        p_drop_attn: float = 0.0,
        frame_stack: int = 1,
        track_type: str = '2d',
        from_pretrained: bool = True,
        hand_mode: Optional[str] = None,
        text_mode: bool = False,
    ):
        """
        Initialize the base Intent Tracker.
        
        Args:
            model_size: Size preset ('s', 'b', 'l') - determines hidden_size, num_heads, etc.
            num_future_steps: Number of future timesteps to predict
            model_resolution: Input image resolution (height, width)
            add_space_attn: Whether to add spatial attention in transformer
            vit_frozen: Whether to freeze ViT encoder initially
            p_drop_attn: Attention dropout probability
            frame_stack: Number of observation frames
            track_type: '2d' or '3d' - determines coordinate dimensionality
            from_pretrained: Whether to load pretrained ViT weights
            hand_mode: 'left', 'right', 'both', or None
            text_mode: Whether to enable text conditioning
        """
        super().__init__()
        
        # Get model configuration from size preset
        cfg = get_model_config(model_size)
        hidden_size = cfg['hidden_size']
        num_heads = cfg['num_heads']
        mlp_ratio = cfg['mlp_ratio']
        time_depth = cfg['time_depth']
        vit_model_name = cfg['vit_model_name']
        
        # Store configuration
        self.model_size = model_size
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.time_depth = time_depth
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack
        self.track_type = track_type
        self.coord_dim = 3 if track_type == '3d' else 2
        self.hand_mode = hand_mode
        self.text_mode = text_mode
        self.add_space_attn = add_space_attn
        self.p_drop_attn = p_drop_attn
        
        # Store encoding dimensions (fixed, independent of model size)
        self.enc_dims = ENCODING_DIMS
        
        # =========================================================================
        # ViT Encoder (DINOv3 via timm)
        # =========================================================================
        self.vit = timm.create_model(vit_model_name, pretrained=from_pretrained)
        
        # Extend ViT to handle RGBD input for 3D mode
        if track_type == '3d':
            extend_vit_to_rgbd(self.vit)
        
        self.vit.requires_grad_(not vit_frozen)
        
        # =========================================================================
        # Temporal Embeddings
        # =========================================================================
        # Future prediction temporal embeddings (fixed 64 dim from ENCODING_DIMS)
        # Note: get_1d_sincos_pos_embed_from_grid returns (1, T, dim), we squeeze to (T, dim)
        time_grid = torch.linspace(0, num_future_steps - 1, num_future_steps).reshape(
            1, num_future_steps, 1
        )
        time_emb = get_1d_sincos_pos_embed_from_grid(self.enc_dims['temporal'], time_grid[0])
        self.register_buffer("time_emb", time_emb.squeeze(0))  # (T, temporal_dim)
        
        # Observation temporal embeddings for past frames
        # NOTE: This is ADDED to ViT features, so it must match ViT hidden_size
        obs_time_grid = torch.linspace(-(frame_stack - 1), 0, frame_stack).reshape(
            1, frame_stack, 1
        )
        obs_time_emb = get_1d_sincos_pos_embed_from_grid(hidden_size, obs_time_grid[0])
        self.register_buffer("obs_time_emb", obs_time_emb.squeeze(0))  # (frame_stack, hidden_size)
        
        # =========================================================================
        # Text Encoder (UMT5) - Optional
        # =========================================================================
        if text_mode:
            from transformers import AutoTokenizer, T5EncoderModel
            text_model_name = cfg['text_model_name']
            text_embed_dim = cfg['text_embed_dim']

            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder = T5EncoderModel.from_pretrained(text_model_name)

            # Project UMT5 embeddings to hidden_size if dimensions don't match
            if text_embed_dim != hidden_size:
                self.text_proj = nn.Linear(text_embed_dim, hidden_size)
            else:
                self.text_proj = nn.Identity()

            # Freeze text encoder (UMT5 is always frozen)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
    
    # =========================================================================
    # Shared Feature Extraction Methods
    # =========================================================================
    
    def extract_vit_features(self, frame: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract DINOv3 patch features from frames with temporal encoding.
        
        Args:
            frame: (B, T_obs, 3, H, W) - RGB images where T_obs = frame_stack
                   (ImageNet normalized in dataloader)
            depth: (B, T_obs, 1, H, W) - Depth maps for 3D mode, None for 2D
            
        Returns:
            scene_tokens: (B, T_obs*num_patches, feature_dim) - ViT patch embeddings 
                         with temporal encoding, concatenated across frames
        """
        B, T_obs, C, H, W = frame.shape
        assert T_obs == self.frame_stack, f"Expected {self.frame_stack} frames, got {T_obs}"
        
        # Concatenate RGB + Depth for 3D mode -> RGBD
        if self.track_type == '3d' and depth is not None:
            frame = torch.cat([frame, depth], dim=2)  # (B, T_obs, 4, H, W)
        
        # Flatten temporal dimension: (B, T_obs, C, H, W) -> (B*T_obs, C, H, W)
        frame_flat = frame.view(B * T_obs, -1, H, W)
        
        # Process all frames through ViT (timm returns tensor with prefix tokens)
        vit_output = self.vit.forward_features(frame_flat)
        # Skip prefix tokens (CLS + register tokens) to get patch tokens only
        scene_tokens = vit_output[:, self.vit.num_prefix_tokens:, :]  # (B*T_obs, num_patches, feature_dim)
        
        # Get dimensions
        num_patches = scene_tokens.shape[1]
        feature_dim = scene_tokens.shape[2]
        
        # Reshape to separate batch and temporal dimensions
        scene_tokens = scene_tokens.view(B, T_obs, num_patches, feature_dim)
        
        # Add temporal encoding to distinguish frames
        # obs_time_emb is (T_obs, feature_dim), need (1, T_obs, 1, feature_dim) for broadcasting
        obs_time_encoding = self.obs_time_emb.unsqueeze(0).unsqueeze(2)  # (1, T_obs, 1, feature_dim)
        scene_tokens = scene_tokens + obs_time_encoding
        
        # Concatenate tokens from all frames
        scene_tokens = scene_tokens.view(B, T_obs * num_patches, feature_dim)
        
        return scene_tokens
    
    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """
        Encode batch of text strings using UMT5 encoder.

        Args:
            text_list: List of B text strings

        Returns:
            text_tokens: (B, L, hidden_size) - Sequence of text token embeddings
                        where L is the number of tokens (varies by text length)
        """
        if not self.text_mode:
            raise RuntimeError("encode_text called but text_mode is False")

        # Tokenize text
        inputs = self.text_tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        # Move to same device as model
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode with UMT5 text encoder (always no grad since frozen)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        # Get sequence of token embeddings (B, L, text_embed_dim)
        text_emb = outputs.last_hidden_state

        # Project to hidden_size: (B, L, text_embed_dim) -> (B, L, hidden_size)
        text_tokens = self.text_proj(text_emb)

        return text_tokens
    
    # =========================================================================
    # Position Encoding Utilities
    # =========================================================================
    
    def _encode_position(self, coords: torch.Tensor, coord_type: str = 'track') -> torch.Tensor:
        """
        Encode 2D or 3D coordinates using sinusoidal positional encoding.
        
        Args:
            coords: (B, N, 2) or (B, N, 3) coordinates in [0, 1] range
            coord_type: 'track' for track points, 'hand' for hand positions
            
        Returns:
            pos_emb: (B, N, pos_dim) sinusoidal position embeddings
        """
        # Get appropriate embedding dimension
        if coord_type == 'track':
            total_dim = self.enc_dims['track_position']
        else:  # hand
            total_dim = self.enc_dims['hand_position']
        
        coord_dim = coords.shape[-1]
        
        if coord_dim == 2:
            # Use get_2d_embedding which expects (B, N, 2)
            # It returns (B, N, C*2) when cat_coords=False
            dim_per_coord = total_dim // 2
            pos_emb = get_2d_embedding(coords, dim_per_coord, cat_coords=False)  # (B, N, total_dim)
            
        else:  # 3D - encode XY and Z with clean dimension split
            # Split: 50% for XY (2D sincos), 50% for Z (sincos via same method)
            # This ensures dimensions add up correctly
            dim_xy = total_dim // 2  # 64 for hand_position=128
            dim_z = total_dim - dim_xy  # 64 for hand_position=128
            dim_per_xy = dim_xy // 2  # 32 for get_2d_embedding
            
            # Encode XY with 2D sincos
            xy_coords = coords[..., :2]  # (B, N, 2)
            xy_emb = get_2d_embedding(xy_coords, dim_per_xy, cat_coords=False)  # (B, N, dim_xy)
            
            # Encode Z with 1D sincos (same approach as xy for each coord)
            z_coords = coords[..., 2:3]  # (B, N, 1)
            # Create z embedding using same sincos formula as get_2d_embedding
            B, N, _ = z_coords.shape
            dim_z_half = dim_z // 2
            div_term = (
                torch.arange(0, dim_z, 2, device=coords.device, dtype=torch.float32) * (1000.0 / dim_z)
            ).reshape(1, 1, dim_z_half)
            z_emb = torch.zeros(B, N, dim_z, device=coords.device, dtype=torch.float32)
            z_emb[:, :, 0::2] = torch.sin(z_coords * div_term)
            z_emb[:, :, 1::2] = torch.cos(z_coords * div_term)
            
            pos_emb = torch.cat([xy_emb, z_emb], dim=-1)  # (B, N, total_dim)
        
        return pos_emb
    
    # =========================================================================
    # Encoder Freezing/Unfreezing Methods (Curriculum Learning)
    # =========================================================================
    
    def freeze_encoders(self):
        """Freeze both ViT and text encoder (if text_mode is enabled)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        if self.text_mode and hasattr(self, 'text_encoder'):
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        print("Encoders frozen (ViT" + (" + UMT5" if self.text_mode else "") + ")")
    
    def unfreeze_encoders(self):
        """Unfreeze both ViT and text encoder (if text_mode is enabled)."""
        for param in self.vit.parameters():
            param.requires_grad = True
        if self.text_mode and hasattr(self, 'text_encoder'):
            for param in self.text_encoder.parameters():
                param.requires_grad = True
        print("Encoders unfrozen (ViT" + (" + UMT5" if self.text_mode else "") + ")")
    
    def set_encoders_frozen(self, frozen: bool):
        """Set the frozen state for both encoders."""
        if frozen:
            self.freeze_encoders()
        else:
            self.unfreeze_encoders()
    
    def get_encoder_params(self) -> List[nn.Parameter]:
        """Return list of encoder parameters (ViT + text encoder if enabled)."""
        params = list(self.vit.parameters())
        if self.text_mode and hasattr(self, 'text_encoder'):
            params.extend(list(self.text_encoder.parameters()))
        return params
    
    def get_non_encoder_params(self) -> List[nn.Parameter]:
        """Return list of non-encoder parameters (transformer head, MLPs, etc.)."""
        encoder_param_ids = set(id(p) for p in self.get_encoder_params())
        return [p for p in self.parameters() if id(p) not in encoder_param_ids]
    
    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Returns:
            Dict containing model outputs (varies by model type)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            batch: Dict containing training data
            
        Returns:
            Dict containing 'total_loss' and component losses
        """
        pass
    
    @abstractmethod
    def predict(self, frames: torch.Tensor, query_coords: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction.
        
        Args:
            frames: (B, frame_stack, C, H, W) input frames
            query_coords: (B, N, coord_dim) query coordinates
            **kwargs: Model-specific arguments
            
        Returns:
            Dict containing predictions
        """
        pass
    
    # =========================================================================
    # Helper Methods for Subclasses
    # =========================================================================
    
    def _create_updateformer(self, use_dit: bool = False):
        """
        Create the UpdateFormer transformer with current configuration.

        Args:
            use_dit: If True, use DiTUpdateFormer with adaLN conditioning.
                    If False, use standard EfficientUpdateFormer.

        Returns:
            UpdateFormer module (either DiT or standard variant)
        """
        if use_dit:
            return DiTUpdateFormer(
                depth=self.time_depth,
                input_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                output_dim=self.hidden_size,
                mlp_ratio=self.mlp_ratio,
                add_space_attn=self.add_space_attn,
                p_drop_attn=self.p_drop_attn,
            )
        else:
            return EfficientUpdateFormer(
                depth=self.time_depth,
                input_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                output_dim=self.hidden_size,
                mlp_ratio=self.mlp_ratio,
                add_space_attn=self.add_space_attn,
                p_drop_attn=self.p_drop_attn,
                linear_layer_for_vis_conf=False,
            )
    
    def _create_track_encoder(self, input_dim: int) -> Mlp:
        """Create track token encoder MLP."""
        return Mlp(
            in_features=input_dim,
            hidden_features=self.hidden_size,
            out_features=self.hidden_size,
            act_layer=nn.GELU,
            drop=0.0,
        )
    
    def _create_hand_encoder(self, input_dim: int) -> Mlp:
        """Create hand token encoder MLP."""
        return Mlp(
            in_features=input_dim,
            hidden_features=self.hidden_size,
            out_features=self.hidden_size,
            act_layer=nn.GELU,
            drop=0.0,
        )

