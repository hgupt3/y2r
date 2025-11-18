"""
Diffusion-based Intent Tracker model.

This module implements a diffusion model for trajectory prediction, following the
Diffusion Policy architecture. The model denoises trajectory outputs using DDIM sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler

from y2r.models.blocks import EfficientUpdateFormer, Mlp
from y2r.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid


class DiffusionIntentTracker(nn.Module):
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
        num_diffusion_steps=100,
        beta_schedule='squaredcos_cap_v2',
        cache_quantized_position_encoding=False,
        disp_mean=None,  # (2,) displacement mean for normalization
        disp_std=None,   # (2,) displacement std for normalization
    ):
        super(DiffusionIntentTracker, self).__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack
        self.num_diffusion_steps = num_diffusion_steps
        self.cache_quantized_position_encoding = cache_quantized_position_encoding
        
        # Register displacement normalization stats as buffers
        if disp_mean is not None and disp_std is not None:
            self.register_buffer('disp_mean', torch.tensor(disp_mean, dtype=torch.float32))
            self.register_buffer('disp_std', torch.tensor(disp_std, dtype=torch.float32))
        else:
            # Default values if not provided (will be set later)
            self.register_buffer('disp_mean', torch.zeros(2))
            self.register_buffer('disp_std', torch.ones(2))

        # ViT encoder (provides all visual context)
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_model_name)
        self.vit.requires_grad_(not vit_frozen)
        
        # Embedding dimensions (following modern diffusion literature like DiT)
        # Position (2D spatial) = full model dim (most complex signal)
        # Temporal/Diffusion timestep (1D discrete) = ~25% of model dim (simpler signals)
        self.position_dim = hidden_size  # 384
        self.temporal_dim = hidden_size // 4  # 96
        self.diffusion_timestep_dim = hidden_size // 4  # 96
        
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
        
        # Diffusion timestep embedding (for encoding which diffusion step we're at)
        # We'll create embeddings for timesteps 0 to num_diffusion_steps
        diffusion_time_grid = torch.linspace(0, num_diffusion_steps - 1, num_diffusion_steps).reshape(
            1, num_diffusion_steps, 1
        )
        self.register_buffer(
            "diffusion_time_emb", get_1d_sincos_pos_embed_from_grid(self.diffusion_timestep_dim, diffusion_time_grid[0])
        )
        
        # MLP to combine position, temporal, and diffusion timestep encodings
        # Input: position_dim + temporal_dim + diffusion_timestep_dim = 384 + 96 + 96 = 576
        self.encoding_combiner = Mlp(
            in_features=self.position_dim + self.temporal_dim + self.diffusion_timestep_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.GELU,
            drop=0.0
        )

        # Transformer (predicts noise instead of direct displacements)
        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=2,                # Predict noise (Δx, Δy)
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            p_drop_attn=p_drop_attn,
            linear_layer_for_vis_conf=False,
        )
        
        # Diffusion scheduler (DDIM)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=False,  # Don't clip, we're in normalized space
            prediction_type='epsilon',  # Predict noise
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
        print(f"  Cache size: {cache.numel() * 4 / 1e6:.1f} MB")
    
    def get_position_encoding_cached(self, positions):
        """
        Get position encodings from pre-computed cache.
        
        Args:
            positions: (..., 2) in [0,1] normalized coordinates
                      Can be (B, N, 2) or (B, N, T, 2)
        
        Returns:
            encodings: (..., H) same shape as input with last dim = hidden_size
        """
        # Quantize to pixel coordinates [0, 223]
        pixel_coords = (positions * 223.0).long().clamp(0, 223)
        
        # Lookup in cache
        encodings = self.pos_encoding_cache[pixel_coords[..., 0], pixel_coords[..., 1]]
        
        return encodings
    
    def forward(self, frames, query_coords, noisy_disp, timestep, scene_tokens=None):
        """
        Forward pass: predict noise in noisy displacements.
        OPTIMIZED: Supports cached vision features for efficiency.
        
        Args:
            frames: (B, frame_stack, 3, 224, 224) - RGB frames (ImageNet normalized)
                   Can be None if scene_tokens is provided.
            query_coords: (B, N, 2) - Initial (x, y) positions in [0, 1] normalized coordinates
            noisy_disp: (B, N, T, 2) - Noisy displacements (normalized space)
            timestep: (B,) - Diffusion timestep for each sample in batch
            scene_tokens: (B, T_obs*num_patches, H) - Optional pre-computed ViT features.
                         If provided, frames can be None. If not provided, computed from frames.
            
        Returns:
            predicted_noise: (B, N, T, 2) - Predicted noise in displacements
        """
        B = query_coords.shape[0]
        N = query_coords.shape[1]
        T = self.num_future_steps
        
        # 1. Extract or use cached ViT scene features
        if scene_tokens is None:
            scene_tokens = self.extract_vit_features(frames)  # (B, frame_stack*256, H)
        
        # 2. Denormalize noisy_disp to [0,1] coordinate space before adding to query_coords
        # noisy_disp is in normalized space (mean-subtracted, std-divided)
        # query_coords is in [0, 1] coordinate space
        # We need to denormalize noisy_disp first!
        disp_mean = self.disp_mean.to(dtype=noisy_disp.dtype)
        disp_std = self.disp_std.to(dtype=noisy_disp.dtype)
        noisy_disp_denorm = noisy_disp * disp_std + disp_mean  # (B, N, T, 2) in [0,1] space
        
        # 3. Position encoding of PREDICTED positions (where the point IS at each timestep)
        predicted_positions = query_coords.unsqueeze(2) + noisy_disp_denorm  # (B, N, T, 2) in [0,1]
        
        if self.cache_quantized_position_encoding:
            # Use cached lookup (fast)
            pos_encoding_expanded = self.get_position_encoding_cached(predicted_positions)  # (B, N, T, H)
        else:
            # Compute position encoding for each predicted position (slower but more precise)
            # Process each timestep separately
            pos_encodings = []
            for t in range(T):
                pred_pos_t = predicted_positions[:, :, t, :]  # (B, N, 2) in [0,1]
                coords_norm_t = pred_pos_t * 224.0  # Scale to [0, 224]
                grid_t = coords_norm_t.permute(2, 0, 1)  # (2, B, N)
                pos_enc_t = get_2d_sincos_pos_embed_from_grid(self.position_dim, grid_t)  # (1, B*N, position_dim)
                pos_enc_t = pos_enc_t.squeeze(0).reshape(B, N, self.position_dim)
                pos_encodings.append(pos_enc_t)
            pos_encoding_expanded = torch.stack(pos_encodings, dim=2)  # (B, N, T, position_dim)
        
        # 4. Prepare temporal encoding for future timesteps
        temporal_encoding = self.time_emb.view(1, 1, T, self.temporal_dim).expand(B, N, -1, -1)  # (B, N, T, temporal_dim)
        
        # 5. Prepare diffusion timestep encoding
        timestep_encoding = self.diffusion_time_emb[0, timestep, :]  # (B, diffusion_timestep_dim)
        timestep_encoding = timestep_encoding.view(B, 1, 1, self.diffusion_timestep_dim).expand(-1, N, T, -1)  # (B, N, T, diffusion_timestep_dim)
        
        # 6. Combine all encodings using MLP (NO trajectory tokens!)
        # Concatenate position, temporal, and diffusion timestep encodings
        combined_encoding = torch.cat([pos_encoding_expanded, temporal_encoding, timestep_encoding], dim=-1)
        # Shape: (B, N, T, 3*H)
        
        # Pass through MLP to combine into single hidden representation
        transformer_input = self.encoding_combiner(combined_encoding)
        # Shape: (B, N, T, H)
        
        # 7. Transformer predicts noise via cross-attention to scene
        predicted_noise = self.updateformer(
            transformer_input, 
            scene_tokens, 
            add_space_attn=True
        )
        # Shape: (B, N, T, 2) - noise prediction
        
        return predicted_noise
    
    def compute_loss(self, batch):
        """
        Compute diffusion loss for training.
        
        Args:
            batch: dict with keys:
                - 'frames': (B, frame_stack, 3, H, W) - RGB frames
                - 'query_coords': (B, N, 2) - initial positions in [0, 1]
                - 'gt_disp_normalized': (B, N, T, 2) - GT displacements (normalized)
                - 'disp_std': (2,) - not used in diffusion loss
        
        Returns:
            loss: scalar tensor (MSE between predicted and actual noise)
        """
        frames = batch['frames']
        query_coords = batch['query_coords']
        gt_disp_normalized = batch['gt_disp_normalized']
        
        B, N, T, _ = gt_disp_normalized.shape
        device = gt_disp_normalized.device
        
        # Sample random diffusion timestep for each sample in batch
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(gt_disp_normalized)
        
        # Add noise to GT displacements using the scheduler
        noisy_disp = self.noise_scheduler.add_noise(gt_disp_normalized, noise, timestep)
        
        # Force t=0 to represent "zero displacement in real space" in normalized coordinates
        # Zero in real space = 0.0
        # Zero in normalized space = (0.0 - mean) / std = -mean/std
        zero_disp_normalized = -self.disp_mean / self.disp_std  # (2,)
        noisy_disp[:, :, 0, :] = zero_disp_normalized
        
        # Set target noise at t=0 to zero (doesn't matter since we mask t=0 in loss)
        noise[:, :, 0, :] = 0.0
        
        # Predict the noise
        predicted_noise = self(frames, query_coords, noisy_disp, timestep)
        
        # Compute MSE loss with masking for t=0
        # t=0 should always be zero displacement (no movement yet)
        # Create mask: 0 for t=0, 1 for t>0
        loss_mask = torch.ones_like(noise)
        loss_mask[:, :, 0, :] = 0.0  # Mask out t=0
        
        # Apply mask to both predicted and target noise
        masked_pred = predicted_noise * loss_mask
        masked_target = noise * loss_mask
        
        # Compute loss only on non-masked elements
        num_elements = loss_mask.sum()
        loss = F.mse_loss(masked_pred, masked_target, reduction='sum') / num_elements
        
        return loss
    
    @torch.no_grad()
    def predict(self, frames, query_coords, num_inference_steps=10, return_intermediate=False):
        """
        Predict clean trajectories using DDIM sampling with cached vision features.
        
        Args:
            frames: (B, frame_stack, 3, H, W) - RGB frames
            query_coords: (B, N, 2) - initial positions in [0, 1]
            num_inference_steps: int - number of denoising steps (default 10)
            return_intermediate: bool - if True, return intermediate predictions
        
        Returns:
            clean_disp: (B, N, T, 2) - final clean displacements (normalized)
            intermediate: list of (B, N, T, 2) tensors at each step (if return_intermediate=True)
        """
        B = frames.shape[0]
        N = query_coords.shape[1]
        T = self.num_future_steps
        device = frames.device
        
        # OPTIMIZATION: Extract vision features ONCE (not at every diffusion step)
        scene_tokens = self.extract_vit_features(frames)  # (B, frame_stack*256, H)
        
        # Set the number of inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Start from pure noise (match the dtype of the model for FP16 support)
        noisy_disp = torch.randn(B, N, T, 2, device=device, dtype=scene_tokens.dtype)
        
        # Force t=0 to represent "zero displacement in real space" in normalized coordinates
        # Zero in real space = 0.0
        # Zero in normalized space = (0.0 - mean) / std = -mean/std
        zero_disp_normalized = (-self.disp_mean / self.disp_std).to(dtype=scene_tokens.dtype)
        noisy_disp[:, :, 0, :] = zero_disp_normalized
        
        # Store intermediate predictions if requested
        intermediate = [] if return_intermediate else None
        
        # DDIM sampling loop with cached vision features
        for t in self.noise_scheduler.timesteps:
            # Create timestep tensor for batch
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise using cached scene tokens (pass scene_tokens to skip re-extraction)
            predicted_noise = self(
                frames=None,  # Not needed since we have scene_tokens
                query_coords=query_coords,
                noisy_disp=noisy_disp,
                timestep=timestep,
                scene_tokens=scene_tokens  # Use cached features
            )
            
            # Denoise using scheduler
            noisy_disp = self.noise_scheduler.step(
                predicted_noise, t, noisy_disp
            ).prev_sample
            
            # Ensure t=0 stays at "zero displacement" after denoising
            noisy_disp[:, :, 0, :] = zero_disp_normalized
            
            # Store intermediate result if requested
            if return_intermediate:
                intermediate.append(noisy_disp.clone())
        
        clean_disp = noisy_disp
        
        if return_intermediate:
            return clean_disp, intermediate
        else:
            return clean_disp

