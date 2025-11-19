"""
Diffusion-based Intent Tracker model.

This module implements a diffusion model for trajectory prediction, following the
Diffusion Policy architecture. The model denoises trajectory outputs using DDIM sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

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
        self.total_token_steps = num_future_steps + 1  # include conditioning slot
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
        
        # Embedding dimensions
        self.position_dim = 384  # Sin/cos encoding of absolute positions
        self.traj_dim = 96      # Linear projection of normalized noisy displacement
        self.temporal_dim = 96
        self.diffusion_timestep_dim = 96
        
        # Temporal embeddings for future predictions
        time_grid = torch.linspace(0, self.total_token_steps - 1, self.total_token_steps).reshape(
            1, self.total_token_steps, 1
        )
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.temporal_dim, time_grid[0])
        )

        # Observation temporal embeddings for past frames
        # Frames at relative times: [-frame_stack+1, ..., -1, 0]
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
        
        # Trajectory embedding: project normalized noisy displacement
        self.traj_embed = nn.Linear(2, self.traj_dim)
        
        # MLP to combine position, trajectory, temporal, and diffusion timestep encodings
        # Input: position_dim + traj_dim + temporal_dim + diffusion_timestep_dim
        self.encoding_combiner = Mlp(
            in_features=self.position_dim + self.traj_dim + self.temporal_dim + self.diffusion_timestep_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.GELU,
            drop=0.0
        )

        # Transformer (predicts noise instead of direct displacements)
        self.updateformer = EfficientUpdateFormer(
            depth=time_depth,
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=2,                # Predict noise (Δx, Δy)
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            p_drop_attn=p_drop_attn,
            linear_layer_for_vis_conf=False,
        )
        
        # Diffusion scheduler (DDPM for training)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=False,  # Don't clip, we're in normalized space
            prediction_type='epsilon',  # Predict noise
        )
        
        # Inference scheduler (DDIM for fast sampling)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=False,
            prediction_type='epsilon',
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
    
    def forward(self, frames, query_coords, noisy_traj, timestep, scene_tokens=None):
        """
        Forward pass: predict noise in noisy trajectory tokens (conditioned slot + future steps).
        OPTIMIZED: Supports cached vision features for efficiency.
        
        Args:
            frames: (B, frame_stack, 3, 224, 224) - RGB frames (ImageNet normalized)
                   Can be None if scene_tokens is provided.
            query_coords: (B, N, 2) - Initial (x, y) positions in [0, 1]
            noisy_traj: (B, N, T+1, 2) - Noisy tokens (t=0 conditioned slot + T future steps)
            timestep: (B,) - Diffusion timestep for each sample in batch
            scene_tokens: (B, T_obs*num_patches, H) - Optional pre-computed ViT features.
                         If provided, frames can be None. If not provided, computed from frames.
            
        Returns:
            predicted_noise: (B, N, T+1, 2) - Predicted noise for each token
        """
        B = noisy_traj.shape[0]
        N = noisy_traj.shape[1]
        T_tokens = self.total_token_steps
        
        # 1. Extract or use cached ViT scene features
        if scene_tokens is None:
            scene_tokens = self.extract_vit_features(frames)  # (B, frame_stack*256, H)
        
        # 2. Encode absolute positions with high-frequency sin/cos
        disp_mean = self.disp_mean.to(dtype=noisy_traj.dtype, device=noisy_traj.device)
        disp_std = self.disp_std.to(dtype=noisy_traj.dtype, device=noisy_traj.device)
        noisy_disp_denorm = noisy_traj * disp_std + disp_mean  # (B, N, T+1, 2)
        current_pos = (query_coords.unsqueeze(2) + noisy_disp_denorm).clamp(0.0, 1.0)
        pos_grid = current_pos.reshape(B * N * T_tokens, 2).permute(1, 0)  # (2, B*N*T_tokens)
        pos_encoding = get_2d_sincos_pos_embed_from_grid(self.position_dim, pos_grid)
        pos_encoding = pos_encoding.squeeze(0).reshape(B, N, T_tokens, self.position_dim)
        
        # 3. Trajectory embedding: direct noisy displacement signal
        traj_embedding = self.traj_embed(noisy_traj)  # (B, N, T+1, traj_dim)
        
        # 4. Prepare temporal encoding for all token steps (including conditioned slot)
        temporal_encoding = self.time_emb.view(1, 1, T_tokens, self.temporal_dim).expand(B, N, -1, -1)  # (B, N, T+1, temporal_dim)
        
        # 5. Prepare diffusion timestep encoding
        timestep_encoding = self.diffusion_time_emb[0, timestep, :]  # (B, diffusion_timestep_dim)
        timestep_encoding = timestep_encoding.view(B, 1, 1, self.diffusion_timestep_dim).expand(-1, N, T_tokens, -1)  # (B, N, T+1, diffusion_timestep_dim)
        
        # 6. Combine all encodings using MLP
        combined_encoding = torch.cat([pos_encoding, traj_embedding, temporal_encoding, timestep_encoding], dim=-1)
        # Shape: (B, N, T+1, position_dim + traj_dim + temporal_dim + diffusion_timestep_dim)
        
        # Pass through MLP to combine into single hidden representation
        transformer_input = self.encoding_combiner(combined_encoding)
        # Shape: (B, N, T+1, H)
        
        # 6. Transformer predicts noise via cross-attention to scene
        predicted_noise = self.updateformer(
            transformer_input, 
            scene_tokens, 
            add_space_attn=True
        )
        # Shape: (B, N, T+1, 2) - noise prediction
        
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
        total_steps = self.total_token_steps
        
        # Build x0 = [zero displacement (normalized), future displacements (normalized)]
        zero_disp_normalized = (-self.disp_mean / self.disp_std).to(device=device, dtype=gt_disp_normalized.dtype)
        x0 = torch.zeros(B, N, total_steps, 2, device=device, dtype=gt_disp_normalized.dtype)
        x0[:, :, 0, :] = zero_disp_normalized
        x0[:, :, 1:, :] = gt_disp_normalized
        
        # Condition mask (True where values must stay fixed)
        condition_mask = torch.zeros_like(x0, dtype=torch.bool)
        condition_mask[:, :, 0, :] = True
        
        # Sample random diffusion timestep for each sample in batch
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # Sample noise for full tensor
        noise = torch.randn_like(x0)
        
        # Add noise using scheduler, then reapply conditioning mask (Diffusion Policy style)
        noisy_traj = self.noise_scheduler.add_noise(x0, noise, timestep)
        noisy_traj = torch.where(condition_mask, x0, noisy_traj)
        noise = torch.where(condition_mask, torch.zeros_like(noise), noise)
        
        # Predict the noise
        predicted_noise = self(frames, query_coords, noisy_traj, timestep)
        
        # Compute loss only on unconditioned slots
        loss_mask = (~condition_mask).float()
        masked_pred = predicted_noise * loss_mask
        masked_target = noise * loss_mask
        num_elements = loss_mask.sum().clamp_min(1.0)
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
        total_steps = self.total_token_steps
        device = frames.device
        
        # OPTIMIZATION: Extract vision features ONCE (not at every diffusion step)
        scene_tokens = self.extract_vit_features(frames)  # (B, frame_stack*256, H)
        
        # Conditioning tensors (slot 0 = zero displacement in normalized space)
        zero_disp_normalized = (-self.disp_mean / self.disp_std).to(device=device, dtype=scene_tokens.dtype)
        condition_data = torch.zeros(B, N, total_steps, 2, device=device, dtype=scene_tokens.dtype)
        condition_data[:, :, 0, :] = zero_disp_normalized
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        condition_mask[:, :, 0, :] = True
        
        # Set the number of inference timesteps
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Start from pure noise but respect conditioning mask
        noisy_traj = torch.randn_like(condition_data)
        noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)
        
        # Store intermediate predictions if requested (only unconditioned slots)
        intermediate = [] if return_intermediate else None
        
        # DDIM sampling loop with cached vision features
        for t in self.inference_scheduler.timesteps:
            # Create timestep tensor for batch
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Ensure conditioned slots stay clean before network evaluation
            noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)
            
            # Predict noise using cached scene tokens (pass scene_tokens to skip re-extraction)
            predicted_noise = self(
                frames=None,  # Not needed since we have scene_tokens
                query_coords=query_coords,
                noisy_traj=noisy_traj,
                timestep=timestep,
                scene_tokens=scene_tokens  # Use cached features
            )
            
            # Denoise using scheduler
            noisy_traj = self.inference_scheduler.step(
                predicted_noise, t, noisy_traj
            ).prev_sample
            
            # Reapply conditioning after the update
            noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)
            
            # Store intermediate result if requested (drop conditioned slot)
            if return_intermediate:
                intermediate.append(noisy_traj[:, :, 1:, :].clone())
        
        clean_traj = noisy_traj
        clean_disp = clean_traj[:, :, 1:, :]
        
        if return_intermediate:
            return clean_disp, intermediate
        else:
            return clean_disp

